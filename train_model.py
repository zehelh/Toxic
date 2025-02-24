import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb  # Remplacer xgboost par lightgbm
from sklearn.metrics import f1_score
import optuna
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import re
import warnings
import joblib
from toxic_words import TOXIC_WORDS, TOXIC_CATEGORIES
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.sparse import hstack
from scipy.sparse import csr_matrix as sparse

warnings.filterwarnings('ignore')

class ToxicClassifier:
    def __init__(self, model_type='logreg'):
        """
        Args:
            model_type (str): 'logreg', 'rf' ou 'lgbm'
        """
        self.model_type = model_type
        self.n_jobs = -1
        self.toxic_words = TOXIC_WORDS
        self.toxic_categories = TOXIC_CATEGORIES
        
        # Retour aux paramètres qui marchaient bien
        custom_stop_words = {'www', 'http', 'https', 'com', 'html', 'htm'} 
        self.stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # Réduit
            strip_accents='unicode',
            lowercase=True,
            stop_words=self.stop_words,
            token_pattern=r'\b\w+\b',
            ngram_range=(1, 1)  # Uniquement unigrammes
        )
    
    def get_toxic_features(self, text):
        """Extrait les features basées sur les mots toxiques"""
        # Gestion des valeurs manquantes
        if pd.isna(text) or not isinstance(text, str):
            return {
                'toxic_word_count': 0.0,  # Valeurs explicitement en float
                'toxic_word_ratio': 0.0,
                **{f"{cat}_count": 0.0 for cat in self.toxic_categories.keys()},
                **{f"{cat}_ratio": 0.0 for cat in self.toxic_categories.keys()}
            }
        
        text = text.lower()
        words = set(re.findall(r'\b\w+\b', text))
        
        # Features globales
        toxic_count = float(len(words.intersection(self.toxic_words)))  # Conversion en float
        total_words = float(len(words)) if len(words) > 0 else 1.0
        toxic_ratio = toxic_count / total_words
        
        # Features par catégorie
        category_counts = {
            f"{cat}_count": float(len(words.intersection(word_set)))  # Conversion en float
            for cat, word_set in self.toxic_categories.items()
        }
        
        category_ratios = {
            f"{cat}_ratio": float(count) / total_words  # Assure que la division donne un float
            for cat, count in category_counts.items()
        }
        
        # Combinaison de toutes les features
        features = {
            'toxic_word_count': toxic_count,
            'toxic_word_ratio': toxic_ratio,
            **category_counts,
            **category_ratios
        }
        
        return features  # Retourne un dictionnaire au lieu d'une Series
    
    def clean_text(self, text):
        """Nettoie le texte en profondeur"""
        if pd.isna(text) or not isinstance(text, str):
            return ''
            
        # Conversion en minuscules
        text = text.lower()
        
        # Suppression des URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Suppression des caractères spéciaux et nombres seuls
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text)
        
        # Suppression des caractères répétés
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        return text.strip() or ' '
    
    def preprocess_text(self, texts):
        """Prétraite les textes et extrait les features toxiques"""
        # Nettoyage des valeurs NaN et conversion en string
        texts = texts.fillna('').astype(str)
        
        # Nettoyage du texte
        cleaned_texts = texts.apply(self.clean_text)
        
        # Extraction des features toxiques
        features_list = [self.get_toxic_features(text) for text in texts]
        toxic_features = pd.DataFrame(features_list)  # Les colonnes seront automatiquement en float
        
        return cleaned_texts, toxic_features
    
    def objective(self, trial):
        try:
            params = {
                'max_features': trial.suggest_int('max_features', 1000, 5000),
                'C': trial.suggest_float('C', 0.5, 5.0, log=True),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced']),
                'max_iter': trial.suggest_int('max_iter', 50, 100)
            }
            
            # Paramètres spécifiques selon le modèle
            if self.model_type == 'logreg':
                model = LogisticRegression(
                    C=params['C'],
                    class_weight=params['class_weight'],
                    max_iter=params['max_iter'],
                    random_state=42,
                    solver='saga',
                    n_jobs=self.n_jobs
                )
            elif self.model_type == 'rf':
                model = RandomForestClassifier(**params)
            else:  # lightgbm
                model = lgb.LGBMClassifier(**params)
            
            # Transformation sans conversion dense
            X_train_tfidf = self.vectorizer.fit_transform(self.X_train_text)
            X_valid_tfidf = self.vectorizer.transform(self.X_valid_text)
            
            # Features toxiques en sparse
            X_train_toxic = sparse.csr_matrix(self.X_train_toxic.astype(float))
            X_valid_toxic = sparse.csr_matrix(self.X_valid_toxic.astype(float))
            
            # Concaténation sparse
            X_train_combined = hstack([X_train_tfidf, X_train_toxic])
            X_valid_combined = hstack([X_valid_tfidf, X_valid_toxic])
            
            # Entraînement et prédiction
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                model.fit(X_train_combined, self.y_train)
            
            y_pred = model.predict(X_valid_combined)
            score = f1_score(self.y_valid, y_pred)
            
            print(f"Essai {trial.number + 1} - Score: {score:.4f} - Params: {params}")
            return score
            
        except Exception as e:
            print(f"Erreur détaillée:", e)
            import traceback
            traceback.print_exc()
            return 0.0
    
    def train(self, X_train, y_train, X_valid, y_valid, n_trials=100):
        print("Prétraitement des données...")
        print(f"Taille du jeu d'entraînement: {len(X_train)} exemples")
        print(f"Taille du jeu de validation initial: {len(X_valid)} exemples")
        
        # Réduction plus agressive du jeu de validation
        if len(X_valid) > 5000:
            print("Réduction du jeu de validation à 5000 exemples...")
            self.valid_idx = np.random.choice(len(X_valid), 5000, replace=False)
            X_valid = X_valid.iloc[self.valid_idx]
            y_valid = y_valid.iloc[self.valid_idx]
        
        # Prétraitement des données
        self.X_train_text, self.X_train_toxic = self.preprocess_text(X_train)
        self.y_train = y_train
        
        self.X_valid_text, self.X_valid_toxic = self.preprocess_text(X_valid)
        self.y_valid = y_valid
        
        print(f"Taille du jeu de validation final: {len(X_valid)} exemples")
        
        print("\nDébut de l'optimisation...")
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        # Entraînement final
        print("\nEntraînement du modèle final...")
        self.pipeline = self.create_final_pipeline(study.best_params)
        
        # Création du DataFrame final avec nettoyage des NaN
        X_train_final = pd.DataFrame({
            'text': self.X_train_text.fillna('').astype(str),
            **{str(col): self.X_train_toxic[col] for col in self.X_train_toxic.columns}
        })
        
        self.pipeline.fit(X_train_final, self.y_train)
        
        # Utilisation des mêmes données de validation que pendant l'optimisation
        X_valid_final = pd.DataFrame({
            'text': self.X_valid_text.fillna('').astype(str),
            **{str(col): self.X_valid_toxic[col] for col in self.X_valid_toxic.columns}
        })
        
        y_pred = self.pipeline.predict(X_valid_final)
        final_score = f1_score(self.y_valid, y_pred)
        
        print("\nMeilleurs paramètres:", study.best_params)
        print(f"Meilleur score pendant l'optimisation: {study.best_value:.4f}")
        print(f"Score final sur la validation: {final_score:.4f}")
        
        return study.best_value, study.best_params
    
    def create_final_pipeline(self, best_params):
        """Crée un pipeline final avec les meilleurs paramètres"""
        self.vectorizer.set_params(max_features=best_params['max_features'])
        if self.model_type == 'logreg':
            self.model = LogisticRegression(
                C=best_params['C'],
                class_weight=best_params['class_weight'],
                max_iter=best_params['max_iter'],
                random_state=42,
                solver='saga',
                n_jobs=self.n_jobs
            )
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=best_params['n_estimators'],
                max_depth=best_params['max_depth'],
                min_samples_split=best_params['min_samples_split'],
                min_samples_leaf=best_params['min_samples_leaf'],
                class_weight=best_params['class_weight'],
                random_state=42,
                n_jobs=self.n_jobs
            )
        else:
            self.model = lgb.LGBMClassifier(**best_params)
        
        class CombinedFeatureTransformer:
            def __init__(self, vectorizer, scaler):
                self.vectorizer = vectorizer
                self.scaler = scaler
            
            def fit(self, X, y=None):
                # Nettoyage et conversion des textes
                text = X['text'].fillna('').astype(str).values
                toxic_features = X.drop('text', axis=1).astype(float)
                
                # Transformation sparse
                self.vectorizer.fit(text)
                self.scaler.fit(toxic_features)
                return self
            
            def transform(self, X):
                # Nettoyage et conversion des textes
                text = X['text'].fillna('').astype(str).values
                toxic_features = X.drop('text', axis=1).astype(float)
                
                # Transformations sparse
                text_features = self.vectorizer.transform(text)
                toxic_features_scaled = sparse.csr_matrix(self.scaler.transform(toxic_features))
                
                # Concaténation sparse
                return hstack([text_features, toxic_features_scaled])
        
        transformer = CombinedFeatureTransformer(self.vectorizer, StandardScaler())
        
        return Pipeline([
            ('features', transformer),
            ('classifier', self.model)
        ])
    
    def save_model(self, path):
        joblib.dump(self.pipeline, path)

if __name__ == "__main__":
    # Configuration
    n_jobs = min(8, joblib.cpu_count() - 1)  # Réduit de 6 à 4 cœurs
    print(f"Utilisation de {n_jobs} cœurs")
    
    # Chargement des données
    print("Chargement des données...")
    data_dir = Path('Data')
    train_df = pd.read_csv(data_dir / 'train_train_forum.csv')
    valid_df = pd.read_csv(data_dir / 'train_valid_forum.csv')
    
    # Entraînement avec moins d'essais
    models = ['logreg']  # On commence par tester uniquement logreg
    for model_type in models:
        print(f"\nEntraînement du modèle {model_type}...")
        classifier = ToxicClassifier(model_type=model_type)
        f1, params = classifier.train(
            train_df['comment_text'],
            train_df['is_toxic'],
            valid_df['comment_text'],
            valid_df['is_toxic'],
            n_trials=3  # Réduit de 5 à 3
        )
        
        # Sauvegarde du modèle
        classifier.save_model(data_dir / f'toxic_classifier_{model_type}.joblib') 