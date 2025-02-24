import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.metrics import f1_score
import optuna
from sklearn.pipeline import Pipeline
import joblib
import re
import string

class ToxicClassifier:
    def __init__(self, model_type='logreg'):
        """
        Args:
            model_type (str): 'logreg', 'rf' ou 'lgbm'
        """
        self.model_type = model_type
        # Ajout de stop words personnalisés
        custom_stop_words = {'www', 'http', 'https', 'com', 'html', 'htm'} 
        self.stop_words = list(ENGLISH_STOP_WORDS.union(custom_stop_words))
        
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            strip_accents='unicode',
            lowercase=True,
            stop_words=self.stop_words,
            token_pattern=r'\b\w+\b'  # Capture les mots uniquement
        )
    
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
        
        # Suppression des caractères répétés (ex: 'hellooooo' -> 'hello')
        text = re.sub(r'(.)\1{2,}', r'\1', text)
        
        # Suppression des lignes qui ne contiennent que des espaces
        text = text.strip()
        
        # Si après nettoyage le texte est vide, retourner un espace
        if not text:
            return ' '
            
        return text
    
    def preprocess_text(self, texts):
        """Applique le nettoyage à une série de textes"""
        return texts.apply(self.clean_text)
        
    def objective(self, trial):
        # Paramètres communs pour TF-IDF
        self.vectorizer.max_features = trial.suggest_int('max_features', 1000, 20000)
        
        # Prétraitement des données
        X_train_clean = self.preprocess_text(self.X_train)
        X_valid_clean = self.preprocess_text(self.X_valid)
        
        # Paramètres spécifiques selon le modèle
        if self.model_type == 'logreg':
            params = {
                'C': trial.suggest_float('C', 1e-4, 1e4, log=True),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
                'max_iter': 1000
            }
            model = LogisticRegression(**params)
            
        elif self.model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
            }
            model = RandomForestClassifier(**params)
            
        else:  # lightgbm
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True)
            }
            model = lgb.LGBMClassifier(**params)
        
        # Transformation et entraînement
        X_train_vec = self.vectorizer.fit_transform(X_train_clean)
        X_valid_vec = self.vectorizer.transform(X_valid_clean)
        
        model.fit(X_train_vec, self.y_train)
        y_pred = model.predict(X_valid_vec)
        
        return f1_score(self.y_valid, y_pred)
    
    def train(self, X_train, y_train, X_valid, y_valid, n_trials=100):
        """
        Entraîne le modèle avec optimisation des hyperparamètres
        """
        # Prétraitement des données
        self.X_train = self.preprocess_text(X_train)
        self.y_train = y_train
        self.X_valid = self.preprocess_text(X_valid)
        self.y_valid = y_valid
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Entraînement final avec les meilleurs paramètres
        if self.model_type == 'logreg':
            self.model = LogisticRegression(**study.best_params, max_iter=1000)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(**study.best_params)
        else:
            self.model = lgb.LGBMClassifier(**study.best_params)
            
        # Pipeline final avec prétraitement
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        self.pipeline.fit(self.X_train, self.y_train)
        return study.best_value, study.best_params
    
    def predict(self, X):
        """Prédiction sur de nouvelles données"""
        X_clean = self.preprocess_text(X)
        return self.pipeline.predict(X_clean)
    
    def save_model(self, path):
        """Sauvegarde le modèle"""
        joblib.dump(self.pipeline, path)
    
    @classmethod
    def load_model(cls, path):
        """Charge un modèle sauvegardé"""
        return joblib.load(path)

if __name__ == "__main__":
    # Chargement des données
    data_dir = Path('Data')
    train_df = pd.read_csv(data_dir / 'train_train_forum.csv')
    valid_df = pd.read_csv(data_dir / 'train_valid_forum.csv')
    
    # Affichage de quelques statistiques sur le nettoyage
    classifier = ToxicClassifier()
    original_texts = train_df['comment_text'].iloc[:5]
    cleaned_texts = classifier.preprocess_text(original_texts)
    
    print("Exemple de nettoyage des textes:")
    for orig, clean in zip(original_texts, cleaned_texts):
        print("\nOriginal:", orig[:100], "...")
        print("Nettoyé:", clean[:100], "...")
    
    # Entraînement des différents modèles
    models = ['logreg', 'rf', 'lgbm']
    results = {}
    
    for model_type in models:
        print(f"\nEntraînement du modèle {model_type}")
        classifier = ToxicClassifier(model_type=model_type)
        
        f1, params = classifier.train(
            train_df['comment_text'], 
            train_df['is_toxic'],
            valid_df['comment_text'], 
            valid_df['is_toxic'],
            n_trials=50  # Réduire pour les tests
        )
        
        results[model_type] = {
            'f1_score': f1,
            'best_params': params
        }
        
        # Sauvegarde du modèle
        classifier.save_model(data_dir / f'toxic_classifier_{model_type}.joblib')
    
    # Affichage des résultats
    print("\nRésultats finaux:")
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"F1 Score: {result['f1_score']:.4f}")
        print("Meilleurs paramètres:", result['best_params']) 