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
import joblib

class ToxicClassifier:
    def __init__(self, model_type='logreg'):
        """
        Args:
            model_type (str): 'logreg', 'rf' ou 'xgb'
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            strip_accents='unicode',
            lowercase=True
        )
        
    def objective(self, trial):
        # Paramètres communs pour TF-IDF
        self.vectorizer.max_features = trial.suggest_int('max_features', 1000, 20000)
        
        # Paramètres spécifiques selon le modèle
        if self.model_type == 'logreg':
            params = {
                'C': trial.suggest_loguniform('C', 1e-4, 1e4),
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
            
        else:  # xgboost
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
                'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params)
        
        # Transformation et entraînement
        X_train_vec = self.vectorizer.fit_transform(self.X_train)
        X_valid_vec = self.vectorizer.transform(self.X_valid)
        
        model.fit(X_train_vec, self.y_train)
        y_pred = model.predict(X_valid_vec)
        
        return f1_score(self.y_valid, y_pred)
    
    def train(self, X_train, y_train, X_valid, y_valid, n_trials=100):
        """
        Entraîne le modèle avec optimisation des hyperparamètres
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Entraînement final avec les meilleurs paramètres
        if self.model_type == 'logreg':
            self.model = LogisticRegression(**study.best_params, max_iter=1000)
        elif self.model_type == 'rf':
            self.model = RandomForestClassifier(**study.best_params)
        else:
            self.model = xgb.XGBClassifier(**study.best_params)
            
        # Pipeline final
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.model)
        ])
        
        self.pipeline.fit(X_train, y_train)
        return study.best_value, study.best_params
    
    def predict(self, X):
        """Prédiction sur de nouvelles données"""
        return self.pipeline.predict(X)
    
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
    
    # Entraînement des différents modèles
    models = ['logreg', 'rf', 'xgb']
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