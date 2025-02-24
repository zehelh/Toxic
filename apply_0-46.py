import pandas as pd
import numpy as np
from pathlib import Path
from train_model import ToxicClassifier

def apply_best_model(model_type='logreg', best_params=None):
    """
    Applique le meilleur modèle trouvé sur les données de validation
    et génère un fichier de soumission
    """
    if best_params is None:
        # Meilleurs paramètres trouvés lors des essais
        best_params = {
            'max_features': 14096,
            'C': 246.9345540162386,
            'class_weight': 'balanced'
        }

    # Chargement des données
    data_dir = Path('Data')
    train_df = pd.read_csv(data_dir / 'train_train_forum.csv')
    test_df = pd.read_csv(data_dir / 'test_forum.csv')
    
    # Création et entraînement du modèle avec les meilleurs paramètres
    classifier = ToxicClassifier(model_type=model_type)
    
    # Configuration du vectorizer avec les meilleurs paramètres
    classifier.vectorizer.max_features = best_params['max_features']
    
    # Création du modèle avec les meilleurs paramètres
    if model_type == 'logreg':
        from sklearn.linear_model import LogisticRegression
        classifier.model = LogisticRegression(
            C=best_params['C'],
            class_weight=best_params['class_weight'],
            max_iter=1000
        )
    
    # Prétraitement et entraînement
    X_train_clean = classifier.preprocess_text(train_df['comment_text'])
    X_test_clean = classifier.preprocess_text(test_df['comment_text'])
    
    # Création et entraînement de la pipeline
    classifier.pipeline.fit(X_train_clean, train_df['is_toxic'])
    
    # Prédictions sur le jeu de test
    predictions = classifier.pipeline.predict(X_test_clean)
    
    # Création du fichier de soumission
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'is_toxic': predictions
    })
    
    # Sauvegarde du fichier de soumission
    submission_df.to_csv(data_dir / 'submission.csv', index=False)
    
    # Affichage des statistiques
    print("\nStatistiques des prédictions:")
    print(f"Nombre total de commentaires: {len(predictions)}")
    print(f"Nombre de commentaires toxiques: {sum(predictions)}")
    print(f"Pourcentage de commentaires toxiques: {(sum(predictions)/len(predictions))*100:.2f}%")
    
    return submission_df

if __name__ == "__main__":
    # Application du meilleur modèle (LogisticRegression)
    best_params = {
        'max_features': 14096,
        'C': 246.9345540162386,
        'class_weight': 'balanced'
    }
    
    print("Application du meilleur modèle trouvé (LogisticRegression)")
    submission_df = apply_best_model(
        model_type='logreg',
        best_params=best_params
    )
    
    print("\nFichier de soumission créé avec succès!")
    print(f"Chemin: Data/submission.csv")