import pandas as pd
import numpy as np
from pathlib import Path
from train_model import ToxicClassifier
from sklearn.pipeline import Pipeline

def apply_best_model(model_type='logreg', best_params=None):
    """
    Applique le meilleur modèle trouvé sur les données de validation
    et génère un fichier de soumission
    """
    if best_params is None:
        # Paramètres plus conservateurs pour éviter le surapprentissage
        best_params = {
            'max_features': 10000,
            'C': 1.0,  # Régularisation plus forte
            'class_weight': 'balanced',  # Pour gérer le déséquilibre des classes
            'penalty': 'l2',  # L2 est généralement meilleur pour la généralisation
            'solver': 'liblinear',
            'max_iter': 1000
        }

    # Chargement des données
    data_dir = Path('Data')
    train_df = pd.read_csv(data_dir / 'train_train_forum.csv')
    test_df = pd.read_csv(data_dir / 'test_forum.csv')
    
    print("Création du modèle avec les meilleurs paramètres...")
    classifier = ToxicClassifier(model_type=model_type)
    
    # Configuration du vectorizer avec les meilleurs paramètres
    classifier.vectorizer.max_features = best_params['max_features']
    
    # Création du modèle avec les meilleurs paramètres
    from sklearn.linear_model import LogisticRegression
    classifier.model = LogisticRegression(
        C=best_params['C'],
        class_weight=best_params['class_weight'],
        penalty=best_params['penalty'],
        solver=best_params['solver'],
        max_iter=best_params['max_iter']
    )
    
    # Prétraitement et entraînement
    print("Prétraitement des données d'entraînement...")
    X_train_clean = classifier.preprocess_text(train_df['comment_text'])
    print("Prétraitement des données de test...")
    X_test_clean = classifier.preprocess_text(test_df['comment_text'])
    
    # Entraînement de la pipeline
    print("Entraînement du modèle...")
    classifier.pipeline = Pipeline([
        ('vectorizer', classifier.vectorizer),
        ('classifier', classifier.model)
    ])
    
    classifier.pipeline.fit(X_train_clean, train_df['is_toxic'])
    
    # Prédictions sur le jeu de test
    print("Génération des prédictions...")
    predictions = classifier.pipeline.predict(X_test_clean)
    
    # Création du fichier de soumission
    submission_df = pd.DataFrame({
        'is_toxic': predictions,
        'id': test_df['id']
    })[['is_toxic', 'id']]  # Réorganisation des colonnes dans le bon ordre
    
    # Sauvegarde du fichier de soumission
    submission_df.to_csv(data_dir / 'submission.csv', index=False)
    
    # Affichage des statistiques
    print("\nStatistiques des prédictions:")
    print(f"Nombre total de commentaires: {len(predictions)}")
    print(f"Nombre de commentaires toxiques: {sum(predictions)}")
    print(f"Pourcentage de commentaires toxiques: {(sum(predictions)/len(predictions))*100:.2f}%")
    
    return submission_df

if __name__ == "__main__":
    print("Application du meilleur modèle trouvé (LogisticRegression)")
    submission_df = apply_best_model(model_type='logreg')
    
    print("\nFichier de soumission créé avec succès!")
    print(f"Chemin: Data/submission.csv") 