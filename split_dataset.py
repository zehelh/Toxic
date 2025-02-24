import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def split_train_dataset(input_file, train_size=0.05, random_state=42):
    """
    Split le dataset d'entraînement en train et validation
    
    Args:
        input_file (str): Chemin vers le fichier train original
        train_size (float): Proportion des données pour l'entraînement (entre 0 et 1)
        random_state (int): Seed pour la reproductibilité
    """
    # Création du dossier Data s'il n'existe pas
    data_dir = Path('Data')
    data_dir.mkdir(exist_ok=True)
    
    # Lecture du fichier original
    df = pd.read_csv(data_dir / input_file)
    
    # Split aléatoire
    np.random.seed(random_state)
    mask = np.random.rand(len(df)) < train_size
    
    # Création des nouveaux dataframes
    train_df = df[mask]
    valid_df = df[~mask]
    
    # Sauvegarde des fichiers
    train_df.to_csv(data_dir / 'train_train_forum.csv', index=False)
    valid_df.to_csv(data_dir / 'train_valid_forum.csv', index=False)
    
    # Affichage des informations
    print(f"Dataset original: {len(df)} exemples")
    print(f"Dataset train: {len(train_df)} exemples ({train_size*100:.1f}%)")
    print(f"Dataset validation: {len(valid_df)} exemples ({(1-train_size)*100:.1f}%)")
    
    return train_df, valid_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split le dataset train en train et validation')
    parser.add_argument('--train-size', type=float, default=0.05,
                      help='Proportion des données pour l\'entraînement (default: 0.05)')
    parser.add_argument('--random-state', type=int, default=42,
                      help='Seed pour la reproductibilité (default: 42)')
    parser.add_argument('--input-file', type=str, default='train_forum.csv',
                      help='Nom du fichier d\'entrée (default: train_forum.csv)')
    
    args = parser.parse_args()
    
    try:
        train_df, valid_df = split_train_dataset(
            args.input_file,
            train_size=args.train_size,
            random_state=args.random_state
        )
        
        # Vérification rapide des données
        print("\nAperçu du train:")
        print(train_df['is_toxic'].value_counts(normalize=True))
        print("\nAperçu du valid:")
        print(valid_df['is_toxic'].value_counts(normalize=True))
        
    except Exception as e:
        print(f"Une erreur s'est produite: {e}") 