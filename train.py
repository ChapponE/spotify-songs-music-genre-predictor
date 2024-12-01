
# train.py

import argparse
import sys
from train import train_mlp, train_rf, train_svc, train_mlp_full

def main():
    parser = argparse.ArgumentParser(description="Script principal pour entraîner différents modèles.")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['mlp', 'rf', 'svc', 'mlp_full'],
        help="Nom du modèle à entraîner. Options : 'mlp', 'rf', 'svc', 'mlp_full'."
    )
    
    args = parser.parse_args()
    
    model_name = args.model.lower()
    
    if model_name == 'mlp':
        train_mlp.train()
    elif model_name == 'rf':
        train_rf.train()
    elif model_name == 'svc':
        train_svc.train()
    elif model_name == 'mlp_full':
        train_mlp_full.train()
    else:
        print(f"Modèle '{model_name}' non reconnu.")
        sys.exit(1)
    
    print(f"Entraînement du modèle '{model_name}' terminé avec succès.")

if __name__ == "__main__":
    main()