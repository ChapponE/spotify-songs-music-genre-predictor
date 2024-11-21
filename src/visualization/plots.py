import matplotlib.pyplot as plt
import pandas as pd
import os
from src.utils.config import RESULTS_DIR

def plot_metrics(results, model_type='mlp'):
    model_name = model_type.lower()
    model_results_dir = os.path.join(RESULTS_DIR, model_name)
    os.makedirs(model_results_dir, exist_ok=True)

    # Créer des labels pour chaque configuration de modèle
    configurations = [f"{i+1}" for i in range(len(results))]

    # Extraire les métriques
    train_accuracies = results['average_train_accuracy'].tolist()
    val_accuracies = results['average_val_accuracy'].tolist()

    # Vérifier si les pertes sont disponibles (indiquant un réseau de neurones)
    has_losses = (
        model_type.lower() == 'mlp' and 
        'average_train_loss' in results.columns and 
        'average_val_loss' in results.columns and
        not results['average_train_loss'].isnull().all() and
        not results['average_val_loss'].isnull().all()
    )

    if has_losses:
        # Pour les réseaux de neurones (ex. MLP) : 2 lignes x 2 colonnes de sous-graphiques
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        axs = axs.flatten()  # Aplatir pour faciliter l'accès
    else:
        # Pour les modèles non-entrainés (ex. SVC) : 1 ligne x 2 colonnes de sous-graphiques
        fig, axs = plt.subplots(1, 2, figsize=(14, 5))
        axs = axs.flatten()  # Convertir en liste plate

    # Tracer l'Accuracy d'entraînement
    axs[0].plot(configurations, train_accuracies, marker='o', color='blue', label='Train Accuracy')
    axs[0].set_title('Train Accuracy')
    axs[0].set_xlabel('Configuration de Hyperparamètres')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()
    axs[0].grid(True)
    axs[0].set_xticks(configurations)

    # Tracer l'Accuracy de validation
    axs[1].plot(configurations, val_accuracies, marker='x', color='orange', label='Validation Accuracy')
    axs[1].set_title('Validation Accuracy')
    axs[1].set_xlabel('Configuration de Hyperparamètres')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_xticks(configurations)

    if has_losses:
        # Tracer la Loss d'entraînement
        train_losses = results['average_train_loss'].tolist()
        axs[2].plot(configurations, train_losses, marker='s', color='green', label='Train Loss')
        axs[2].set_title('Train Loss')
        axs[2].set_xlabel('Configuration de Hyperparamètres')
        axs[2].set_ylabel('Loss')
        axs[2].legend()
        axs[2].grid(True)
        axs[2].set_xticks(configurations)

        # Tracer la Loss de validation
        val_losses = results['average_val_loss'].tolist()
        axs[3].plot(configurations, val_losses, marker='^', color='red', label='Validation Loss')
        axs[3].set_title('Validation Loss')
        axs[3].set_xlabel('Configuration de Hyperparamètres')
        axs[3].set_ylabel('Loss')
        axs[3].legend()
        axs[3].grid(True)
        axs[3].set_xticks(configurations)

    # Ajouter un titre général
    plt.suptitle(f"Évolution des Métriques pour chaque Configuration {model_type.upper()}", fontsize=16)

    # Ajuster la disposition
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Sauvegarder et afficher la figure
    plt.savefig(os.path.join(model_results_dir, f'metrics_evolution_{model_type}.png'))
    plt.show()