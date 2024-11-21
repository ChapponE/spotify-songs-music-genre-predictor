import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from src.data.data_loader import load_data, get_numerical_features
from src.utils.config import PROCESSED_DIR, RESULTS_DIR, TEST_CSV

def main():
    # Charger les données de test
    _, test_df = load_data()

    # Charger les objets de prétraitement
    scaler = joblib.load(os.path.join(PROCESSED_DIR, 'scaler.pkl'))
    label_encoder = joblib.load(os.path.join(PROCESSED_DIR, 'label_encoder.pkl'))

    # Sélectionner les features numériques
    X_test = get_numerical_features(test_df)

    # Appliquer la normalisation
    X_test_scaled = scaler.transform(X_test)

    # Charger le meilleur modèle SVM
    svc_path = os.path.join(RESULTS_DIR, 'svc')
    model_svc_path = os.path.join(svc_path, 'best_svc_model.pkl')
    model_svc = joblib.load(model_svc_path)

    # Prédire les genres avec le SVM
    predicted_classes_svc = model_svc.predict(X_test_scaled)
    predicted_labels_svc = label_encoder.inverse_transform(predicted_classes_svc)

    # Ajouter les prédictions au DataFrame de test
    test_df['predicted_genre_svc'] = predicted_labels_svc

    # Sauvegarder les prédictions
    prediction_path = os.path.join(RESULTS_DIR, 'test_predictions_svc.csv')
    test_df.to_csv(prediction_path, index=False)
    print(f"Les prédictions SVM ont été sauvegardées dans '{prediction_path}'.")

    # Charger les données d'entraînement pour obtenir les vrais labels
    train_df, _ = load_data()
    train_genres = train_df['playlist_genre'].unique()

    # Filtrer les prédictions pour n'inclure que les genres présents dans l'ensemble d'entraînement
    mask = test_df['playlist_genre'].isin(train_genres)
    filtered_test = test_df[mask]

    # Calculer l'accuracy
    correct_predictions = (filtered_test['playlist_genre'] == filtered_test['predicted_genre_svc']).sum()
    total_predictions = len(filtered_test)
    accuracy = correct_predictions / total_predictions

    print(f"\nAccuracy sur l'ensemble de test: {accuracy:.4f}")

    # Charger les données de test avec les vraies étiquettes
    test_df_with_labels = pd.read_csv(TEST_CSV, sep='\t')
    true_labels = test_df_with_labels['playlist_genre']

    # Calculer l'accuracy
    accuracy = accuracy_score(true_labels, predicted_labels_svc)
    print(f"\nAccuracy du modèle SVC: {accuracy:.2%}")

    # Calculer et afficher la matrice de confusion
    conf_matrix = confusion_matrix(true_labels, predicted_labels_svc)
    
    # Créer une figure plus grande
    plt.figure(figsize=(12, 8))
    
    # Créer la heatmap
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    
    plt.title('Matrice de Confusion - SVC')
    plt.xlabel('Prédictions')
    plt.ylabel('Vraies Étiquettes')
    
    # Rotation des labels pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Ajuster le layout pour éviter que les labels soient coupés
    plt.tight_layout()
    
    # Sauvegarder la matrice de confusion
    confusion_matrix_path = os.path.join(RESULTS_DIR, 'svc', 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    print(f"La matrice de confusion a été sauvegardée dans '{confusion_matrix_path}'")
    plt.close()

if __name__ == "__main__":
    main()