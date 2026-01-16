import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Importation de tes propres fonctions (assure-toi que les dossiers sont bien dans ton PYTHONPATH)
from preprocessing.filters import hybrid_preprocess
from features.texture import extract_haralick, extract_lbp

# --- CONFIGURATION DES CHEMINS ---
# Modifie ces chemins pour qu'ils correspondent à ton installation locale ou Kaggle
INPUT_CSV = "data/train.csv"          # Ton CSV original
INPUT_IMG_DIR = "data/train_images/"   # Tes images brutes
OUTPUT_DIR = "processed_data"          # Où sauvegarder les résultats
PROCESSED_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

# Création des dossiers de sortie
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)

def main():
    # 1. Chargement du CSV
    print(f"Loading metadata from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Listes pour stocker les résultats
    feature_list = []
    processed_paths = []
    
    print("Starting Preprocessing and Feature Extraction (Haralick + LBP)...")
    
    # 2. Boucle de traitement (Utilisation de tqdm pour voir la barre de progression)
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['id_code']
        img_path = os.path.join(INPUT_IMG_DIR, f"{img_id}.png")
        
        # --- Étape A : Preprocessing (Ben Graham + CLAHE) ---
        # Utilise TA fonction hybrid_preprocess
        img_color = hybrid_preprocess(img_path, img_size=256)
        
        # Sauvegarde de l'image traitée pour le Deep Learning (ResNet)
        save_name = f"{img_id}_proc.png"
        save_path = os.path.join(PROCESSED_IMG_DIR, save_name)
        cv2.imwrite(save_path, cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))
        processed_paths.append(save_path)
        
        # --- Étape B : Extraction de Texture ---
        # Convertir en gris pour Haralick et LBP
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        
        # Utilise TES fonctions extract_haralick et extract_lbp
        h_feat = extract_haralick(img_gray)
        l_feat = extract_lbp(img_gray)
        
        # Fusion des deux vecteurs (6 + 10 = 16 dimensions)
        full_vector = h_feat + l_feat
        feature_list.append(full_vector)

    # 3. Sauvegarde finale
    print("\nSaving processed data...")
    
    # Sauvegarde des vecteurs de texture en format binaire (très rapide à charger)
    features_np = np.array(feature_list)
    np.save(os.path.join(OUTPUT_DIR, "texture_features.npy"), features_np)
    
    # Sauvegarde d'un nouveau CSV avec les chemins vers les images filtrées
    df['processed_path'] = processed_paths
    df.to_csv(os.path.join(OUTPUT_DIR, "processed_metadata.csv"), index=False)
    
    print(f"✅ Success! Processed images in: {PROCESSED_IMG_DIR}")
    print(f"✅ Success! Texture features (Shape {features_np.shape}) in: texture_features.npy")

if __name__ == "__main__":
    main()
