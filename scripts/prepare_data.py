import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys

# Ensure parent directory is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.filters import hybrid_preprocess
from feature.texture import extract_haralick, extract_lbp

# --- PATH CONFIGURATION ---
INPUT_CSV = "data/train.csv"          # Original dataset CSV
INPUT_IMG_DIR = "data/train_images/"   # Raw retinal images
OUTPUT_DIR = "processed_data"          # Output directory
PROCESSED_IMG_DIR = os.path.join(OUTPUT_DIR, "images")

# Create output directories
os.makedirs(PROCESSED_IMG_DIR, exist_ok=True)

def main():
    """
    Main preprocessing pipeline:
    1. Reads raw images from the dataset.
    2. Applies Hybrid Preprocessing (Ben Graham's method + CLAHE).
    3. Extracts 16 texture biomarkers (Haralick + LBP).
    4. Saves processed images and feature vectors for training.
    """
    print(f"[*] Loading metadata from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    feature_list = []
    processed_paths = []
    
    print("[*] Starting Preprocessing and Feature Extraction...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row['id_code']
        img_path = os.path.join(INPUT_IMG_DIR, f"{img_id}.png")
        
        if not os.path.exists(img_path):
            continue

        # --- Step A: Preprocessing ---
        img_color = hybrid_preprocess(img_path, img_size=256)
        
        # Save processed image for the Deep Semantic Stream (ResNet)
        save_name = f"{img_id}_proc.png"
        save_path = os.path.join(PROCESSED_IMG_DIR, save_name)
        cv2.imwrite(save_path, cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR))
        processed_paths.append(save_path)
        
        # --- Step B: Texture Extraction ---
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        
        # Extract features (16-dimensional vector)
        h_feat = extract_haralick(img_gray)
        l_feat = extract_lbp(img_gray)
        
        full_vector = h_feat + l_feat
        feature_list.append(full_vector)

    # --- Step 3: Final Export ---
    print("\n[*] Saving processed data to disk...")
    
    # Save texture features as a binary numpy file for efficient loading
    features_np = np.array(feature_list)
    np.save(os.path.join(OUTPUT_DIR, "texture_features.npy"), features_np)
    
    # Update DataFrame with processed paths and export new CSV
    df = df.iloc[:len(processed_paths)].copy() # Align in case of missing files
    df['processed_path'] = processed_paths
    df.to_csv(os.path.join(OUTPUT_DIR, "processed_metadata.csv"), index=False)
    
    print(f"✅ Success! Processed data ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
