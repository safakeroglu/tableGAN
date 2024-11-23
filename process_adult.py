import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pickle
import os

def preprocess_adult_dataset(input_file, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading file: {input_file}")
    df = pd.read_csv(input_file)
    print("Data loaded successfully")
    print("Original data shape:", df.shape)
    
    # Separate features and labels
    X = df.iloc[:, :-1].values  # All columns except the last one
    y = df.iloc[:, -1].values   # Last column (native-country) as label
    
    # Normalize features to [-1, 1] range
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)
    
    # Convert labels to binary (0 or 1)
    # Using median split for binarization
    median_value = np.median(y)
    y_binary = (y > median_value).astype(int)
    
    print("\nBefore saving:")
    print(f"Features shape: {X_scaled.shape}")
    print(f"Labels shape: {y_binary.shape}")
    print(f"Unique labels after binarization: {np.unique(y_binary)}")
    print(f"Label distribution: {np.bincount(y_binary)}")
    
    # Save preprocessed data
    with open(os.path.join(output_dir, 'train_Adult_cleaned.pickle'), 'wb') as f:
        pickle.dump(X_scaled, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(os.path.join(output_dir, 'train_Adult_labels.pickle'), 'wb') as f:
        pickle.dump(y_binary, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"\nFiles saved in: {output_dir}")
    
    # Save scaler for later use
    with open(os.path.join(output_dir, 'scaler.pickle'), 'wb') as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    input_file = "data/Adult/train_Adult_cleaned.csv"
    output_dir = "data/Adult"
    preprocess_adult_dataset(input_file, output_dir)