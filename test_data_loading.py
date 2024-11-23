import pickle
import os
import numpy as np

def test_data_loading():
    data_path = 'data/Adult'
    y_dim = 2  # This should match your FLAGS.y_dim
    
    # Load features
    with open(os.path.join(data_path, 'train_Adult_cleaned.pickle'), 'rb') as f:
        data_X = pickle.load(f)
    
    # Load labels
    with open(os.path.join(data_path, 'train_Adult_labels.pickle'), 'rb') as f:
        data_y = pickle.load(f)
    
    print("Data loaded successfully!")
    print(f"Features shape: {data_X.shape}")
    print(f"Labels shape: {data_y.shape}")
    print(f"Unique labels: {np.unique(data_y)}")
    
    # Test one-hot encoding
    data_y_onehot = np.eye(y_dim)[data_y]
    print(f"\nOne-hot encoded labels shape: {data_y_onehot.shape}")
    print("Sample one-hot labels (first 3 rows):")
    print(data_y_onehot[:3])

if __name__ == "__main__":
    test_data_loading()