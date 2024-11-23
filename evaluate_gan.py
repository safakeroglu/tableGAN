# evaluate_gan.py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from model import TableGan  # Import your GAN model

def generate_synthetic_data(model_path, num_samples=2000):
    """Generate synthetic data using the trained GAN"""
    print("Loading model and generating synthetic data...")
    
    # Initialize the model
    tablegan = TableGan(
        input_width=7,
        input_height=7,
        output_width=7,
        output_height=7,
        batch_size=100,
        y_dim=2,
        dataset_name='Adult',
        checkpoint_dir='checkpoint/Adult/OI_11_00'
    )
    
    # Load the trained weights
    tablegan.load_weights(model_path)
    
    # Generate random noise and labels
    noise = tf.random.normal([num_samples, 100], dtype=tf.float32)
    labels = tf.random.uniform([num_samples], 0, 2, dtype=tf.int32)
    labels_onehot = tf.one_hot(labels, depth=2)
    
    # Generate synthetic data
    generator_input = tf.concat([noise, labels_onehot], axis=1)
    synthetic_data = tablegan.generator(generator_input, training=False)
    
    print(f"Generated {num_samples} synthetic samples")
    return synthetic_data.numpy(), labels.numpy(), tablegan.data_X.numpy(), tablegan.data_y_normal

def evaluate_synthetic_data(real_X, real_y, synthetic_X, synthetic_y):
    """Compare statistical properties of real and synthetic data"""
    print("\n=== Statistical Evaluation ===")
    print("Data Shapes:")
    print(f"Real data: {real_X.shape}, Synthetic data: {synthetic_X.shape}")
    
    # Compare means
    real_means = np.mean(real_X, axis=0)
    synthetic_means = np.mean(synthetic_X, axis=0)
    print("\nFeature Means Comparison:")
    for i, (real_mean, syn_mean) in enumerate(zip(real_means, synthetic_means)):
        print(f"Feature {i}: Real = {real_mean:.4f}, Synthetic = {syn_mean:.4f}")
        print(f"  Difference: {abs(real_mean - syn_mean):.4f}")
    
    # Compare standard deviations
    real_stds = np.std(real_X, axis=0)
    synthetic_stds = np.std(synthetic_X, axis=0)
    print("\nFeature Standard Deviations Comparison:")
    for i, (real_std, syn_std) in enumerate(zip(real_stds, synthetic_stds)):
        print(f"Feature {i}: Real = {real_std:.4f}, Synthetic = {syn_std:.4f}")
        print(f"  Difference: {abs(real_std - syn_std):.4f}")
    
    # Compare label distributions
    real_label_dist = np.bincount(real_y) / len(real_y)
    synthetic_label_dist = np.bincount(synthetic_y) / len(synthetic_y)
    print("\nLabel Distribution Comparison:")
    for i, (real_freq, syn_freq) in enumerate(zip(real_label_dist, synthetic_label_dist)):
        print(f"Label {i}: Real = {real_freq:.4f}, Synthetic = {syn_freq:.4f}")
        print(f"  Difference: {abs(real_freq - syn_freq):.4f}")

def visualize_distributions(real_X, synthetic_X, output_dir, feature_names=None):
    """Create distribution plots comparing real and synthetic data"""
    print("\n=== Generating Visualizations ===")
    n_features = real_X.shape[1]
    n_rows = (n_features + 2) // 3
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall distribution plot
    plt.figure(figsize=(15, 5*n_rows))
    for i in range(n_features):
        plt.subplot(n_rows, 3, i+1)
        sns.kdeplot(data=real_X[:, i], label='Real', color='blue', alpha=0.5)
        sns.kdeplot(data=synthetic_X[:, i], label='Synthetic', color='red', alpha=0.5)
        plt.title(f'Feature {i}' if feature_names is None else feature_names[i])
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_comparison.png'))
    plt.close()
    
    # Individual feature plots
    print("Generating individual feature plots...")
    for i in range(n_features):
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=real_X[:, i], label='Real', color='blue', alpha=0.5)
        sns.kdeplot(data=synthetic_X[:, i], label='Synthetic', color='red', alpha=0.5)
        plt.title(f'Feature {i}' if feature_names is None else feature_names[i])
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'feature_{i}_distribution.png'))
        plt.close()

def main():
    # Set paths
    model_path = "checkpoint/Adult/OI_11_00/final_model.weights.h5"
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Feature names for the Adult dataset
    feature_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week'
    ]
    
    # Generate synthetic data
    synthetic_X, synthetic_y, real_X, real_y = generate_synthetic_data(model_path)
    
    # Save synthetic data
    np.save(os.path.join(output_dir, "synthetic_features.npy"), synthetic_X)
    np.save(os.path.join(output_dir, "synthetic_labels.npy"), synthetic_y)
    print(f"Saved synthetic data to {output_dir}")
    
    # Evaluate the data
    evaluate_synthetic_data(real_X, real_y, synthetic_X, synthetic_y)
    
    # Create visualizations
    visualize_distributions(real_X, synthetic_X, output_dir, feature_names)
    print(f"Saved visualizations to {output_dir}")

if __name__ == "__main__":
    main()