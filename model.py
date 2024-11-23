"""
Paper: http://www.vldb.org/pvldb/vol11/p1071-park.pdf
Authors: Mahmoud Mohammadi, Noseong Park Adopted from https://github.com/carpedm20/DCGAN-tensorflow
Created : 07/20/2017
Modified: 10/15/2018
"""

from __future__ import division

import time
import tensorflow as tf
from ops import *
from utils import *
import os
import pickle
import numpy as np


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


class TableGan(tf.keras.Model):
    def __init__(self, input_width=7, input_height=7, output_width=7, output_height=7,
                 batch_size=100, sample_num=100, y_dim=2, dataset_name='Adult',
                 crop=False, checkpoint_dir=None, sample_dir=None, alpha=1.0, beta=1.0,
                 delta_mean=0.0, delta_var=0.0, label_col=13, attrib_num=13,
                 is_shadow_gan=False, test_id='test'):
        super(TableGan, self).__init__()
        
        self.input_width = input_width
        self.input_height = input_height
        self.output_width = output_width
        self.output_height = output_height
        self.batch_size = batch_size
        self.sample_num = sample_num
        self.y_dim = y_dim
        self.dataset_name = dataset_name
        self.crop = crop
        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir
        self.alpha = alpha
        self.beta = beta
        self.delta_mean = delta_mean
        self.delta_var = delta_var
        self.label_col = label_col
        self.attrib_num = attrib_num
        self.is_shadow_gan = is_shadow_gan
        self.test_id = test_id

        # Load dataset and convert to float32
        self.data_X, self.data_y, self.data_y_normal = self.load_dataset(is_shadow_gan)
        self.data_X = tf.cast(self.data_X, tf.float32)
        self.data_y = tf.cast(self.data_y, tf.float32)
        
        print(f"Feature dimension: {self.data_X.shape[1]}")
        print(f"Label dimension: {self.data_y.shape[1]}")
        
        # Build the model
        self.build_model()

    def build_model(self):
        # Generator
        self.generator = self.build_generator()
        
        # Discriminator
        self.discriminator = self.build_discriminator()
        
        # Optimizers
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

    def build_generator(self):
        input_dim = 100 + self.y_dim  # noise_dim + label_dim
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),  # Explicit input layer
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.BatchNormalization(),
            
            tf.keras.layers.Dense(self.attrib_num, activation='tanh')  # Output size matches feature dimension
        ])
        return model

    def build_discriminator(self):
        input_dim = self.attrib_num + self.y_dim  # features + labels
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),  # Explicit input layer
            tf.keras.layers.Dense(1024),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(512),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        return model

    @tf.function
    def train_step(self, real_data, labels):
        # Ensure input types are float32
        real_data = tf.cast(real_data, tf.float32)
        labels = tf.cast(labels, tf.float32)
        batch_size = tf.shape(real_data)[0]
        
        # Generate random noise
        noise = tf.random.normal([batch_size, 100], dtype=tf.float32)
        
        # Concatenate noise and labels for generator input
        generator_input = tf.concat([noise, labels], axis=1)
        
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake data
            generated_data = self.generator(generator_input, training=True)
            
            # Prepare discriminator inputs
            real_input = tf.concat([real_data, labels], axis=1)
            fake_input = tf.concat([generated_data, labels], axis=1)
            
            # Get discriminator outputs
            real_output = self.discriminator(real_input, training=True)
            fake_output = self.discriminator(fake_input, training=True)
            
            # Calculate losses
            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)
        
        # Calculate gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        
        # Apply gradients
        self.g_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    def train(self, config):
        print("Starting training...")
        
        # Create checkpoint directory
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Training loop
        for epoch in range(config.epoch):
            # Shuffle the data
            indices = np.random.permutation(len(self.data_X))
            
            total_gen_loss = 0
            total_disc_loss = 0
            num_batches = 0
            
            # Train on batches
            for i in range(0, len(self.data_X), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_data = tf.gather(self.data_X, batch_indices)
                batch_labels = tf.gather(self.data_y, batch_indices)
                
                # Train step
                gen_loss, disc_loss = self.train_step(batch_data, batch_labels)
                
                total_gen_loss += gen_loss
                total_disc_loss += disc_loss
                num_batches += 1
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{config.epoch}")
                print(f"Generator Loss: {total_gen_loss/num_batches}")
                print(f"Discriminator Loss: {total_disc_loss/num_batches}")
                
                # Save checkpoint with proper extension
                checkpoint_path = os.path.join(self.checkpoint_dir, f'ckpt_{epoch}.weights.h5')
                self.save_weights(checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")
        
        # Save final model
        final_checkpoint = os.path.join(self.checkpoint_dir, 'final_model.weights.h5')
        self.save_weights(final_checkpoint)
        print(f"Training completed! Final model saved to {final_checkpoint}")

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(tf.math.log(fake_output + 1e-8))

    def discriminator_loss(self, real_output, fake_output):
        real_loss = -tf.reduce_mean(tf.math.log(real_output + 1e-8))
        fake_loss = -tf.reduce_mean(tf.math.log(1 - fake_output + 1e-8))
        return real_loss + fake_loss

    def load_dataset(self, is_shadow_gan=False):
        """Load the preprocessed Adult dataset"""
        data_path = os.path.join('data', self.dataset_name)
        
        try:
            # Load features
            with open(os.path.join(data_path, 'train_Adult_cleaned.pickle'), 'rb') as f:
                data_X = pickle.load(f)
            
            # Load labels
            with open(os.path.join(data_path, 'train_Adult_labels.pickle'), 'rb') as f:
                data_y = pickle.load(f)
            
            # Convert labels to one-hot encoding
            data_y_normal = data_y.copy()
            data_y = np.eye(self.y_dim)[data_y]
            
            # Convert to float32
            data_X = data_X.astype(np.float32)
            data_y = data_y.astype(np.float32)
            
            print(f"Loaded data shapes - X: {data_X.shape}, y: {data_y.shape}")
            return data_X, data_y, data_y_normal
            
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            print(f"Looking for files in: {data_path}")
            print(f"Current working directory: {os.getcwd()}")
            raise

    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        try:
            self.load_weights(checkpoint_path)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
            return True
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            return False

    def save_model(self):
        """Save both generator and discriminator"""
        gen_path = os.path.join(self.checkpoint_dir, 'generator.weights.h5')
        disc_path = os.path.join(self.checkpoint_dir, 'discriminator.weights.h5')
        
        self.generator.save_weights(gen_path)
        self.discriminator.save_weights(disc_path)
        print(f"Saved models to {self.checkpoint_dir}")

    def load_model(self):
        """Load both generator and discriminator"""
        gen_path = os.path.join(self.checkpoint_dir, 'generator.weights.h5')
        disc_path = os.path.join(self.checkpoint_dir, 'discriminator.weights.h5')
        
        try:
            self.generator.load_weights(gen_path)
            self.discriminator.load_weights(disc_path)
            print("Successfully loaded models")
            return True
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False
