#!/usr/bin/env python3
"""
Synthetic data generation using VAE for pose sequences.
This script trains VAE models on fall and normal sequences to generate synthetic data.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Linux
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from body import BODY_CONNECTIONS, BODY_PARTS_NAMES, draw_skeleton
from vae_model import VAE, vae_loss_function, generate_synthetic_data

# Set style for plots
sns.set_style('darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use default Linux font

class SyntheticDataGenerator:
    def __init__(self, data_path='output/data'):
        """Initialize the synthetic data generator."""
        self.data_path = data_path
        self.output_dir = 'output'
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.animations_dir = os.path.join(self.output_dir, 'animations')
        self.skeleton_dir = os.path.join(self.plots_dir, 'skeleton_samples')
        self.models_dir = 'models'
        self.augmentation_dir = os.path.join('data', 'augmentation')
        
        # Create output directories
        for directory in [self.plots_dir, self.animations_dir, self.skeleton_dir, 
                         self.models_dir, self.augmentation_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load keypoints sequences and labels."""
        keypoints_path = os.path.join(self.data_path, 'keypoints_sequences.npy')
        labels_path = os.path.join(self.data_path, 'labels.npy')
        
        if not os.path.exists(keypoints_path) or not os.path.exists(labels_path):
            # Try alternative paths
            keypoints_path = 'keypoints_sequences.npy'
            labels_path = 'labels.npy'
            
            if not os.path.exists(keypoints_path) or not os.path.exists(labels_path):
                raise FileNotFoundError(
                    "Keypoints data not found. Please run extract_data.py first to generate the data."
                )
        
        print(f"Loading data from: {keypoints_path}, {labels_path}")
        self.keypoints = np.load(keypoints_path)
        self.labels = np.load(labels_path)
        
        print(f"Data shape: {self.keypoints.shape}, Labels shape: {self.labels.shape}")
        
        # Separate fall and normal sequences
        self.fall_keypoints = self.keypoints[self.labels == 1]
        self.normal_keypoints = self.keypoints[self.labels == 0]
        
        print(f"Fall sequences: {self.fall_keypoints.shape}")
        print(f"Normal sequences: {self.normal_keypoints.shape}")
    
    def visualize_skeleton_static(self, keypoints_seq, frame_idx, title, save_name):
        """Visualize a single frame of skeleton keypoints."""
        plt.figure(figsize=(8, 8))
        
        keypoints_frame = keypoints_seq[frame_idx].reshape(-1, 2)
        
        # Draw skeleton
        ax = plt.gca()
        draw_skeleton(ax, keypoints_frame, title)
        
        plt.tight_layout()
        save_path = os.path.join(self.skeleton_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Skeleton visualization saved to: {save_path}")
    
    def visualize_sequence_frames(self, keypoints_seq, sequence_name, num_frames=5):
        """Visualize multiple frames from a sequence."""
        total_frames = keypoints_seq.shape[0]
        frame_indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
        
        fig, axes = plt.subplots(1, num_frames, figsize=(4*num_frames, 4))
        if num_frames == 1:
            axes = [axes]
        
        for i, frame_idx in enumerate(frame_indices):
            keypoints_frame = keypoints_seq[frame_idx].reshape(-1, 2)
            draw_skeleton(axes[i], keypoints_frame, f'Frame {frame_idx}')
        
        plt.suptitle(f'{sequence_name} - Multiple Frames', fontsize=16)
        plt.tight_layout()
        
        save_path = os.path.join(self.skeleton_dir, f'{sequence_name}_frames.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Multi-frame visualization saved to: {save_path}")
    
    def prepare_data_loaders(self, batch_size=16):
        """Prepare data loaders for training."""
        # Get dimensions
        self.num_sequences_fall, self.num_frames_fall, self.num_keypoints_fall = self.fall_keypoints.shape
        self.num_sequences_normal, self.num_frames_normal, self.num_keypoints_normal = self.normal_keypoints.shape
        
        # Flatten sequences
        fall_keypoints_flattened = self.fall_keypoints.reshape(
            self.num_sequences_fall, self.num_frames_fall * self.num_keypoints_fall
        )
        normal_keypoints_flattened = self.normal_keypoints.reshape(
            self.num_sequences_normal, self.num_frames_normal * self.num_keypoints_normal
        )
        
        print(f"Flattened shapes - Fall: {fall_keypoints_flattened.shape}, Normal: {normal_keypoints_flattened.shape}")
        
        # Convert to PyTorch tensors
        fall_tensor = torch.tensor(fall_keypoints_flattened, dtype=torch.float32)
        normal_tensor = torch.tensor(normal_keypoints_flattened, dtype=torch.float32)
        
        # Create datasets and loaders
        fall_dataset = TensorDataset(fall_tensor)
        normal_dataset = TensorDataset(normal_tensor)
        
        self.fall_loader = DataLoader(fall_dataset, batch_size=batch_size, shuffle=True)
        self.normal_loader = DataLoader(normal_dataset, batch_size=batch_size, shuffle=True)
        
        # Set input dimensions
        self.input_dim_fall = self.num_frames_fall * self.num_keypoints_fall
        self.input_dim_normal = self.num_frames_normal * self.num_keypoints_normal
    
    def train_vae_model(self, data_loader, input_dim, model_name, num_epochs=2000, 
                       hidden_dim=256, latent_dim=64, learning_rate=1e-3):
        """Train VAE model on given data."""
        print(f"\nTraining VAE model: {model_name}")
        print(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")
        
        # Initialize model
        model = VAE(input_dim, hidden_dim, latent_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        losses = []
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            num_batches = 0
            
            for batch in data_loader:
                data = batch[0].to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar = model(data)
                
                # Calculate loss
                loss = vae_loss_function(recon_batch, data, mu, logvar)
                
                # Backward pass
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
                num_batches += 1
            
            avg_loss = epoch_loss / len(data_loader.dataset)
            losses.append(avg_loss)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
        
        # Save model
        model_path = os.path.join(self.models_dir, f'{model_name}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")
        
        # Plot training loss
        self.plot_training_loss(losses, model_name)
        
        return model, losses
    
    def plot_training_loss(self, losses, model_name):
        """Plot training loss curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title(f'VAE Training Loss - {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        save_path = os.path.join(self.plots_dir, f'training_loss_{model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training loss plot saved to: {save_path}")
    
    def generate_and_visualize_synthetic_data(self, model, num_samples, latent_dim, 
                                            num_frames, num_keypoints, data_type):
        """Generate synthetic data and visualize samples."""
        print(f"\nGenerating {num_samples} synthetic {data_type} sequences...")
        
        # Generate synthetic data
        synthetic_data = generate_synthetic_data(model, num_samples, latent_dim, self.device)
        
        # Reshape to original format
        synthetic_sequences = synthetic_data.view(-1, num_frames, num_keypoints).cpu().numpy()
        
        print(f"Generated synthetic data shape: {synthetic_sequences.shape}")
        
        # Visualize some samples
        num_samples_to_visualize = min(3, num_samples)
        for i in range(num_samples_to_visualize):
            self.visualize_sequence_frames(
                synthetic_sequences[i], 
                f'synthetic_{data_type}_{i}', 
                num_frames=5
            )
        
        return synthetic_sequences
    
    def create_augmented_dataset(self, synthetic_fall_data, synthetic_normal_data):
        """Create augmented dataset combining real and synthetic data."""
        print("\nCreating augmented dataset...")
        
        # Combine all keypoints
        all_keypoints = np.concatenate([
            self.fall_keypoints, 
            self.normal_keypoints, 
            synthetic_fall_data, 
            synthetic_normal_data
        ])
        
        # Create labels
        all_labels = np.concatenate([
            np.ones(self.fall_keypoints.shape[0] + synthetic_fall_data.shape[0]),  # Falls = 1
            np.zeros(self.normal_keypoints.shape[0] + synthetic_normal_data.shape[0])  # Normal = 0
        ])
        
        print(f"Augmented dataset shape: {all_keypoints.shape}")
        print(f"Augmented labels shape: {all_labels.shape}")
        print(f"Total fall sequences: {np.sum(all_labels == 1)}")
        print(f"Total normal sequences: {np.sum(all_labels == 0)}")
        
        # Save augmented data
        keypoints_path = os.path.join(self.augmentation_dir, 'keypoints_sequences_vae.npy')
        labels_path = os.path.join(self.augmentation_dir, 'labels_vae.npy')
        
        np.save(keypoints_path, all_keypoints)
        np.save(labels_path, all_labels)
        
        print(f"Augmented keypoints saved to: {keypoints_path}")
        print(f"Augmented labels saved to: {labels_path}")
        
        return all_keypoints, all_labels
    
    def plot_data_distribution(self, original_labels, augmented_labels):
        """Plot data distribution comparison."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original data distribution
        original_counts = np.bincount(original_labels.astype(int))
        ax1.bar(['Normal', 'Fall'], original_counts, color=['blue', 'red'], alpha=0.7)
        ax1.set_title('Original Data Distribution')
        ax1.set_ylabel('Number of Sequences')
        
        # Augmented data distribution
        augmented_counts = np.bincount(augmented_labels.astype(int))
        ax2.bar(['Normal', 'Fall'], augmented_counts, color=['blue', 'red'], alpha=0.7)
        ax2.set_title('Augmented Data Distribution')
        ax2.set_ylabel('Number of Sequences')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, 'data_distribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Data distribution plot saved to: {save_path}")
    
    def run_pipeline(self, num_epochs=2000, num_synthetic_samples=200):
        """Run the complete synthetic data generation pipeline."""
        print("Starting Synthetic Data Generation Pipeline")
        print("=" * 60)
        
        # 1. Visualize some original samples
        print("\n1. Visualizing original data samples...")
        if len(self.fall_keypoints) > 0:
            self.visualize_sequence_frames(self.fall_keypoints[0], 'original_fall', num_frames=5)
        if len(self.normal_keypoints) > 0:
            self.visualize_sequence_frames(self.normal_keypoints[0], 'original_normal', num_frames=5)
        
        # 2. Prepare data loaders
        print("\n2. Preparing data loaders...")
        self.prepare_data_loaders(batch_size=16)
        
        # 3. Train VAE on fall sequences
        print("\n3. Training VAE on fall sequences...")
        fall_model, fall_losses = self.train_vae_model(
            self.fall_loader, 
            self.input_dim_fall, 
            'vae_fall_model',
            num_epochs=num_epochs
        )
        
        # 4. Generate synthetic fall data
        synthetic_fall_data = self.generate_and_visualize_synthetic_data(
            fall_model, 
            num_synthetic_samples, 
            64,  # latent_dim
            self.num_frames_fall, 
            self.num_keypoints_fall,
            'fall'
        )
        
        # 5. Train VAE on normal sequences
        print("\n5. Training VAE on normal sequences...")
        normal_model, normal_losses = self.train_vae_model(
            self.normal_loader, 
            self.input_dim_normal, 
            'vae_normal_model',
            num_epochs=num_epochs
        )
        
        # 6. Generate synthetic normal data
        synthetic_normal_data = self.generate_and_visualize_synthetic_data(
            normal_model, 
            num_synthetic_samples, 
            64,  # latent_dim
            self.num_frames_normal, 
            self.num_keypoints_normal,
            'normal'
        )
        
        # 7. Create augmented dataset
        all_keypoints, all_labels = self.create_augmented_dataset(
            synthetic_fall_data, 
            synthetic_normal_data
        )
        
        # 8. Plot data distribution comparison
        self.plot_data_distribution(self.labels, all_labels)
        
        print("\n" + "=" * 60)
        print("Synthetic data generation completed!")
        print(f"Check the following directories for results:")
        print(f"  - Plots: {self.plots_dir}")
        print(f"  - Models: {self.models_dir}")
        print(f"  - Augmented data: {self.augmentation_dir}")

def main():
    """Main function to run the synthetic data generation pipeline."""
    try:
        # Initialize generator
        generator = SyntheticDataGenerator()
        
        # Run the complete pipeline
        generator.run_pipeline(num_epochs=2000, num_synthetic_samples=200)
        
    except Exception as e:
        print(f"Error in synthetic data generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
