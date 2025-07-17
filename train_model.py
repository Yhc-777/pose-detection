import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.fall_detection_gru import FallDetectionGRU
from src.models.fall_detection_lstm import FallDetectionLSTM
from src.training.train_model import train_model
from src.training.utils import plot_training_history
from src.evaluation.evaluate import evaluate_model
from src.utils.video_detect_falls import video_detect_activities

# Set style for plots
sns.set_style('darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

class ActivityDetectionTrainer:
    def __init__(self, data_path='output/data', use_augmented=False):
        """Initialize the activity detection trainer."""
        self.data_path = data_path
        self.use_augmented = use_augmented
        self.output_dir = 'output'
        self.plots_dir = os.path.join(self.output_dir, 'plots', 'training')
        self.models_dir = 'models'
        
        # Create output directories
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load keypoints sequences and labels."""
        if self.use_augmented:
            keypoints_path = os.path.join('data', 'augmentation', 'keypoints_sequences_vae.npy')
            labels_path = os.path.join('data', 'augmentation', 'labels_vae.npy')
            print("Loading augmented data...")
        else:
            keypoints_path = os.path.join(self.data_path, 'keypoints_sequences.npy')
            labels_path = os.path.join(self.data_path, 'labels.npy')
            print("Loading original data...")
        
        if not os.path.exists(keypoints_path) or not os.path.exists(labels_path):

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

        print(f"Stand sequences: {np.sum(self.labels == 0)}")
        print(f"Walk sequences: {np.sum(self.labels == 1)}")
        print(f"Fall sequences: {np.sum(self.labels == 2)}")
    
    def prepare_data_splits(self, test_size=0.2, random_state=42):
        """Prepare train/validation/test splits."""
        print("Preparing data splits...")
        
        # First split: train + val vs test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.keypoints, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=0.25,  # 0.25 * 0.8 = 0.2 of total data
            random_state=random_state,
            stratify=y_train_val
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples") 
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Convert to PyTorch tensors - 修改标签类型
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.X_test = torch.tensor(X_test, dtype=torch.float32)
        self.y_train = torch.tensor(y_train, dtype=torch.long)  # 改为long类型用于多分类
        self.y_val = torch.tensor(y_val, dtype=torch.long)
        self.y_test = torch.tensor(y_test, dtype=torch.long)
    
    def create_data_loaders(self, batch_size=8):
        """Create PyTorch data loaders."""
        print(f"Creating data loaders with batch size: {batch_size}")
        
        # Create datasets
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        # Create loaders
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Validation batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
    
    def create_model(self, model_type='GRU', **model_params):
        """Create activity detection model."""
        input_size = self.keypoints.shape[-1]  # Number of keypoints features
        
        default_params = {
            'input_size': input_size,
            'hidden_size': 64,
            'num_layers': 2,
            'output_size': 3,  # 3分类：Stand, Walk, Fall
            'dropout_prob': 0.6
        }
        default_params.update(model_params)
        
        print(f"Creating {model_type} model with parameters: {default_params}")
        
        if model_type.upper() == 'GRU':
            model = FallDetectionGRU(**default_params)
        elif model_type.upper() == 'LSTM':
            model = FallDetectionLSTM(**default_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return model
    
    def train_model(self, model, **training_params):
        """Train the activity detection model."""
        default_params = {
            'num_epochs': 300,
            'learning_rate': 0.0005,
            'weight_decay': 1e-6,
            'patience': 15,
            'device': self.device
        }
        default_params.update(training_params)
        
        print(f"Training model with parameters: {default_params}")
        
        trained_model, history = train_model(
            model,
            self.train_loader,
            self.val_loader,
            **default_params
        )
        
        return trained_model, history
    
    def plot_training_history_custom(self, history, model_name='model'):
        """Plot and save training history."""
        print("Plotting training history...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(history['train_acc'], label='Train Accuracy', color='blue')
        axes[1].plot(history['val_acc'], label='Validation Accuracy', color='red')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, f'training_history_{model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training history plot saved to: {save_path}")
    
    def evaluate_model_custom(self, model, model_name='model'):
        """Evaluate model and create visualizations."""
        print(f"Evaluating {model_name} model...")
        
        # Get predictions
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                predictions = model(X_batch)
                predicted_classes = torch.argmax(predictions, dim=1)
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Classification report
        class_names = ['Stand', 'Walk', 'Fall']
        report = classification_report(all_labels, all_predictions, 
                                     target_names=class_names)
        print("\nClassification Report:")
        print(report)
        
        # Save classification report
        report_path = os.path.join(self.plots_dir, f'classification_report_{model_name}.txt')
        with open(report_path, 'w') as f:
            f.write(f"Model: {model_name}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
        print(f"Classification report saved to: {report_path}")
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, 
                   yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix - {model_name}')
        
        cm_path = os.path.join(self.plots_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to: {cm_path}")
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'accuracy': accuracy,
            'f1_score': f1
        }
    
    def save_model(self, model, model_name='gru_model'):
        """Save trained model."""
        # Save state dict
        state_dict_path = os.path.join(self.models_dir, f'{model_name}.pth')
        torch.save(model.state_dict(), state_dict_path)
        print(f"Model state dict saved to: {state_dict_path}")
        
        # Save full model
        full_model_path = os.path.join(self.models_dir, f'{model_name}_full.pth')
        torch.save(model, full_model_path)
        print(f"Full model saved to: {full_model_path}")
    
    def test_video_detection(self, model, video_path, model_type='GRU', yolo_model_path='models/yolo11x-pose.pt'):
        """Test activity detection on a video."""
        print(f"Testing activity detection on video: {video_path}")
        
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
        
        if not os.path.exists(yolo_model_path):
            print(f"YOLO model not found: {yolo_model_path}")
            return
        
        try:
            # Save model state dict temporarily for video processing
            temp_model_path = os.path.join(self.models_dir, 'temp_model.pth')
            torch.save(model.state_dict(), temp_model_path)
            
            video_detect_activities(  # 使用新函数
                video_path,
                yolo_model_path,
                temp_model_path,
                model_type=model_type,
                show_pose=True,
                record=True
            )
            
            # Clean up temporary file
            if os.path.exists(temp_model_path):
                os.remove(temp_model_path)
                
        except Exception as e:
            print(f"Error in video detection: {e}")
    
    def run_training_pipeline(self, model_type='GRU', use_video_test=False, 
                            test_video_path='data/videos/fall/banana-peel.mp4'):  # 修改默认路径
        """Run the complete training pipeline."""
        print("Starting Activity Detection Training Pipeline")  # 修改标题
        print("=" * 60)
        
        # 1. Prepare data
        print("\n1. Preparing data splits...")
        self.prepare_data_splits()
        
        # 2. Create data loaders
        print("\n2. Creating data loaders...")
        self.create_data_loaders(batch_size=8)
        
        # 3. Create model
        print(f"\n3. Creating {model_type} model...")
        model = self.create_model(model_type=model_type)
        print(f"Model architecture:\n{model}")
        
        # 4. Train model
        print(f"\n4. Training {model_type} model...")
        trained_model, history = self.train_model(model)
        
        # 5. Plot training history
        print("\n5. Plotting training history...")
        self.plot_training_history_custom(history, model_type.lower())
        
        # 6. Evaluate model
        print("\n6. Evaluating model...")
        evaluation_results = self.evaluate_model_custom(trained_model, model_type.lower())
        
        # 7. Save model
        print("\n7. Saving model...")
        self.save_model(trained_model, f'{model_type.lower()}_model')
        
        # 8. Test on video (optional)
        if use_video_test and os.path.exists(test_video_path):
            print(f"\n8. Testing on video: {test_video_path}")
            self.test_video_detection(trained_model, test_video_path, model_type)  # 传递model_type
        elif use_video_test:
            print(f"\n8. Video test skipped - file not found: {test_video_path}")
        
        print("\n" + "=" * 60)
        print("Training pipeline completed!")
        print(f"Best F1 Score: {evaluation_results['f1_score']:.4f}")
        print(f"Best Accuracy: {evaluation_results['accuracy']:.4f}")  # 移除threshold
        print(f"Check the following directories for results:")
        print(f"  - Training plots: {self.plots_dir}")
        print(f"  - Saved models: {self.models_dir}")

def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train activity detection model')  # 修改描述
    parser.add_argument('--model', type=str, default='GRU', choices=['GRU', 'LSTM'],
                       help='Model type to train (default: GRU)')
    parser.add_argument('--augmented', action='store_true',
                       help='Use augmented data generated by VAE')
    parser.add_argument('--test-video', type=str, default=None,
                       help='Path to test video for activity detection')  # 修改描述
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = ActivityDetectionTrainer(use_augmented=args.augmented)  # 使用新类名
        
        # Run training pipeline
        trainer.run_training_pipeline(
            model_type=args.model,
            use_video_test=args.test_video is not None,
            test_video_path=args.test_video or 'data/videos/fall/banana-peel.mp4'
        )
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
