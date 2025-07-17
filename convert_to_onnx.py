#!/usr/bin/env python3
"""
Convert trained PyTorch models to ONNX format for C++ inference.
"""

import torch
import torch.onnx
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.fall_detection_gru import FallDetectionGRU
from src.models.fall_detection_lstm import FallDetectionLSTM

def convert_gru_to_onnx(model_path, output_path):
    """Convert GRU model to ONNX format."""
    print(f"Converting GRU model: {model_path}")
    
    # Create model
    model = FallDetectionGRU(
        input_size=34,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout_prob=0.6
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input (batch_size=1, sequence_length=20, input_size=34)
    dummy_input = torch.randn(1, 20, 34)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"GRU model exported to: {output_path}")

def convert_lstm_to_onnx(model_path, output_path):
    """Convert LSTM model to ONNX format."""
    print(f"Converting LSTM model: {model_path}")
    
    # Create model
    model = FallDetectionLSTM(
        input_size=34,
        hidden_size=64,
        num_layers=2,
        output_size=3,
        dropout_prob=0.6
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 20, 34)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"LSTM model exported to: {output_path}")

def convert_yolo_to_onnx():
    """Convert YOLO model to ONNX format."""
    from ultralytics import YOLO
    
    print("Converting YOLO model to ONNX...")
    
    # Load YOLO model
    model = YOLO('models/yolo11n-pose.pt')
    
    # Export to ONNX
    model.export(format='onnx', opset=11, simplify=True)
    
    print("YOLO model exported to: models/yolo11n-pose.onnx")

def main():
    """Main conversion function."""
    print("=== Model Conversion to ONNX ===")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Convert GRU model
    if os.path.exists('models/gru_model.pth'):
        convert_gru_to_onnx('models/gru_model.pth', 'models/gru_model.onnx')
    else:
        print("Warning: GRU model not found, skipping conversion")
    
    # Convert LSTM model
    if os.path.exists('models/lstm_model.pth'):
        convert_lstm_to_onnx('models/lstm_model.pth', 'models/lstm_model.onnx')
    else:
        print("Warning: LSTM model not found, skipping conversion")
    
    # Convert YOLO model
    if os.path.exists('models/yolo11n-pose.pt'):
        convert_yolo_to_onnx()
    else:
        print("Warning: YOLO model not found, skipping conversion")
    
    print("\n=== Conversion Summary ===")
    print("Available ONNX models:")
    for model_file in ['gru_model.onnx', 'lstm_model.onnx', 'yolo11n-pose.onnx']:
        model_path = os.path.join('models', model_file)
        if os.path.exists(model_path):
            file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            print(f"  ✓ {model_file} ({file_size:.1f} MB)")
        else:
            print(f"  ✗ {model_file} (not found)")

if __name__ == "__main__":
    main()
