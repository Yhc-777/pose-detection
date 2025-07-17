import torch
import torch.onnx
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.fall_detection_gru import FallDetectionGRU
from src.models.fall_detection_lstm import FallDetectionLSTM

def convert_model_to_onnx(model_path, model_type='GRU', output_path=None):
    """
    Convert PyTorch model to ONNX format
    """
    device = torch.device('cpu')  # 使用CPU进行转换
    
    # Create model
    if model_type.upper() == 'GRU':
        model = FallDetectionGRU(
            input_size=34, 
            hidden_size=64, 
            num_layers=2, 
            output_size=3,
            dropout_prob=0.6
        )
    elif model_type.upper() == 'LSTM':
        model = FallDetectionLSTM(
            input_size=34, 
            hidden_size=64, 
            num_layers=2, 
            output_size=3,
            dropout_prob=0.6
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Create dummy input (batch_size=1, sequence_length=20, input_size=34)
    dummy_input = torch.randn(1, 20, 34)
    
    # Set output path
    if output_path is None:
        base_name = os.path.splitext(os.path.basename(model_path))[0]
        output_path = f"{base_name}_{model_type.lower()}.onnx"
    
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
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model converted to ONNX: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model', required=True, help='Path to PyTorch model (.pth file)')
    parser.add_argument('--model-type', default='GRU', choices=['GRU', 'LSTM'], 
                       help='Model type (default: GRU)')
    parser.add_argument('--output', help='Output ONNX file path')
    
    args = parser.parse_args()
    
    convert_model_to_onnx(args.model, args.model_type, args.output)
