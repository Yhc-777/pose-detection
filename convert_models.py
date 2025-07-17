#!/usr/bin/env python3
"""
Convert PyTorch models to ONNX format for C++ inference
"""

import torch
import sys
import os
import warnings

# 抑制ONNX导出警告
warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append('src')

from src.models.fall_detection_gru import FallDetectionGRU
from ultralytics import YOLO

def convert_gru_model():
    """Convert GRU model to ONNX"""
    print("Converting GRU model to ONNX...")
    
    try:
        # Load trained model
        model = FallDetectionGRU(
            input_size=34,
            hidden_size=64,
            num_layers=2,
            output_size=3,
            dropout_prob=0.6
        )
        
        # 检查模型文件是否存在
        model_path = 'models/gru_model.pth'
        if not os.path.exists(model_path):
            print(f"Error: Model file not found: {model_path}")
            return False
        
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        # 创建dummy input - 批次大小设为1以避免警告
        dummy_input = torch.randn(1, 20, 34)  # (batch_size=1, sequence_length, input_size)
        
        # 确保输出目录存在
        os.makedirs('models', exist_ok=True)
        
        # Export to ONNX with more specific settings
        torch.onnx.export(
            model,
            dummy_input,
            'models/gru_model.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},  # 允许动态批次大小
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print("✓ GRU model converted to: models/gru_model.onnx")
        
        # 验证ONNX模型
        try:
            import onnx
            onnx_model = onnx.load('models/gru_model.onnx')
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model validation passed")
        except Exception as e:
            print(f"Warning: ONNX model validation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"Error converting GRU model: {e}")
        return False

def convert_yolo_model():
    """Convert YOLO model to ONNX"""
    print("Converting YOLO model to ONNX...")
    
    try:
        # 检查YOLO模型文件是否存在
        yolo_model_path = 'models/yolo11n-pose.pt'
        if not os.path.exists(yolo_model_path):
            print(f"Error: YOLO model file not found: {yolo_model_path}")
            print("Please download the model first:")
            print("wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n-pose.pt -O models/yolo11n-pose.pt")
            return False
        
        # Load YOLO model
        model = YOLO(yolo_model_path)
        
        # Export to ONNX
        success = model.export(
            format='onnx',
            imgsz=640,
            optimize=True,
            half=False,
            int8=False,
            dynamic=False,
            simplify=True,
            verbose=False
        )
        
        if success:
            # 检查导出的文件位置
            exported_file = yolo_model_path.replace('.pt', '.onnx')
            target_file = 'models/yolo11n-pose.onnx'
            
            if os.path.exists(exported_file) and exported_file != target_file:
                os.rename(exported_file, target_file)
            
            print("✓ YOLO model converted to: models/yolo11n-pose.onnx")
            return True
        else:
            print("✗ YOLO model conversion failed")
            return False
            
    except Exception as e:
        print(f"Error converting YOLO model: {e}")
        return False

def verify_models():
    """Verify that the converted models exist and are valid"""
    print("\nVerifying converted models...")
    
    models_to_check = [
        'models/gru_model.onnx',
        'models/yolo11n-pose.onnx'
    ]
    
    all_good = True
    for model_path in models_to_check:
        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            print(f"✓ {model_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {model_path} - Not found")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    print("PyTorch to ONNX Model Converter")
    print("=" * 40)
    
    # 检查必要的包
    try:
        import onnx
        import onnxruntime
        print("✓ ONNX packages available")
    except ImportError as e:
        print(f"✗ Missing required packages: {e}")
        print("Please install: pip install onnx onnxruntime")
        sys.exit(1)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Convert models
    gru_success = convert_gru_model()
    yolo_success = convert_yolo_model()
    
    # Verify results
    if gru_success and yolo_success:
        if verify_models():
            print("\n🎉 All models converted successfully!")
            print("\nNext steps:")
            print("1. Install C++ dependencies: ./install_dependencies.sh")
            print("2. Build the project: ./build.sh")
            print("3. Run inference: ./build/activity_detection input.mp4 --output result.mp4")
        else:
            print("\n⚠️  Some models may not have been converted properly")
    else:
        print("\n❌ Model conversion failed")
        if not gru_success:
            print("- GRU model conversion failed")
        if not yolo_success:
            print("- YOLO model conversion failed")
