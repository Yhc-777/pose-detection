#!/usr/bin/env python3
import torch
from ultralytics import YOLO
import onnx
import onnxsim

def convert_to_compatible_onnx():
    """转换为OpenCV兼容的ONNX模型"""
    
    try:
        # 加载模型
        model = YOLO('models/yolo11n-pose.pt')
        
        # 导出为ONNX，使用OpenCV兼容的设置
        model.export(
            format='onnx',
            imgsz=640,
            opset=11,  # OpenCV 4.5.4 更好支持opset 11
            simplify=True,
            dynamic=False,
            half=False,  # 不使用半精度
        )
        
        # 进一步简化模型
        onnx_path = 'models/yolo11n-pose.onnx'
        simplified_path = 'models/yolo11n-pose_simplified.onnx'
        
        try:
            import onnxsim
            print("Simplifying ONNX model for better OpenCV compatibility...")
            
            model_onnx = onnx.load(onnx_path)
            model_simplified, check = onnxsim.simplify(model_onnx)
            
            if check:
                onnx.save(model_simplified, simplified_path)
                print(f"Simplified model saved to: {simplified_path}")
            else:
                print("Simplification failed, using original model")
                
        except ImportError:
            print("onnxsim not installed, skipping simplification")
            print("Install with: pip install onnxsim")
        
        print("Model conversion completed!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = convert_to_compatible_onnx()
    if not success:
        print("Conversion failed!")
        exit(1)
