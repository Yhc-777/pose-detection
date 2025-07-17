# #!/usr/bin/env python3
# """
# Inference script for fall detection using trained GRU/LSTM model.
# Supports both video and image processing with pose keypoints visualization.
# Default headless mode for video processing.
# """

# import os
# import sys
# import argparse
# import cv2
# import time
# import numpy as np
# import torch
# from ultralytics import YOLO
# import supervision as sv

# # Add src to path
# sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# from src.models.fall_detection_gru import FallDetectionGRU
# from src.models.fall_detection_lstm import FallDetectionLSTM

# class FallDetectionInference:
#     def __init__(self, model_path, model_type='GRU', yolo_model_path='models/yolo11n-pose.pt'):
#         """
#         Initialize fall detection inference.
        
#         Args:
#             model_path (str): Path to trained model (.pth file)
#             model_type (str): Model type ('GRU' or 'LSTM')
#             yolo_model_path (str): Path to YOLO pose model
#         """
#         self.model_path = model_path
#         self.model_type = model_type.upper()
#         self.yolo_model_path = yolo_model_path
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#         self.class_names = ['Stand', 'Walk', 'Fall']
#         self.class_colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255)]  # 绿色、黄色、红色
        
#         # Body parts and connections for pose visualization
#         self.BODY_PARTS_NAMES = [
#             "nose", "left_eye", "right_eye", "left_ear", "right_ear",
#             "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
#             "left_wrist", "right_wrist", "left_hip", "right_hip",
#             "left_knee", "right_knee", "left_ankle", "right_ankle"
#         ]

#         self.BODY_CONNECTIONS_DRAW = {
#             "legs": ([("left_hip", "left_knee"), ("left_knee", "left_ankle"), 
#                      ("right_hip", "right_knee"), ("right_knee", "right_ankle")], (255, 0, 0)),
#             "arms": ([("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), 
#                      ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")], (0, 255, 0)),
#             "head": ([("nose", "left_eye"), ("nose", "right_eye"), 
#                      ("left_eye", "left_ear"), ("right_eye", "right_ear")], (0, 0, 255)),
#             "torso": ([("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), 
#                       ("right_shoulder", "right_hip"), ("left_hip", "right_hip")], (255, 255, 0))
#         }
        
#         # Load models
#         self.load_models()
        
#         # Initialize trackers and annotators
#         self.byte_tracker = sv.ByteTrack()
        
#     def load_models(self):
#         """Load YOLO and fall detection models."""
#         print(f"Loading YOLO model from: {self.yolo_model_path}")
#         if not os.path.exists(self.yolo_model_path):
#             raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")
        
#         self.yolo_model = YOLO(self.yolo_model_path, verbose=False)
        
#         print(f"Loading {self.model_type} model from: {self.model_path}")
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(f"Fall detection model not found: {self.model_path}")
        
#         # Create model based on type
#         if self.model_type == 'GRU':
#             self.fall_model = FallDetectionGRU(
#                 input_size=34, 
#                 hidden_size=64, 
#                 num_layers=2, 
#                 output_size=3,  # 修改为3分类
#                 dropout_prob=0.6
#             )
#         elif self.model_type == 'LSTM':
#             self.fall_model = FallDetectionLSTM(
#                 input_size=34, 
#                 hidden_size=64, 
#                 num_layers=2, 
#                 output_size=3,  # 修改为3分类
#                 dropout_prob=0.6
#             )
#         else:
#             raise ValueError(f"Unsupported model type: {self.model_type}. Use 'GRU' or 'LSTM'")
        
#         # Load model weights
#         self.fall_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
#         self.fall_model.to(self.device)
#         self.fall_model.eval()
        
#         print(f"{self.model_type} model loaded successfully on device: {self.device}")
    
#     def draw_pose_keypoints(self, frame, keypoints_2d):
#         """Draw pose keypoints and connections on frame."""
#         if keypoints_2d.size(0) == 0:
#             return frame
        
#         # Create body parts dictionary
#         body = {part: keypoints_2d[i] for i, part in enumerate(self.BODY_PARTS_NAMES)}
        
#         # Draw connections
#         for group, (connections, color) in self.BODY_CONNECTIONS_DRAW.items():
#             for part_a, part_b in connections:
#                 if part_a in body and part_b in body:
#                     x1, y1 = map(int, body[part_a])
#                     x2, y2 = map(int, body[part_b])
                    
#                     # Skip if coordinates are invalid
#                     if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
#                         continue
                    
#                     cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
#         # Draw keypoints
#         for group, (connections, color) in self.BODY_CONNECTIONS_DRAW.items():
#             for part_a, _ in connections:
#                 if part_a in body:
#                     x, y = map(int, body[part_a])
#                     if x != 0 and y != 0:
#                         cv2.circle(frame, (x, y), 4, color, -1)
        
#         return frame
    
#     def predict_activity(self, keypoints_sequence):
#         """Predict activity from keypoints sequence."""
#         if len(keypoints_sequence) == 0:
#             return 0, np.array([0.33, 0.33, 0.34])  # 返回numpy数组
        
#         # Convert to tensor
#         keypoints_tensor = torch.tensor(keypoints_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
#         with torch.no_grad():
#             prediction = self.fall_model(keypoints_tensor)
#             # 使用softmax获取概率分布
#             probabilities = torch.softmax(prediction, dim=1)
#             probabilities_np = probabilities[0].cpu().numpy()  # 转换为numpy数组
#             predicted_class = np.argmax(probabilities_np)
        
#         return predicted_class, probabilities_np


#     def process_image(self, image_path, output_path=None, show_pose=True):
#         """Process single image for activity detection."""
#         print(f"Processing image: {image_path}")
        
#         if not os.path.exists(image_path):
#             raise FileNotFoundError(f"Image not found: {image_path}")
        
#         # Read image
#         frame = cv2.imread(image_path)
#         if frame is None:
#             raise ValueError(f"Could not read image: {image_path}")
        
#         height, width = frame.shape[:2]
#         annotated_frame = frame.copy()
        
#         # Run YOLO detection
#         results = self.yolo_model(frame)[0]
        
#         # Process detections
#         if len(results.keypoints.xy) > 0:
#             detections = sv.Detections.from_ultralytics(results)
            
#             # Annotate with bounding boxes (smaller text)
#             bounding_box_annotator = sv.BoxAnnotator(thickness=2)
#             label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
            
#             labels = [f'{results.names[class_id]} {confidence:.2f}' 
#                     for class_id, confidence in zip(detections.class_id, detections.confidence)]
            
#             annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
#             annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
#             # Process each person
#             for person_idx in range(len(results.keypoints.xy)):
#                 keypoints_2d = results.keypoints.xy[person_idx].cpu()
                
#                 if keypoints_2d.size(0) == 0:
#                     continue
                
#                 # Extract keypoints for activity detection
#                 keypoints_flat = keypoints_2d.numpy().flatten()
                
#                 # For single image, we can't use sequence, so we repeat the keypoints
#                 # This is not ideal but works for demonstration
#                 sequence_length = 20
#                 keypoints_sequence = np.tile(keypoints_flat, (sequence_length, 1))
                
#                 # Predict activity
#                 predicted_class, probabilities = self.predict_activity(keypoints_sequence)
                
#                 # Draw pose if enabled
#                 if show_pose:
#                     annotated_frame = self.draw_pose_keypoints(annotated_frame, keypoints_2d)
                
#                 # Add activity detection result
#                 class_name = self.class_names[predicted_class]
#                 class_color = self.class_colors[predicted_class]
#                 confidence = probabilities[predicted_class]
                
#                 status_text = f'{self.model_type} - {class_name}: {confidence:.3f}'
                
#                 cv2.putText(annotated_frame, status_text, (10, 25 + person_idx * 20), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 1, cv2.LINE_AA)
                
#                 # Show all probabilities
#                 prob_text = f'Stand:{probabilities[0]:.2f} Walk:{probabilities[1]:.2f} Fall:{probabilities[2]:.2f}'
#                 cv2.putText(annotated_frame, prob_text, (10, 45 + person_idx * 20), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
#                 # Highlight person based on activity
#                 if hasattr(results, 'boxes') and results.boxes is not None:
#                     boxes = results.boxes.xyxy.cpu().numpy()
#                     if person_idx < len(boxes):
#                         x1, y1, x2, y2 = map(int, boxes[person_idx])
#                         cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_color, 3)
#                         # 在边框旁边显示活动类别
#                         cv2.putText(annotated_frame, f'{class_name.upper()}!', (x2 + 10, y1 + 20), 
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2, cv2.LINE_AA)
        
#         # Add image info
#         cv2.putText(annotated_frame, f'Model: {self.model_type}', (10, height - 45), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#         cv2.putText(annotated_frame, f'Size: {width}x{height}', (10, height - 25), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
#         cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', (10, height - 5), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
#         # Save or display result
#         if output_path:
#             cv2.imwrite(output_path, annotated_frame)
#             print(f"Processed image saved to: {output_path}")
#         else:
#             # Display image
#             cv2.imshow(f'Activity Detection - {self.model_type} - Image', annotated_frame)
#             print("Press any key to close the image window...")
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
        
#         return annotated_frame

#     def process_video(self, video_path, output_path=None, sequence_length=20, 
#                     show_pose=True, scale_percent=100, headless=True, show_display=False):
#         """
#         Process video for activity detection.
#         """
#         print(f"Processing video: {video_path}")
#         print(f"Using {self.model_type} model for activity detection")
        
#         if not os.path.exists(video_path):
#             raise FileNotFoundError(f"Video not found: {video_path}")
        
#         # Force headless for video unless explicitly requested to show display
#         if not show_display:
#             headless = True
        
#         # Check for headless mode
#         if headless or not os.environ.get('DISPLAY'):
#             headless = True
#             print("Running in headless mode (no display window)")
#         else:
#             print("Display mode enabled - press 'q' to quit")
        
#         # Open video
#         cap = cv2.VideoCapture(video_path)
#         if not cap.isOpened():
#             raise ValueError(f"Cannot open video: {video_path}")
        
#         # Get video properties
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = int(cap.get(cv2.CAP_PROP_FPS))
#         total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
#         new_width = int(width * scale_percent / 100)
#         new_height = int(height * scale_percent / 100)
        
#         text_scale = 0.5 if scale_percent >= 80 else 0.4
#         text_thickness = 1
        
#         print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
#         # Setup video writer - 修复这里
#         out = None
#         if output_path:
#             # 确保输出路径有正确的扩展名
#             if not output_path.endswith('.mp4'):
#                 output_path = output_path + '.mp4'
            
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
#             # 检查VideoWriter是否成功初始化
#             if not out.isOpened():
#                 print(f"Warning: Could not initialize video writer for {output_path}")
#                 print("Video will be processed but not saved")
#                 out = None
#             else:
#                 print(f"Recording output video to: {output_path}")
        
#         # Initialize annotators
#         bounding_box_annotator = sv.BoxAnnotator(thickness=2)
#         label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)
#         trace_annotator = sv.TraceAnnotator(thickness=2)
        
#         # Keypoints buffer for sequence
#         keypoints_buffer = []
#         frame_count = 0
#         activity_detections = {'stand': [], 'walk': [], 'fall': []}
        
#         # 当前活动状态（用于稳定显示）
#         current_activity = 0
#         current_probabilities = np.array([0.33, 0.33, 0.34])
        
#         print("Starting video processing...")
#         if not headless:
#             print("Press 'q' to quit")
#         else:
#             print("Press Ctrl+C to stop processing")
        
#         try:
#             while True:
#                 start_time = time.time()
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("Video processing completed")
#                     break
                
#                 frame_count += 1
#                 annotated_frame = frame.copy()
                
#                 # Run YOLO detection
#                 results = self.yolo_model(frame)[0]
                
#                 if len(results.keypoints.xy) > 0:
#                     detections = sv.Detections.from_ultralytics(results)
#                     detections = self.byte_tracker.update_with_detections(detections)
                    
#                     # Create labels
#                     labels = [f'#{tracker_id} {results.names[class_id]} {confidence:.2f}' 
#                             for class_id, confidence, tracker_id in 
#                             zip(detections.class_id, detections.confidence, detections.tracker_id)]
                    
#                     # Annotate frame
#                     annotated_frame = trace_annotator.annotate(annotated_frame, detections)
#                     annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
#                     annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
                    
#                     # Process each person
#                     for person_idx in range(len(results.keypoints.xy)):
#                         keypoints_2d = results.keypoints.xy[person_idx].cpu()
                        
#                         if keypoints_2d.size(0) == 0:
#                             continue
                        
#                         # Extract and buffer keypoints
#                         keypoints_flat = keypoints_2d.numpy().flatten()
#                         keypoints_buffer.append(keypoints_flat)
                        
#                         # Maintain buffer size
#                         if len(keypoints_buffer) > sequence_length:
#                             keypoints_buffer.pop(0)
                        
#                         # Predict activity if we have enough frames
#                         if len(keypoints_buffer) == sequence_length:
#                             keypoints_sequence = np.array(keypoints_buffer, dtype=np.float32)
#                             predicted_class, probabilities = self.predict_activity(keypoints_sequence)
                            
#                             # 更新当前状态
#                             current_activity = predicted_class
#                             current_probabilities = probabilities
                            
#                             class_name = self.class_names[predicted_class]
#                             class_color = self.class_colors[predicted_class]
                            
#                             # Record detections
#                             activity_detections[class_name.lower()].append(frame_count)
                            
#                             # Print detection
#                             if predicted_class != 0:  # 如果不是站立，则打印
#                                 print(f"Frame {frame_count}: {self.model_type} detected {class_name}! Probability: {probabilities[predicted_class]:.3f}")
                            
#                             # Highlight if specific activity detected
#                             if hasattr(results, 'boxes') and results.boxes is not None:
#                                 boxes = results.boxes.xyxy.cpu().numpy()
#                                 if person_idx < len(boxes):
#                                     x1, y1, x2, y2 = map(int, boxes[person_idx])
#                                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_color, 3)
#                                     cv2.putText(annotated_frame, f'{class_name.upper()}!', (x2 + 10, y1 + 20), 
#                                             cv2.FONT_HERSHEY_SIMPLEX, text_scale, class_color, text_thickness + 1, cv2.LINE_AA)
                        
#                         # Draw pose if enabled
#                         if show_pose:
#                             annotated_frame = self.draw_pose_keypoints(annotated_frame, keypoints_2d)
                
#                 # 始终显示当前活动状态（即使没有新的预测）
#                 class_name = self.class_names[current_activity]
#                 class_color = self.class_colors[current_activity]
                
#                 # Display prediction with probabilities
#                 status_text = f'{self.model_type} - {class_name}: {current_probabilities[current_activity]:.3f}'
#                 cv2.putText(annotated_frame, status_text, (10, 100), 
#                         cv2.FONT_HERSHEY_SIMPLEX, text_scale, class_color, text_thickness, cv2.LINE_AA)
                
#                 # Show all probabilities
#                 prob_text = f'Stand:{current_probabilities[0]:.2f} Walk:{current_probabilities[1]:.2f} Fall:{current_probabilities[2]:.2f}'
#                 cv2.putText(annotated_frame, prob_text, (10, 120), 
#                         cv2.FONT_HERSHEY_SIMPLEX, text_scale * 0.8, (255, 255, 255), text_thickness, cv2.LINE_AA)
                
#                 # Calculate and display FPS
#                 processing_time = time.time() - start_time
#                 fps_real = 1 / processing_time if processing_time > 0 else 0
                
#                 # Add frame information
#                 cv2.putText(annotated_frame, f'Model: {self.model_type}', (10, 20), 
#                         cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 0), text_thickness, cv2.LINE_AA)
#                 cv2.putText(annotated_frame, f'Size: {width}x{height}', (10, 40), 
#                         cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
#                 cv2.putText(annotated_frame, f'FPS: {fps_real:.1f}', (10, 60), 
#                         cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
#                 cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}', (10, 80), 
#                         cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), text_thickness, cv2.LINE_AA)
#                 cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', (10, height - 20), 
#                         cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 255, 255), text_thickness, cv2.LINE_AA)
                
#                 # Save frame if recording - 确保这部分正常工作
#                 if out is not None and out.isOpened():
#                     out.write(annotated_frame)
                    
#                     # 每100帧打印一次保存状态
#                     if frame_count % 100 == 0:
#                         print(f"Saving frame {frame_count} to output video...")
                
#                 # Display frame if not headless
#                 if not headless:
#                     try:
#                         display_frame = cv2.resize(annotated_frame, (new_width, new_height))
#                         cv2.imshow(f'Activity Detection - {self.model_type}', display_frame)
                        
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             print("User pressed 'q', stopping processing...")
#                             break
#                     except cv2.error as e:
#                         print(f"Display error: {e}")
#                         print("Switching to headless mode")
#                         headless = True
                
#                 # Progress update in headless mode
#                 if headless and frame_count % 100 == 0:
#                     progress = (frame_count / total_frames) * 100
#                     print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames), FPS: {fps_real:.1f}")
        
#         except KeyboardInterrupt:
#             print("\nUser interrupted, stopping processing...")
        
#         finally:
#             # Cleanup
#             cap.release()
#             if out is not None:
#                 out.release()
#                 if output_path and os.path.exists(output_path):
#                     file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
#                     print(f"Output video saved to: {output_path} (Size: {file_size:.1f} MB)")
#                 else:
#                     print("Warning: Output video may not have been saved properly")
            
#             if not headless:
#                 try:
#                     cv2.destroyAllWindows()
#                 except:
#                     pass
            
#             # Summary
#             print(f"\nProcessing Summary:")
#             print(f"Model used: {self.model_type}")
#             print(f"Total frames processed: {frame_count}")
#             for activity, frames in activity_detections.items():
#                 if frames:
#                     print(f"{activity.capitalize()} detections: {len(frames)} frames")
#                     if len(frames) <= 10:
#                         print(f"  Detected at frames: {frames}")
#                     else:
#                         print(f"  First 10 frames: {frames[:10]}...")


# def main():
#     """Main function for command line interface."""
#     parser = argparse.ArgumentParser(description='Activity Detection Inference')
#     parser.add_argument('input', help='Input video or image path')
#     parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
#     parser.add_argument('--model-type', default='GRU', choices=['GRU', 'LSTM'], 
#                        help='Model type (default: GRU)')
#     parser.add_argument('--yolo-model', default='models/yolo11n-pose.pt', 
#                        help='Path to YOLO pose model')
#     parser.add_argument('--output', help='Output path for processed video/image')
#     parser.add_argument('--sequence-length', type=int, default=20, 
#                        help='Sequence length for video processing (default: 20)')
#     parser.add_argument('--no-pose', action='store_true', 
#                        help='Disable pose keypoints visualization')
#     parser.add_argument('--scale', type=int, default=100, 
#                        help='Display scale percentage (default: 100)')
#     parser.add_argument('--show-display', action='store_true', 
#                        help='Force show display window for video (default: headless for video)')
    
#     args = parser.parse_args()
    
#     try:
#         # Initialize inference - 修改这里，移除fall_threshold参数
#         inference = FallDetectionInference(
#             model_path=args.model,
#             model_type=args.model_type,
#             yolo_model_path=args.yolo_model
#         )
        
#         # Check if input is image or video
#         input_ext = os.path.splitext(args.input)[1].lower()
        
#         if input_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
#             # Process image
#             print(f"Processing image with {args.model_type} model...")
#             inference.process_image(
#                 image_path=args.input,
#                 output_path=args.output,
#                 show_pose=not args.no_pose
#             )
#         elif input_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
#             # Process video
#             print(f"Processing video with {args.model_type} model...")
#             inference.process_video(
#                 video_path=args.input,
#                 output_path=args.output,
#                 sequence_length=args.sequence_length,
#                 show_pose=not args.no_pose,
#                 scale_percent=args.scale,
#                 headless=True,
#                 show_display=args.show_display
#             )
#         else:
#             print(f"Unsupported file format: {input_ext}")
#             print("Supported formats: Images (.jpg, .png, etc.), Videos (.mp4, .avi, etc.)")
    
#     except Exception as e:
#         print(f"Error during inference: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()


































#!/usr/bin/env python3
"""
Inference script for fall detection using trained GRU/LSTM model.
Supports both video and image processing with pose keypoints visualization.
Headless mode only - no GUI display.
"""

import os
import sys
import argparse
import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.fall_detection_gru import FallDetectionGRU
from src.models.fall_detection_lstm import FallDetectionLSTM

class FallDetectionInference:
    def __init__(self, model_path, model_type='GRU', yolo_model_path='models/yolo11n-pose.pt'):
        """
        Initialize fall detection inference.
        
        Args:
            model_path (str): Path to trained model (.pth file)
            model_type (str): Model type ('GRU' or 'LSTM')
            yolo_model_path (str): Path to YOLO pose model
        """
        self.model_path = model_path
        self.model_type = model_type.upper()
        self.yolo_model_path = yolo_model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.class_names = ['Stand', 'Walk', 'Fall']
        self.class_colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255)]  # 绿色、黄色、红色
        
        # Body parts and connections for pose visualization
        self.BODY_PARTS_NAMES = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        self.BODY_CONNECTIONS_DRAW = {
            "legs": ([("left_hip", "left_knee"), ("left_knee", "left_ankle"), 
                     ("right_hip", "right_knee"), ("right_knee", "right_ankle")], (255, 0, 0)),
            "arms": ([("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), 
                     ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")], (0, 255, 0)),
            "head": ([("nose", "left_eye"), ("nose", "right_eye"), 
                     ("left_eye", "left_ear"), ("right_eye", "right_ear")], (0, 0, 255)),
            "torso": ([("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), 
                      ("right_shoulder", "right_hip"), ("left_hip", "right_hip")], (255, 255, 0))
        }
        
        # Load models
        self.load_models()
        
        # Initialize tracker
        self.byte_tracker = sv.ByteTrack()
        
    def load_models(self):
        """Load YOLO and fall detection models."""
        print(f"Loading YOLO model from: {self.yolo_model_path}")
        if not os.path.exists(self.yolo_model_path):
            raise FileNotFoundError(f"YOLO model not found: {self.yolo_model_path}")
        
        self.yolo_model = YOLO(self.yolo_model_path, verbose=False)
        
        print(f"Loading {self.model_type} model from: {self.model_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Fall detection model not found: {self.model_path}")
        
        # Create model based on type
        if self.model_type == 'GRU':
            self.fall_model = FallDetectionGRU(
                input_size=34, 
                hidden_size=64, 
                num_layers=2, 
                output_size=3,
                dropout_prob=0.6
            )
        elif self.model_type == 'LSTM':
            self.fall_model = FallDetectionLSTM(
                input_size=34, 
                hidden_size=64, 
                num_layers=2, 
                output_size=3,
                dropout_prob=0.6
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}. Use 'GRU' or 'LSTM'")
        
        # Load model weights
        self.fall_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.fall_model.to(self.device)
        self.fall_model.eval()
        
        print(f"{self.model_type} model loaded successfully on device: {self.device}")
    
    def draw_pose_keypoints(self, frame, keypoints_2d):
        """Draw pose keypoints and connections on frame."""
        if keypoints_2d.size(0) == 0:
            return frame
        
        # Create body parts dictionary
        body = {part: keypoints_2d[i] for i, part in enumerate(self.BODY_PARTS_NAMES)}
        
        # Draw connections
        for group, (connections, color) in self.BODY_CONNECTIONS_DRAW.items():
            for part_a, part_b in connections:
                if part_a in body and part_b in body:
                    x1, y1 = map(int, body[part_a])
                    x2, y2 = map(int, body[part_b])
                    
                    # Skip if coordinates are invalid
                    if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
                        continue
                    
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw keypoints
        for group, (connections, color) in self.BODY_CONNECTIONS_DRAW.items():
            for part_a, _ in connections:
                if part_a in body:
                    x, y = map(int, body[part_a])
                    if x != 0 and y != 0:
                        cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame
    
    def predict_activity(self, keypoints_sequence):
        """Predict activity from keypoints sequence."""
        if len(keypoints_sequence) == 0:
            return 0, np.array([0.33, 0.33, 0.34])
        
        # Convert to tensor
        keypoints_tensor = torch.tensor(keypoints_sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.fall_model(keypoints_tensor)
            # 使用softmax获取概率分布
            probabilities = torch.softmax(prediction, dim=1)
            probabilities_np = probabilities[0].cpu().numpy()
            predicted_class = np.argmax(probabilities_np)
        
        return predicted_class, probabilities_np

    def process_image(self, image_path, output_path=None, show_pose=True):
        """Process single image for activity detection."""
        print(f"Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        height, width = frame.shape[:2]
        annotated_frame = frame.copy()
        
        # Run YOLO detection
        results = self.yolo_model(frame)[0]
        
        # Process detections
        if len(results.keypoints.xy) > 0:
            detections = sv.Detections.from_ultralytics(results)
            
            # Annotate with bounding boxes
            bounding_box_annotator = sv.BoxAnnotator(thickness=2)
            label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
            
            labels = [f'{results.names[class_id]} {confidence:.2f}' 
                    for class_id, confidence in zip(detections.class_id, detections.confidence)]
            
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            # Process each person
            for person_idx in range(len(results.keypoints.xy)):
                keypoints_2d = results.keypoints.xy[person_idx].cpu()
                
                if keypoints_2d.size(0) == 0:
                    continue
                
                # Extract keypoints for activity detection
                keypoints_flat = keypoints_2d.numpy().flatten()
                
                # For single image, repeat the keypoints to create sequence
                sequence_length = 20
                keypoints_sequence = np.tile(keypoints_flat, (sequence_length, 1))
                
                # Predict activity
                predicted_class, probabilities = self.predict_activity(keypoints_sequence)
                
                # Draw pose if enabled
                if show_pose:
                    annotated_frame = self.draw_pose_keypoints(annotated_frame, keypoints_2d)
                
                # Add activity detection result
                class_name = self.class_names[predicted_class]
                class_color = self.class_colors[predicted_class]
                confidence = probabilities[predicted_class]
                
                status_text = f'{self.model_type} - {class_name}: {confidence:.3f}'
                
                cv2.putText(annotated_frame, status_text, (10, 25 + person_idx * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 1, cv2.LINE_AA)
                
                # Show all probabilities
                prob_text = f'Stand:{probabilities[0]:.2f} Walk:{probabilities[1]:.2f} Fall:{probabilities[2]:.2f}'
                cv2.putText(annotated_frame, prob_text, (10, 45 + person_idx * 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Highlight person based on activity
                if hasattr(results, 'boxes') and results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().numpy()
                    if person_idx < len(boxes):
                        x1, y1, x2, y2 = map(int, boxes[person_idx])
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_color, 3)
                        cv2.putText(annotated_frame, f'{class_name.upper()}!', (x2 + 10, y1 + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_color, 2, cv2.LINE_AA)
        
        # Add image info
        cv2.putText(annotated_frame, f'Model: {self.model_type}', (10, height - 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Size: {width}x{height}', (10, height - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', (10, height - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Save result
        if output_path:
            cv2.imwrite(output_path, annotated_frame)
            print(f"Processed image saved to: {output_path}")
        else:
            # Generate default output path
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = f"{base_name}_processed.jpg"
            cv2.imwrite(output_path, annotated_frame)
            print(f"Processed image saved to: {output_path}")
        
        return annotated_frame

    def process_video(self, video_path, output_path=None, sequence_length=20, show_pose=True):
        """Process video for activity detection."""
        print(f"Processing video: {video_path}")
        print(f"Using {self.model_type} model for activity detection")
        print("Running in headless mode (no display window)")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer
        out = None
        if output_path:
            # 确保输出路径有正确的扩展名
            if not output_path.endswith('.mp4'):
                output_path = output_path + '.mp4'
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"Warning: Could not initialize video writer for {output_path}")
                print("Video will be processed but not saved")
                out = None
            else:
                print(f"Recording output video to: {output_path}")
        
        # Initialize annotators
        bounding_box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.4)
        trace_annotator = sv.TraceAnnotator(thickness=2)
        
        # Keypoints buffer for sequence
        keypoints_buffer = []
        frame_count = 0
        activity_detections = {'stand': [], 'walk': [], 'fall': []}
        
        # 当前活动状态
        current_activity = 0
        current_probabilities = np.array([0.33, 0.33, 0.34])
        
        print("Starting video processing...")
        print("Press Ctrl+C to stop processing")
        
        try:
            while True:
                start_time = time.time()
                ret, frame = cap.read()
                if not ret:
                    print("Video processing completed")
                    break
                
                frame_count += 1
                annotated_frame = frame.copy()
                
                # Run YOLO detection
                results = self.yolo_model(frame)[0]
                
                if len(results.keypoints.xy) > 0:
                    detections = sv.Detections.from_ultralytics(results)
                    detections = self.byte_tracker.update_with_detections(detections)
                    
                    # Create labels
                    labels = [f'#{tracker_id} {results.names[class_id]} {confidence:.2f}' 
                            for class_id, confidence, tracker_id in 
                            zip(detections.class_id, detections.confidence, detections.tracker_id)]
                    
                    # Annotate frame
                    annotated_frame = trace_annotator.annotate(annotated_frame, detections)
                    annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
                    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
                    
                    # Process each person
                    for person_idx in range(len(results.keypoints.xy)):
                        keypoints_2d = results.keypoints.xy[person_idx].cpu()
                        
                        if keypoints_2d.size(0) == 0:
                            continue
                        
                        # Extract and buffer keypoints
                        keypoints_flat = keypoints_2d.numpy().flatten()
                        keypoints_buffer.append(keypoints_flat)
                        
                        # Maintain buffer size
                        if len(keypoints_buffer) > sequence_length:
                            keypoints_buffer.pop(0)
                        
                        # Predict activity if we have enough frames
                        if len(keypoints_buffer) == sequence_length:
                            keypoints_sequence = np.array(keypoints_buffer, dtype=np.float32)
                            predicted_class, probabilities = self.predict_activity(keypoints_sequence)
                            
                            # 更新当前状态
                            current_activity = predicted_class
                            current_probabilities = probabilities
                            
                            class_name = self.class_names[predicted_class]
                            class_color = self.class_colors[predicted_class]
                            
                            # Record detections
                            activity_detections[class_name.lower()].append(frame_count)
                            
                            # Print detection
                            if predicted_class != 0:  # 如果不是站立，则打印
                                print(f"Frame {frame_count}: {self.model_type} detected {class_name}! Probability: {probabilities[predicted_class]:.3f}")
                            
                            # Highlight if specific activity detected
                            if hasattr(results, 'boxes') and results.boxes is not None:
                                boxes = results.boxes.xyxy.cpu().numpy()
                                if person_idx < len(boxes):
                                    x1, y1, x2, y2 = map(int, boxes[person_idx])
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), class_color, 3)
                                    cv2.putText(annotated_frame, f'{class_name.upper()}!', (x2 + 10, y1 + 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 2, cv2.LINE_AA)
                        
                        # Draw pose if enabled
                        if show_pose:
                            annotated_frame = self.draw_pose_keypoints(annotated_frame, keypoints_2d)
                
                # 显示当前活动状态
                class_name = self.class_names[current_activity]
                class_color = self.class_colors[current_activity]
                
                # Display prediction with probabilities
                status_text = f'{self.model_type} - {class_name}: {current_probabilities[current_activity]:.3f}'
                cv2.putText(annotated_frame, status_text, (10, 100), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, class_color, 1, cv2.LINE_AA)
                
                # Show all probabilities
                prob_text = f'Stand:{current_probabilities[0]:.2f} Walk:{current_probabilities[1]:.2f} Fall:{current_probabilities[2]:.2f}'
                cv2.putText(annotated_frame, prob_text, (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Calculate and display FPS
                processing_time = time.time() - start_time
                fps_real = 1 / processing_time if processing_time > 0 else 0
                
                # Add frame information
                cv2.putText(annotated_frame, f'Model: {self.model_type}', (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f'Size: {width}x{height}', (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f'FPS: {fps_real:.1f}', (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f'Frame: {frame_count}/{total_frames}', (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Save frame if recording
                if out is not None and out.isOpened():
                    out.write(annotated_frame)
                    
                    # 每100帧打印一次保存状态
                    if frame_count % 100 == 0:
                        print(f"Saving frame {frame_count} to output video...")
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames), FPS: {fps_real:.1f}")
        
        except KeyboardInterrupt:
            print("\nUser interrupted, stopping processing...")
        
        finally:
            # Cleanup
            cap.release()
            if out is not None:
                out.release()
                if output_path and os.path.exists(output_path):
                    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                    print(f"Output video saved to: {output_path} (Size: {file_size:.1f} MB)")
                else:
                    print("Warning: Output video may not have been saved properly")
            
            # Summary
            print(f"\nProcessing Summary:")
            print(f"Model used: {self.model_type}")
            print(f"Total frames processed: {frame_count}")
            for activity, frames in activity_detections.items():
                if frames:
                    print(f"{activity.capitalize()} detections: {len(frames)} frames")
                    if len(frames) <= 10:
                        print(f"  Detected at frames: {frames}")
                    else:
                        print(f"  First 10 frames: {frames[:10]}...")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description='Activity Detection Inference (Headless Mode)')
    parser.add_argument('input', help='Input video or image path')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth file)')
    parser.add_argument('--model-type', default='GRU', choices=['GRU', 'LSTM'], 
                       help='Model type (default: GRU)')
    parser.add_argument('--yolo-model', default='models/yolo11n-pose.pt', 
                       help='Path to YOLO pose model')
    parser.add_argument('--output', help='Output path for processed video/image')
    parser.add_argument('--sequence-length', type=int, default=20, 
                       help='Sequence length for video processing (default: 20)')
    parser.add_argument('--no-pose', action='store_true', 
                       help='Disable pose keypoints visualization')
    
    args = parser.parse_args()
    
    try:
        # Initialize inference
        inference = FallDetectionInference(
            model_path=args.model,
            model_type=args.model_type,
            yolo_model_path=args.yolo_model
        )
        
        # Check if input is image or video
        input_ext = os.path.splitext(args.input)[1].lower()
        
        if input_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Process image
            print(f"Processing image with {args.model_type} model...")
            inference.process_image(
                image_path=args.input,
                output_path=args.output,
                show_pose=not args.no_pose
            )
        elif input_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
            # Process video
            print(f"Processing video with {args.model_type} model...")
            inference.process_video(
                video_path=args.input,
                output_path=args.output,
                sequence_length=args.sequence_length,
                show_pose=not args.no_pose
            )
        else:
            print(f"Unsupported file format: {input_ext}")
            print("Supported formats: Images (.jpg, .png, etc.), Videos (.mp4, .avi, etc.)")
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
