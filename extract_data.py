#!/usr/bin/env python3
"""
Main script for pose detection and fall detection data extraction.
This script processes videos to extract keypoints and generate analysis plots.
"""

import os
import sys
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Linux
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from ultralytics import YOLO
import supervision as sv

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
from body import BODY_PARTS_NAMES, BODY_CONNECTIONS_DRAW, BODY_CONNECTIONS, draw_skeleton
from extract_keypoints_from_video import extract_keypoints_from_video

# Set style for plots
sns.set_style('darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'  # Use default Linux font

class PoseDataExtractor:
    def __init__(self, model_path='models/yolo11n-pose.pt'):
        """Initialize the pose data extractor."""
        self.model = YOLO(model_path)
        self.byte_tracker = sv.ByteTrack()
        self.output_dir = 'output'
        self.plots_dir = os.path.join(self.output_dir, 'plots')
        self.data_dir = os.path.join(self.output_dir, 'data')
        
        # Create output directories
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
    def process_video_realtime(self, video_path, save_video=False):
        """Process video in real-time and save annotated frames."""
        print(f"Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {frame_width}x{frame_height}, {fps} FPS")
        
        # Video writer setup if saving
        if save_video:
            output_video_path = os.path.join(self.output_dir, 'annotated_video.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        processing_times = []
        
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("Video processing completed")
                break
            
            results = self.model(frame)[0]
            annotated_frame = frame.copy()
            
            if len(results.keypoints.xy) > 0:
                detections = sv.Detections.from_ultralytics(results)
                detections = self.byte_tracker.update_with_detections(detections)
                
                # Draw pose keypoints
                for person_idx in range(len(results.keypoints.xy)):
                    if results.keypoints.xy[person_idx].size(0) == 0:
                        continue
                    # 修复: 添加 .cpu() 来处理GPU tensor
                    keypoints = results.keypoints.xy[person_idx].cpu()
                    body = {part: keypoints[i] for i, part in enumerate(BODY_PARTS_NAMES)}
                    
                    # Draw connections
                    for group, (connections, color) in BODY_CONNECTIONS_DRAW.items():
                        for part_a, part_b in connections:
                            x1, y1 = map(int, body[part_a])
                            x2, y2 = map(int, body[part_b])
                            if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
                                continue
                            cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw keypoints
                        for part_a, _ in connections:
                            x, y = map(int, body[part_a])
                            if x == 0 or y == 0:
                                continue
                            cv2.circle(annotated_frame, (x, y), 4, color, -2)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            fps_real = 1 / processing_time if processing_time > 0 else 0
            
            # Add information text
            cv2.putText(annotated_frame, f'{frame_width}x{frame_height}', 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f'Real: {fps_real:.2f} FPS', 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            
            if save_video:
                out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, avg FPS: {np.mean([1/t for t in processing_times[-100:]]):.2f}")
        
        cap.release()
        if save_video:
            out.release()
            print(f"Annotated video saved to: {output_video_path}")
        
        return processing_times
    
    def extract_single_video_keypoints(self, video_path, sequence_length=400):
        """Extract keypoints from a single video."""
        print(f"Extracting keypoints from: {video_path}")
        keypoints = extract_keypoints_from_video(video_path, self.model, 
                                               sequence_length=sequence_length, 
                                               save=False)
        
        output_path = os.path.join(self.data_dir, 'single_video_keypoints.npy')
        np.save(output_path, keypoints)
        print(f"Keypoints saved to: {output_path}")
        
        return keypoints
    
    def visualize_keypoints_trajectory(self, keypoints, save_name='keypoints_trajectory.png'):
        """Visualize keypoints trajectory."""
        print("Creating keypoints trajectory visualization...")
        
        # Normalize keypoints
        keypoints_norm = keypoints / np.nanmax(keypoints)
        keypoints_norm[keypoints_norm == 0] = np.nan
        
        # Extract foot positions
        left_foot = keypoints_norm[:, 15:17]
        
        plt.figure(figsize=(12, 6))
        
        # Plot left foot trajectory
        plt.subplot(1, 2, 1)
        plt.plot(left_foot[:, 0], label='X coordinate')
        plt.plot(left_foot[:, 1], label='Y coordinate')
        plt.legend()
        plt.xlabel('Frame')
        plt.ylabel('Position (normalized)')
        plt.title('Left Foot Position Over Time')
        plt.grid(True)
        
        # Plot motion capture visualization
        plt.subplot(1, 2, 2)
        num_frames = keypoints_norm.shape[0]
        
        left_foot_trace = []
        right_foot_trace = []
        left_hand_trace = []
        right_hand_trace = []
        
        for i in range(0, num_frames, 10):
            kps = keypoints_norm[i].reshape(-1, 2)
            
            # Draw skeleton connections
            for joint1, joint2 in BODY_CONNECTIONS:
                if not (np.isnan(kps[joint1]).any() or np.isnan(kps[joint2]).any()):
                    x_values = [kps[joint1, 0], kps[joint2, 0]]
                    y_values = [kps[joint1, 1], kps[joint2, 1]]
                    plt.plot(x_values, y_values, 'bo-', markersize=2, alpha=0.3)
            
            # Collect traces
            if len(kps) > 15:  # 确保索引不越界
                left_foot_trace.append(kps[15])
            if len(kps) > 16:
                right_foot_trace.append(kps[16])
            if len(kps) > 9:
                left_hand_trace.append(kps[9])
            if len(kps) > 10:
                right_hand_trace.append(kps[10])
        
        # Plot traces
        traces = [
            (left_foot_trace, 'red', 'Left foot'),
            (right_foot_trace, 'green', 'Right foot'),
            (left_hand_trace, 'blue', 'Left hand'),
            (right_hand_trace, 'yellow', 'Right hand')
        ]
        
        for trace, color, label in traces:
            if len(trace) > 0:
                trace_array = np.array(trace)
                valid_points = ~np.isnan(trace_array).any(axis=1)
                if np.any(valid_points):
                    plt.plot(trace_array[valid_points, 0], trace_array[valid_points, 1], 
                            '--', color=color, alpha=0.7, linewidth=2)
                    plt.scatter(trace_array[valid_points, 0], trace_array[valid_points, 1], 
                               color=color, s=30, label=label)
        
        plt.gca().invert_yaxis()
        plt.title('Motion Capture Visualization')
        plt.legend()
        plt.axis('equal')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trajectory plot saved to: {save_path}")
    
    def compare_original_vs_imputed(self, keypoints_original, keypoints_imputed, 
                                  save_name='original_vs_imputed.png'):
        """Compare original vs imputed keypoints."""
        print("Creating original vs imputed comparison...")
        
        plt.figure(figsize=(15, 10))
        
        # Time series comparison
        left_foot_original = keypoints_original[:, 15:17]
        left_foot_imputed = keypoints_imputed[:, 15:17]
        
        plt.subplot(2, 2, 1)
        plt.plot(left_foot_original[:, 0], label='X coordinate')
        plt.plot(left_foot_original[:, 1], label='Y coordinate')
        plt.legend()
        plt.title('Left Foot Position - Original')
        plt.xlabel('Frame')
        plt.ylabel('Position')
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(left_foot_imputed[:, 0], label='X coordinate')
        plt.plot(left_foot_imputed[:, 1], label='Y coordinate')
        plt.legend()
        plt.title('Left Foot Position - Imputed (KNN)')
        plt.xlabel('Frame')
        plt.ylabel('Position')
        plt.grid(True)
        
        # Skeleton comparison
        frame_idx = min(40, len(keypoints_original) - 1)  # 防止索引越界
        plt.subplot(2, 2, 3)
        ax1 = plt.gca()
        draw_skeleton(ax1, keypoints_original[frame_idx].reshape(-1, 2), 'Original Skeleton')
        
        plt.subplot(2, 2, 4)
        ax2 = plt.gca()
        draw_skeleton(ax2, keypoints_imputed[frame_idx].reshape(-1, 2), 'Imputed Skeleton (KNN)')
        
        plt.tight_layout()
        save_path = os.path.join(self.plots_dir, save_name)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to: {save_path}")
    
    def analyze_video_statistics(self, folder_path):
        """Analyze video statistics in a folder."""
        print(f"Analyzing videos in: {folder_path}")
        
        frame_counts = []
        video_files = []
        
        for filename in os.listdir(folder_path):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(folder_path, filename)
                cap = cv2.VideoCapture(video_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                frame_counts.append(frame_count)
                video_files.append(filename)
        
        if frame_counts:
            stats = {
                "min": int(np.min(frame_counts)),
                "max": int(np.max(frame_counts)),
                "mean": float(np.mean(frame_counts)),
                "median": float(np.median(frame_counts)),
                "total_videos": len(frame_counts)
            }
            print(f"Video statistics: {stats}")
            
            # Save statistics
            stats_df = pd.DataFrame({
                'filename': video_files,
                'frame_count': frame_counts
            })
            stats_path = os.path.join(self.data_dir, 'video_statistics.csv')
            stats_df.to_csv(stats_path, index=False)
            print(f"Video statistics saved to: {stats_path}")
            
            return stats
        else:
            print("No video files found in the folder.")
            return None
    
    def process_dataset(self, video_folders, sequence_length=20):
        """Process entire dataset and create training data."""
        print("Processing dataset...")
        
        keypoints_sequences = []
        labels = []
        
        for folder, label in video_folders.items():
            print(f"Processing folder: {folder} (label: {label})")
            
            if not os.path.exists(folder):
                print(f"Warning: Folder {folder} does not exist, skipping...")
                continue
            
            for filename in os.listdir(folder):
                if not filename.endswith('.mp4'):
                    continue
                
                video_path = os.path.join(folder, filename)
                print(f"  Processing: {filename}")
                
                try:
                    keypoints = extract_keypoints_from_video(video_path, self.model, 
                                                           sequence_length=sequence_length)
                    keypoints_sequences.append(keypoints)
                    labels.append(label)
                except Exception as e:
                    print(f"  Error processing {filename}: {e}")
                    continue
        
        if not keypoints_sequences:
            print("No valid sequences extracted!")
            return None, None
        
        keypoints_sequences = np.array(keypoints_sequences)
        labels = np.array(labels)
        
        print(f"Dataset shape: {keypoints_sequences.shape}, Labels shape: {labels.shape}")
        
        # Handle missing values
        print("Handling missing values...")
        for i in range(keypoints_sequences.shape[0]):
            keypoints_sequences[i][keypoints_sequences[i] == 0] = np.nan
        
        # Impute missing values
        print("Imputing missing values with KNN...")
        imputer = KNNImputer(n_neighbors=2)
        original_shape = keypoints_sequences.shape
        keypoints_sequences_imputed = imputer.fit_transform(
            keypoints_sequences.reshape(-1, original_shape[-1])
        ).reshape(original_shape)
        
        # Normalize
        print("Normalizing data...")
        keypoints_sequences_normalized = keypoints_sequences_imputed / np.nanmax(keypoints_sequences_imputed)
        keypoints_sequences_normalized = np.nan_to_num(keypoints_sequences_normalized)
        
        # Save processed data
        sequences_path = os.path.join(self.data_dir, 'keypoints_sequences.npy')
        labels_path = os.path.join(self.data_dir, 'labels.npy')
        
        np.save(sequences_path, keypoints_sequences_normalized)
        np.save(labels_path, labels)
        
        print(f"Processed sequences saved to: {sequences_path}")
        print(f"Labels saved to: {labels_path}")
        print(f"Data range: [{keypoints_sequences_normalized.min():.3f}, {keypoints_sequences_normalized.max():.3f}]")
        
        return keypoints_sequences_normalized, labels

def main():
    """Main function to run the pose data extraction pipeline."""
    print("Starting Pose Data Extraction Pipeline")
    print("=" * 50)
    
    # Initialize extractor
    extractor = PoseDataExtractor()
    
    # Example usage - modify paths as needed
    
    # 1. Process a single video for real-time analysis
    single_video_path = 'data/videos/falls/banana-peel.mp4'
    if os.path.exists(single_video_path):
        print("\n1. Processing single video for real-time analysis...")
        try:
            processing_times = extractor.process_video_realtime(single_video_path, save_video=True)
            print(f"Average processing FPS: {np.mean([1/t for t in processing_times]):.2f}")
        except Exception as e:
            print(f"Error processing single video: {e}")
    else:
        print(f"\n1. Single video not found: {single_video_path}")
    
    # 2. Extract keypoints from a specific video
    target_video = 'data/videos/falls/trust-fall-fail-fall.mp4'
    if os.path.exists(target_video):
        print("\n2. Extracting keypoints from specific video...")
        try:
            keypoints = extractor.extract_single_video_keypoints(target_video, sequence_length=400)
            
            # Visualize trajectory
            extractor.visualize_keypoints_trajectory(keypoints)
            
            # Compare original vs imputed
            keypoints_norm = keypoints / np.nanmax(keypoints)
            keypoints_norm[keypoints_norm == 0] = np.nan
            
            imputer = KNNImputer(n_neighbors=2)
            keypoints_imputed = imputer.fit_transform(keypoints_norm)
            
            extractor.compare_original_vs_imputed(keypoints_norm, keypoints_imputed)
        except Exception as e:
            print(f"Error extracting keypoints: {e}")
    else:
        print(f"\n2. Target video not found: {target_video}")
    
    # 3. Analyze video statistics
    falls_folder = 'data/videos/falls'
    if os.path.exists(falls_folder):
        print("\n3. Analyzing video statistics...")
        try:
            stats = extractor.analyze_video_statistics(falls_folder)
        except Exception as e:
            print(f"Error analyzing video statistics: {e}")
    else:
        print(f"\n3. Falls folder not found: {falls_folder}")
    
    # 4. Process entire dataset
    print("\n4. Processing entire dataset...")
    video_folders = {
        'data/videos/falls': 1,           # Fall videos -> label 1
        'data/videos/normal': 0,          # Normal videos -> label 0
        'data/videos/no_fall_static': 0   # Static no-fall videos -> label 0
    }
    
    try:
        sequences, labels = extractor.process_dataset(video_folders, sequence_length=20)
        
        if sequences is not None:
            print(f"\nDataset processing completed!")
            print(f"Total sequences: {len(sequences)}")
            print(f"Fall sequences: {np.sum(labels == 1)}")
            print(f"Normal sequences: {np.sum(labels == 0)}")
        else:
            print("\nNo sequences were processed successfully.")
    except Exception as e:
        print(f"Error processing dataset: {e}")
    
    print("\n" + "=" * 50)
    print("Pipeline completed! Check the 'output' folder for results.")

if __name__ == "__main__":
    main()
