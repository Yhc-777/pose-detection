import cv2
import time
import numpy as np
import torch
from ultralytics import YOLO
import supervision as sv
import os
import sys

def video_detect_falls(video_path, yolo_model_path, gru_model_path, fall_threshold=0.95, scale_percent=100, sequence_length=20, show_pose=False, record=False):
    """
    Detecta caídas en un video usando YOLO para pose detection y un modelo GRU para clasificación.
    
    Args:
        video_path (str): Ruta al video de entrada
        yolo_model_path (str): Ruta al modelo YOLO para pose detection
        gru_model_path (str): Ruta al modelo GRU entrenado (archivo .pth)
        fall_threshold (float): Umbral de probabilidad para detectar caídas
        scale_percent (int): Porcentaje de escalado del video para visualización
        sequence_length (int): Longitud de la secuencia de keypoints para el modelo
        show_pose (bool): Si mostrar las poses detectadas
        record (bool): Si grabar el video con anotaciones
    """
    # Add src to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(os.path.join(src_dir, 'src'))
    
    from models.fall_detection_gru import FallDetectionGRU
    
    # Definir conexiones del cuerpo para visualización
    BODY_PARTS_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    BODY_CONNECTIONS_DRAW = {
        "piernas": ([("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")], (255, 0, 0)),
        "brazos": ([("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")], (0, 255, 0)),
        "cabeza": ([("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear")], (0, 0, 255)),
        "torso": ([("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip")], (255, 255, 0))
    }
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    model_yolo = YOLO(yolo_model_path, verbose=False)
    byte_tracker = sv.ByteTrack()
    
    # Cargar modelo GRU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FallDetectionGRU(
        input_size=34, 
        hidden_size=64, 
        num_layers=2, 
        output_size=1, 
        dropout_prob=0.6
    )
    
    # Cargar pesos del modelo
    if isinstance(gru_model_path, str):
        if os.path.exists(gru_model_path):
            model.load_state_dict(torch.load(gru_model_path, map_location=device))
        else:
            raise FileNotFoundError(f"Model file not found: {gru_model_path}")
    else:
        # Si gru_model_path es el modelo ya cargado
        model = gru_model_path
    
    model.to(device)
    model.eval()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    text_scale = 2 if scale_percent < 80 else 0.5
    text_thickness = 4 if scale_percent < 80 else 1
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale)
    trace_annotator = sv.TraceAnnotator(thickness=2)
    
    keypoints_buffer = []

    # Configurar el grabador de video si record=True
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
        output_path = 'output_fall_detection.mp4'  # Nombre del archivo de salida
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording output video to: {output_path}")

    frame_count = 0
    print("Starting video processing...")
    print("Press 'q' to quit")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed")
            break
        
        results = model_yolo(frame)[0]
        annotated_frame = frame.copy()
        
        if len(results.keypoints.xy) > 0:
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)
            
            # Crear etiquetas para las detecciones
            labels = [f'#{tracker_id} {results.names[class_id]} {confidence:.2f}' 
                      for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id)]
            
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            for person_idx in range(len(results.keypoints.xy)):
                if results.keypoints.xy[person_idx].size(0) == 0:
                    continue
                
                # Extraer keypoints y agregar al buffer
                keypoints = results.keypoints.xy[person_idx].cpu().numpy().flatten()
                keypoints_buffer.append(keypoints)
                
                if len(keypoints_buffer) > sequence_length:
                    keypoints_buffer.pop(0)
                
                # Realizar predicción si tenemos suficientes frames
                if len(keypoints_buffer) == sequence_length:
                    keypoints_sequence = np.array(keypoints_buffer, dtype=np.float32)
                    keypoints_tensor = torch.tensor(keypoints_sequence).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        prediction = model(keypoints_tensor)
                        fall_probability = prediction.item()
                        is_fall = fall_probability > fall_threshold
                    
                    # Mostrar información de predicción
                    cv2.putText(annotated_frame, f'Fall: {is_fall} ({fall_probability:.2f})', 
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 255), text_thickness, cv2.LINE_AA)
                    
                    # Si se detecta caída, resaltar la persona
                    if is_fall:
                        if hasattr(results, 'boxes') and results.boxes is not None:
                            boxes = results.boxes.xyxy.cpu().numpy()
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(annotated_frame, 'FALL DETECTED!', (x1 + 10, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
                
                # Dibujar pose si está habilitado
                if show_pose:
                    keypoints_2d = results.keypoints.xy[person_idx].cpu()
                    body = {part: keypoints_2d[i] for i, part in enumerate(BODY_PARTS_NAMES)}
                                
                    for group, (connections, color) in BODY_CONNECTIONS_DRAW.items():
                        for part_a, part_b in connections:
                            x1, y1 = map(int, body[part_a])
                            x2, y2 = map(int, body[part_b])
                            
                            if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
                                continue
                            
                            cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                        for part_a, _ in connections:
                            x, y = map(int, body[part_a])
                            if x == 0 or y == 0:
                                continue
                            cv2.circle(annotated_frame, (x, y), 4, color, -2)
        
        # Calcular FPS y agregar información
        processing_time = time.time() - start_time
        fps_real = 1 / processing_time if processing_time > 0 else 0
        
        cv2.putText(annotated_frame, f'{width}x{height}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Real: {fps_real:.2f} FPS', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), text_thickness, cv2.LINE_AA)
        
        # Guardar frame si está grabando
        if record:
            out.write(annotated_frame)
        
        # Mostrar frame redimensionado
        display_frame = cv2.resize(annotated_frame, (new_width, new_height))
        cv2.imshow('Fall Detection', display_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, FPS: {fps_real:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video processing...")
            break

    cap.release()
    if record:
        out.release()
        print(f"Output video saved to: {output_path}")
    cv2.destroyAllWindows()
    print(f"Video processing completed. Total frames processed: {frame_count}")


def video_detect_activities(video_path, yolo_model_path, model_path, model_type='GRU', 
                           scale_percent=100, sequence_length=20, show_pose=False, record=False):
    """
    Detecta actividades (Stand, Walk, Fall) en un video usando YOLO para pose detection y un modelo GRU/LSTM para clasificación.
    
    Args:
        video_path (str): Ruta al video de entrada
        yolo_model_path (str): Ruta al modelo YOLO para pose detection
        model_path (str): Ruta al modelo entrenado (archivo .pth)
        model_type (str): Tipo de modelo ('GRU' o 'LSTM')
        scale_percent (int): Porcentaje de escalado del video para visualización
        sequence_length (int): Longitud de la secuencia de keypoints para el modelo
        show_pose (bool): Si mostrar las poses detectadas
        record (bool): Si grabar el video con anotaciones
    """
    # Add src to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(os.path.dirname(current_dir))
    sys.path.append(os.path.join(src_dir, 'src'))
    
    if model_type.upper() == 'GRU':
        from models.fall_detection_gru import FallDetectionGRU as Model
    elif model_type.upper() == 'LSTM':
        from models.fall_detection_lstm import FallDetectionLSTM as Model
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Definir conexiones del cuerpo para visualización
    BODY_PARTS_NAMES = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    BODY_CONNECTIONS_DRAW = {
        "piernas": ([("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")], (255, 0, 0)),
        "brazos": ([("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist")], (0, 255, 0)),
        "cabeza": ([("nose", "left_eye"), ("nose", "right_eye"), ("left_eye", "left_ear"), ("right_eye", "right_ear")], (0, 0, 255)),
        "torso": ([("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip")], (255, 255, 0))
    }
    
    # Nombres de las clases
    CLASS_NAMES = ['Stand', 'Walk', 'Fall']
    CLASS_COLORS = [(0, 255, 0), (255, 255, 0), (0, 0, 255)]  # Verde, Amarillo, Rojo
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    model_yolo = YOLO(yolo_model_path, verbose=False)
    byte_tracker = sv.ByteTrack()
    
    # Cargar modelo de actividades
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(
        input_size=34, 
        hidden_size=64, 
        num_layers=2, 
        output_size=3,  # 3 clases: Stand, Walk, Fall
        dropout_prob=0.6
    )
    
    # Cargar pesos del modelo
    if isinstance(model_path, str):
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
    else:
        # Si model_path es el modelo ya cargado
        model = model_path
    
    model.to(device)
    model.eval()
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    
    text_scale = 2 if scale_percent < 80 else 0.5
    text_thickness = 4 if scale_percent < 80 else 1
    
    bounding_box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=text_thickness, text_scale=text_scale)
    trace_annotator = sv.TraceAnnotator(thickness=2)
    
    keypoints_buffer = []

    # Configurar el grabador de video si record=True
    if record:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para .mp4
        output_path = 'output_activity_detection.mp4'  # Nombre del archivo de salida
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Recording output video to: {output_path}")

    frame_count = 0
    print("Starting activity detection...")
    print("Press 'q' to quit")

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Video processing completed")
            break
        
        results = model_yolo(frame)[0]
        annotated_frame = frame.copy()
        
        if len(results.keypoints.xy) > 0:
            detections = sv.Detections.from_ultralytics(results)
            detections = byte_tracker.update_with_detections(detections)
            
            # Crear etiquetas para las detecciones
            labels = [f'#{tracker_id} {results.names[class_id]} {confidence:.2f}' 
                      for class_id, confidence, tracker_id in zip(detections.class_id, detections.confidence, detections.tracker_id)]
            
            annotated_frame = trace_annotator.annotate(annotated_frame, detections)
            annotated_frame = bounding_box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            
            for person_idx in range(len(results.keypoints.xy)):
                if results.keypoints.xy[person_idx].size(0) == 0:
                    continue
                
                # Extraer keypoints y agregar al buffer
                keypoints = results.keypoints.xy[person_idx].cpu().numpy().flatten()
                keypoints_buffer.append(keypoints)
                
                if len(keypoints_buffer) > sequence_length:
                    keypoints_buffer.pop(0)
                
                # Realizar predicción si tenemos suficientes frames
                if len(keypoints_buffer) == sequence_length:
                    keypoints_sequence = np.array(keypoints_buffer, dtype=np.float32)
                    keypoints_tensor = torch.tensor(keypoints_sequence).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        predictions = model(keypoints_tensor)
                        probabilities = torch.softmax(predictions, dim=1)
                        predicted_class = torch.argmax(probabilities, dim=1).item()
                        confidence = probabilities[0][predicted_class].item()
                    
                    # Obtener nombre de la clase y color
                    class_name = CLASS_NAMES[predicted_class]
                    class_color = CLASS_COLORS[predicted_class]
                    
                    # Mostrar información de predicción
                    cv2.putText(annotated_frame, f'Activity: {class_name} ({confidence:.2f})', 
                                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, text_scale, class_color, text_thickness, cv2.LINE_AA)
                    
                    # Si se detecta caída, resaltar la persona
                    if predicted_class == 2:  # Fall
                        if hasattr(results, 'boxes') and results.boxes is not None:
                            boxes = results.boxes.xyxy.cpu().numpy()
                            for box in boxes:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                cv2.putText(annotated_frame, 'FALL DETECTED!', (x1 + 10, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
                    
                    # Mostrar todas las probabilidades
                    for i, (prob, name) in enumerate(zip(probabilities[0], CLASS_NAMES)):
                        cv2.putText(annotated_frame, f'{name}: {prob:.3f}', 
                                    (10, 100 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, text_scale*0.6, 
                                    CLASS_COLORS[i], text_thickness, cv2.LINE_AA)
                
                # Dibujar pose si está habilitado
                if show_pose:
                    keypoints_2d = results.keypoints.xy[person_idx].cpu()
                    body = {part: keypoints_2d[i] for i, part in enumerate(BODY_PARTS_NAMES)}
                                
                    for group, (connections, color) in BODY_CONNECTIONS_DRAW.items():
                        for part_a, part_b in connections:
                            x1, y1 = map(int, body[part_a])
                            x2, y2 = map(int, body[part_b])
                            
                            if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
                                continue
                            
                            cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                        for part_a, _ in connections:
                            x, y = map(int, body[part_a])
                            if x == 0 or y == 0:
                                continue
                            cv2.circle(annotated_frame, (x, y), 4, color, -2)
        
        # Calcular FPS y agregar información
        processing_time = time.time() - start_time
        fps_real = 1 / processing_time if processing_time > 0 else 0
        
        cv2.putText(annotated_frame, f'{width}x{height}', (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 255, 0), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Real: {fps_real:.2f} FPS', (10, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 0, 255), text_thickness, cv2.LINE_AA)
        cv2.putText(annotated_frame, f'Persons: {len(results.keypoints.xy)}', (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, text_scale, (255, 0, 0), text_thickness, cv2.LINE_AA)
        
        # Guardar frame si está grabando
        if record:
            out.write(annotated_frame)
        
        # Mostrar frame redimensionado
        display_frame = cv2.resize(annotated_frame, (new_width, new_height))
        cv2.imshow('Activity Detection', display_frame)
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames, FPS: {fps_real:.2f}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video processing...")
            break

    cap.release()
    if record:
        out.release()
        print(f"Output video saved to: {output_path}")
    cv2.destroyAllWindows()
    print(f"Video processing completed. Total frames processed: {frame_count}")
