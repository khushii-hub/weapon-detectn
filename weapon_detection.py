from ultralytics import YOLO
import cv2
import numpy as np

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Class names - ensure correct order
CLASS_NAMES = ['drill', 'knife']

def preprocess_image(image):
    """
    Preprocess the image to improve detection accuracy
    """
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Resize image while maintaining aspect ratio
    target_size = 640
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    image = cv2.resize(image, (new_w, new_h))
    
    # Add padding to make it square
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    top_pad = pad_h // 2
    bottom_pad = pad_h - top_pad
    left_pad = pad_w // 2
    right_pad = pad_w - left_pad
    image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, 
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # Enhance image for better detection
    image = cv2.convertScaleAbs(image, alpha=1.1, beta=5)  # Slight contrast enhancement
    image = cv2.GaussianBlur(image, (3, 3), 0)  # Reduce noise
    
    return image, (scale, (left_pad, top_pad))

def postprocess_detections(results, scale, padding):
    """
    Convert detections back to original image coordinates
    """
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()
        
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            # Convert coordinates back to original image space
            x1, y1, x2, y2 = box
            x1 = (x1 - padding[0]) / scale
            y1 = (y1 - padding[1]) / scale
            x2 = (x2 - padding[0]) / scale
            y2 = (y2 - padding[1]) / scale
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, 640))
            y1 = max(0, min(y1, 640))
            x2 = max(0, min(x2, 640))
            y2 = max(0, min(y2, 640))
            
            # Calculate box area and aspect ratio
            box_area = (x2 - x1) * (y2 - y1)
            aspect_ratio = (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
            
            # Only include high confidence detections with reasonable box size and aspect ratio
            if conf > 0.4:  # Higher confidence threshold
                class_name = CLASS_NAMES[int(class_id)]
                
                # Class-specific checks
                if class_name == 'knife':
                    if conf < 0.5:  # Higher threshold for knives
                        continue
                    if box_area < 1000:  # Minimum size for knife detection
                        continue
                    if aspect_ratio < 0.2 or aspect_ratio > 5:  # Knife aspect ratio check
                        continue
                elif class_name == 'drill':
                    if conf < 0.45:  # Higher threshold for drills
                        continue
                    if box_area < 2000:  # Minimum size for drill detection
                        continue
                    if aspect_ratio < 0.5 or aspect_ratio > 2:  # Drill aspect ratio check
                        continue
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': class_name
                })
    
    return detections

def detect_weapons(image):
    """
    Detect weapons in the given image using YOLO model.
    
    Args:
        image: OpenCV image (numpy array)
        
    Returns:
        dict: Detection results containing bounding boxes and confidence scores
    """
    # Preprocess the image
    processed_image, (scale, padding) = preprocess_image(image)
    
    # Run inference with higher confidence threshold
    results = model(processed_image, conf=0.4)  # Higher confidence threshold
    
    # Postprocess detections
    detections = postprocess_detections(results, scale, padding)
    
    return {
        'detected': len(detections) > 0,
        'detections': detections,
        'count': len(detections)
    } 