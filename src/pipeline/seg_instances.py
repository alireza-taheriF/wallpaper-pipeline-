"""
Instance segmentation for object detection and masking
"""

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
except ImportError:
    maskrcnn_resnet50_fpn = None
    MaskRCNN_ResNet50_FPN_Weights = None
    FastRCNNPredictor = None
    MaskRCNNPredictor = None

logger = logging.getLogger(__name__)


class InstanceSegmenter:
    """
    Instance segmentation for indoor objects using Mask R-CNN
    """
    
    # COCO class indices for indoor objects
    INDOOR_OBJECT_CLASSES = {
        'person': 0,
        'bicycle': 1,
        'car': 2,
        'motorcycle': 3,
        'airplane': 4,
        'bus': 5,
        'train': 6,
        'truck': 7,
        'boat': 8,
        'traffic light': 9,
        'fire hydrant': 10,
        'stop sign': 11,
        'parking meter': 12,
        'bench': 13,
        'bird': 14,
        'cat': 15,
        'dog': 16,
        'horse': 17,
        'sheep': 18,
        'cow': 19,
        'elephant': 20,
        'bear': 21,
        'zebra': 22,
        'giraffe': 23,
        'backpack': 24,
        'umbrella': 25,
        'handbag': 26,
        'tie': 27,
        'suitcase': 28,
        'frisbee': 29,
        'skis': 30,
        'snowboard': 31,
        'sports ball': 32,
        'kite': 33,
        'baseball bat': 34,
        'baseball glove': 35,
        'skateboard': 36,
        'surfboard': 37,
        'tennis racket': 38,
        'bottle': 39,
        'wine glass': 40,
        'cup': 41,
        'fork': 42,
        'knife': 43,
        'spoon': 44,
        'bowl': 45,
        'banana': 46,
        'apple': 47,
        'sandwich': 48,
        'orange': 49,
        'broccoli': 50,
        'carrot': 51,
        'hot dog': 52,
        'pizza': 53,
        'donut': 54,
        'cake': 55,
        'chair': 56,
        'couch': 57,
        'potted plant': 58,
        'bed': 59,
        'dining table': 60,
        'toilet': 61,
        'tv': 62,
        'laptop': 63,
        'mouse': 64,
        'remote': 65,
        'keyboard': 66,
        'cell phone': 67,
        'microwave': 68,
        'oven': 69,
        'toaster': 70,
        'sink': 71,
        'refrigerator': 72,
        'book': 73,
        'clock': 74,
        'vase': 75,
        'scissors': 76,
        'teddy bear': 77,
        'hair drier': 78,
        'toothbrush': 79
    }
    
    # Indoor objects we want to detect and mask
    TARGET_OBJECTS = [
        'person', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'teddy bear', 'bottle', 'wine glass', 'cup',
        'bowl', 'lamp', 'shelf', 'picture frame'
    ]
    
    def __init__(
        self,
        device: str = 'auto',
        confidence_threshold: float = 0.1,  # Lower threshold for more objects
        mask_threshold: float = 0.3  # Lower threshold for finer details
    ):
        """
        Initialize instance segmenter
        
        Args:
            device: Device to run on ('auto', 'cpu', 'cuda')
            confidence_threshold: Minimum confidence for detections
            mask_threshold: Threshold for mask binarization
        """
        if maskrcnn_resnet50_fpn is None:
            raise ImportError("torchvision is required for Mask R-CNN")
        
        self.device = self._get_device(device)
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = T.Compose([
            T.ToTensor(),
        ])
        
        logger.info(f"Initialized Mask R-CNN on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine device to use"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'  # Apple Silicon
            else:
                return 'cpu'
        return device
    
    def _create_model(self):
        """Create Mask R-CNN model"""
        # Load pre-trained model
        if MaskRCNN_ResNet50_FPN_Weights is not None:
            weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
            model = maskrcnn_resnet50_fpn(weights=weights)
        else:
            model = maskrcnn_resnet50_fpn(pretrained=True)
        
        return model
    
    def detect_objects(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Detect and segment objects in the image
        
        Args:
            image: Input RGB image (H, W, 3)
            
        Returns:
            Dictionary with object masks and detection info
        """
        original_shape = image.shape[:2]
        
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(input_tensor)
        
        # Process predictions
        pred = predictions[0]  # Remove batch dimension
        
        # Filter detections
        scores = pred['scores'].cpu().numpy().astype(np.float32)
        labels = pred['labels'].cpu().numpy().astype(np.int32)
        masks = pred['masks'].cpu().numpy().astype(np.float32)
        boxes = pred['boxes'].cpu().numpy().astype(np.float32)
        
        # Filter by confidence and target objects
        valid_indices = []
        for i, (score, label) in enumerate(zip(scores, labels)):
            if score >= self.confidence_threshold:
                class_name = self._get_class_name(label)
                if class_name in self.TARGET_OBJECTS:
                    valid_indices.append(i)
        
        if not valid_indices:
            logger.warning("No indoor objects detected")
            return {
                'objects_mask': np.zeros(original_shape, dtype=np.uint8),
                'detections': []
            }
        
        # Combine all object masks
        objects_mask = self._combine_masks(masks[valid_indices], original_shape)
        
        # Get detection info
        detections = []
        for i in valid_indices:
            class_name = self._get_class_name(labels[i])
            detections.append({
                'class': class_name,
                'confidence': float(scores[i]),
                'bbox': boxes[i].tolist(),
                'mask': masks[i][0]  # Remove channel dimension
            })
        
        return {
            'objects_mask': objects_mask,
            'detections': detections
        }
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID"""
        for name, idx in self.INDOOR_OBJECT_CLASSES.items():
            if idx == class_id:
                return name
        return f"unknown_{class_id}"
    
    def _combine_masks(self, masks: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """
        Combine multiple masks into a single objects mask
        
        Args:
            masks: Array of masks (N, 1, H, W)
            target_shape: Target output shape (H, W)
            
        Returns:
            Combined binary mask (H, W)
        """
        combined_mask = np.zeros(target_shape, dtype=np.uint8)
        
        for mask in masks:
            # Resize mask to target shape
            mask_resized = cv2.resize(
                mask[0],  # Remove channel dimension
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
            
            # Threshold and add to combined mask
            binary_mask = (mask_resized > self.mask_threshold).astype(np.uint8)
            combined_mask = np.maximum(combined_mask, binary_mask)
        
        # Ultra-precision post-processing
        # Small kernel for fine details
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        
        # Remove noise while preserving fine details
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Fill small holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Dilate slightly to avoid seam halos (but preserve details)
        combined_mask = cv2.dilate(combined_mask, kernel_small, iterations=1)
        
        return combined_mask * 255
    
    def get_objects_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Get combined mask of all detected objects
        
        Args:
            image: Input RGB image
            
        Returns:
            Binary objects mask (0 or 255)
        """
        result = self.detect_objects(image)
        return result['objects_mask']
    
    def get_detailed_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Get detailed information about detected objects
        
        Args:
            image: Input RGB image
            
        Returns:
            List of detection dictionaries
        """
        result = self.detect_objects(image)
        return result['detections']
    
    def filter_objects_by_size(self, objects_mask: np.ndarray, min_area: int = 100) -> np.ndarray:
        """
        Filter out small objects that might be noise
        
        Args:
            objects_mask: Binary objects mask
            min_area: Minimum area in pixels
            
        Returns:
            Filtered objects mask
        """
        # Find contours
        contours, _ = cv2.findContours(objects_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create new mask with only large objects
        filtered_mask = np.zeros_like(objects_mask)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                cv2.fillPoly(filtered_mask, [contour], 255)
        
        return filtered_mask
    
    def validate_objects_mask(self, objects_mask: np.ndarray, max_area_ratio: float = 0.8) -> bool:
        """
        Validate objects mask quality
        
        Args:
            objects_mask: Objects mask to validate
            max_area_ratio: Maximum ratio of object area to total image area
            
        Returns:
            True if mask is reasonable
        """
        total_pixels = objects_mask.shape[0] * objects_mask.shape[1]
        object_pixels = np.sum(objects_mask > 128)
        area_ratio = object_pixels / total_pixels
        
        if area_ratio > max_area_ratio:
            logger.warning(f"Objects mask too large: {area_ratio:.3f} > {max_area_ratio}")
            return False
        
        return True
