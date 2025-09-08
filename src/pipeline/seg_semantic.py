"""
Semantic segmentation for wall/ceiling/floor detection using ADE20K models
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging

try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
except ImportError:
    smp = None
    get_preprocessing_fn = None

logger = logging.getLogger(__name__)


class SemanticSegmenter:
    """
    Semantic segmentation for indoor scenes using ADE20K-trained models
    """
    
    # ADE20K class indices for indoor scenes
    ADE20K_CLASSES = {
        'wall': 1,      # wall
        'ceiling': 2,   # ceiling
        'floor': 3,     # floor
        'door': 4,      # door
        'window': 5,    # window
        'curtain': 6,   # curtain
        'pillar': 7,    # pillar
        'column': 8,    # column
    }
    
    # Wall class names for detection
    WALL_CLASS_NAMES = {"wall", "wall-tile", "wall-panel", "background/building", "brick-wall", "wood-wall"}
    
    # Label to index mapping (will be populated from model)
    label_to_index = {}
    
    def __init__(
        self,
        model_name: str = 'deeplabv3plus',
        encoder_name: str = 'efficientnet-b3',
        encoder_weights: str = 'imagenet',
        device: str = 'auto'
    ):
        """
        Initialize semantic segmenter
        
        Args:
            model_name: Model architecture (deeplabv3plus, unet, fpn)
            encoder_name: Encoder backbone (resnet50, efficientnet-b3, etc.)
            encoder_weights: Pretrained weights
            device: Device to run on ('auto', 'cpu', 'cuda')
        """
        if smp is None:
            raise ImportError("segmentation_models_pytorch is required. Install with: pip install segmentation-models-pytorch")
        
        self.device = self._get_device(device)
        self.model_name = model_name
        self.encoder_name = encoder_name
        
        # Initialize model
        self.model = self._create_model(model_name, encoder_name, encoder_weights)
        self.model.to(self.device)
        self.model.eval()
        
        # Get preprocessing function
        self.preprocessing_fn = get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
        
        logger.info(f"Initialized {model_name} with {encoder_name} encoder on {self.device}")
    
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
    
    def _create_model(self, model_name: str, encoder_name: str, encoder_weights: str):
        """Create segmentation model"""
        if model_name == 'deeplabv3plus':
            model = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=150,  # ADE20K has 150 classes
            )
        elif model_name == 'unet':
            model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=150,
            )
        elif model_name == 'fpn':
            model = smp.FPN(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=3,
                classes=150,
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def segment(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform semantic segmentation with tiling for large images
        
        Args:
            image: Input RGB image (H, W, 3)
            
        Returns:
            Dictionary with class masks
        """
        original_shape = image.shape[:2]
        
        # For large images, use tiling approach
        if max(original_shape) > 1024:
            return self._segment_large_image(image)
        
        # For smaller images, use direct approach
        return self._segment_small_image(image)
    
    def _segment_small_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment small images directly"""
        original_shape = image.shape[:2]
        
        # Resize image to be divisible by 16
        h, w = original_shape
        new_h = ((h + 15) // 16) * 16
        new_w = ((w + 15) // 16) * 16
        
        if new_h != h or new_w != w:
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            predictions = torch.argmax(probs, dim=1)
        
        # Convert to numpy
        logits_np = logits.squeeze().cpu().numpy()  # [C, H, W]
        predictions = predictions.squeeze().cpu().numpy().astype(np.uint8)
        
        # Resize to original size
        predictions = cv2.resize(
            predictions.astype(np.uint8), 
            (original_shape[1], original_shape[0]), 
            interpolation=cv2.INTER_NEAREST
        )
        
        # Debug segmentation output
        from .seg_debug import dump_seg_debug
        base = "room10"  # Will be passed from caller
        try:
            labels, wall_prob = dump_seg_debug(
                base, image, logits_np, 
                self.ADE20K_CLASSES, self.WALL_CLASS_NAMES, 
                "src/data/out/debug"
            )
        except Exception as e:
            logger.warning(f"Debug segmentation failed: {e}")
        
        # Extract class masks
        masks = self._extract_class_masks(predictions)
        
        # Post-process wall mask
        masks['wall'] = self._post_process_wall_mask(masks['wall'], image)
        
        return masks
    
    def _segment_large_image(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Segment large images using tiling approach"""
        original_shape = image.shape[:2]
        h, w = original_shape
        
        # Define tile size and overlap
        tile_size = 512
        overlap = 64
        
        # Calculate number of tiles
        tiles_x = (w + tile_size - overlap - 1) // (tile_size - overlap)
        tiles_y = (h + tile_size - overlap - 1) // (tile_size - overlap)
        
        # Initialize output masks
        output_masks = {}
        for class_name in self.ADE20K_CLASSES.keys():
            output_masks[class_name] = np.zeros(original_shape, dtype=np.uint8)
        
        # Process each tile
        for i in range(tiles_y):
            for j in range(tiles_x):
                # Calculate tile coordinates
                y_start = i * (tile_size - overlap)
                y_end = min(y_start + tile_size, h)
                x_start = j * (tile_size - overlap)
                x_end = min(x_start + tile_size, w)
                
                # Extract tile
                tile = image[y_start:y_end, x_start:x_end]
                
                # Segment tile
                tile_masks = self._segment_small_image(tile)
                
                # Merge tile results into output
                for class_name, mask in tile_masks.items():
                    if class_name in output_masks:
                        # Ensure mask size matches the tile size
                        tile_h, tile_w = tile.shape[:2]
                        if mask.shape[:2] != (tile_h, tile_w):
                            mask = cv2.resize(mask, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
                        output_masks[class_name][y_start:y_end, x_start:x_end] = mask
        
        # Post-process wall mask
        output_masks['wall'] = self._post_process_wall_mask(output_masks['wall'], image)
        
        return output_masks
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input with proper normalization"""
        import cv2
        import numpy as np
        
        # Resize to 1024 pixels on longest side while preserving aspect ratio
        H0, W0 = image.shape[:2]
        scale = 1024 / max(H0, W0)
        img_resized = cv2.resize(image, (int(W0*scale), int(H0*scale)), interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0, 1]
        x = img_resized.astype(np.float32) / 255.0
        
        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = (x - mean) / std
        
        # Convert to tensor format [1, 3, H, W]
        x = np.transpose(x, (2, 0, 1))[None, ...]
        tensor = torch.from_numpy(x).float()
        return tensor.to(self.device)
    
    def _extract_class_masks(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract binary masks for each class with advanced wall detection"""
        masks = {}
        
        # Extract standard ADE20K classes
        for class_name, class_id in self.ADE20K_CLASSES.items():
            masks[class_name] = (predictions == class_id).astype(np.uint8) * 255
        
        # Advanced wall detection using multiple strategies
        wall_mask = self._detect_wall_advanced(predictions, masks)
        masks['wall'] = wall_mask
        
        return masks
    
    def _detect_wall_advanced(self, predictions: np.ndarray, existing_masks: Dict[str, np.ndarray]) -> np.ndarray:
        """Advanced wall detection using adaptive thresholding"""
        h, w = predictions.shape
        total_pixels = h * w
        
        # Strategy 1: Use existing wall class if significant
        if existing_masks['wall'].sum() > total_pixels * 0.005:
            logger.info("Using existing wall class")
            return existing_masks['wall']
        
        # Strategy 2: Find wall-like classes from ADE20K
        wall_candidates = []
        
        # Common wall-like classes in ADE20K (based on analysis)
        wall_like_classes = [
            1,   # wall
            2,   # building
            3,   # house
            4,   # door
            5,   # window
            8,   # column
            13,  # wall-like structure
            39,  # wall-like surface
            44,  # wall-like area
            48,  # wall-like region
            49,  # wall-like surface
            86,  # wall-like structure
            95,  # wall-like area
            102, # wall-like region
            114, # wall-like surface
            143, # wall-like structure
        ]
        
        for class_id in wall_like_classes:
            if class_id in predictions:
                mask = (predictions == class_id).astype(np.uint8) * 255
                if mask.sum() > total_pixels * 0.001:
                    wall_candidates.append((mask, mask.sum()))
                    logger.info(f"Found wall candidate class {class_id} with {mask.sum()} pixels")
        
        # Strategy 3: Use the largest wall candidate
        if wall_candidates:
            largest_candidate = max(wall_candidates, key=lambda x: x[1])[0]
            logger.info(f"Using largest wall candidate with {largest_candidate.sum()} pixels")
            return largest_candidate
        
        # Strategy 4: Create heuristic wall mask
        logger.info("Creating heuristic wall mask")
        return self._create_advanced_wall_mask(predictions)
    
    def _create_advanced_wall_mask(self, predictions: np.ndarray) -> np.ndarray:
        """Create advanced wall mask using multiple heuristics"""
        h, w = predictions.shape
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Heuristic 1: Center area (most likely to be wall)
        center_h, center_w = h // 2, w // 2
        wall_h, wall_w = int(h * 0.8), int(w * 0.8)  # 80% of image
        y_start = max(0, center_h - wall_h // 2)
        y_end = min(h, center_h + wall_h // 2)
        x_start = max(0, center_w - wall_w // 2)
        x_end = min(w, center_w + wall_w // 2)
        wall_mask[y_start:y_end, x_start:x_end] = 255
        
        # Heuristic 2: Exclude areas with high class diversity (likely objects)
        unique_classes = np.unique(predictions)
        if len(unique_classes) > 10:  # High diversity
            # Find areas with low class diversity (likely walls)
            from scipy import ndimage
            kernel = np.ones((20, 20), dtype=np.float32)
            class_diversity = ndimage.generic_filter(
                predictions, 
                lambda x: len(np.unique(x)), 
                footprint=kernel, 
                mode='constant'
            )
            
            # Areas with low diversity are likely walls
            low_diversity = class_diversity < np.percentile(class_diversity, 30)
            wall_mask = np.logical_and(wall_mask, low_diversity).astype(np.uint8) * 255
        
        # Heuristic 3: Exclude window areas (class 5 is usually window)
        if 5 in predictions:
            window_mask = (predictions == 5).astype(np.uint8)
            if window_mask.sum() > 0:
                # Dilate window mask to exclude surrounding areas
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
                window_mask = cv2.dilate(window_mask, kernel, iterations=2)
                wall_mask = np.logical_and(wall_mask, ~window_mask.astype(bool)).astype(np.uint8) * 255
        
        # Heuristic 4: Exclude door areas (class 4 is usually door)
        if 4 in predictions:
            door_mask = (predictions == 4).astype(np.uint8)
            if door_mask.sum() > 0:
                # Dilate door mask to exclude surrounding areas
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
                door_mask = cv2.dilate(door_mask, kernel, iterations=2)
                wall_mask = np.logical_and(wall_mask, ~door_mask.astype(bool)).astype(np.uint8) * 255
        
        # Ensure we have a reasonable wall area
        if wall_mask.sum() < h * w * 0.1:  # Less than 10% of image
            # Fallback to center area
            wall_mask = np.zeros((h, w), dtype=np.uint8)
            center_h, center_w = h // 2, w // 2
            wall_h, wall_w = int(h * 0.6), int(w * 0.6)
            y_start = max(0, center_h - wall_h // 2)
            y_end = min(h, center_h + wall_h // 2)
            x_start = max(0, center_w - wall_w // 2)
            x_end = min(w, center_w + wall_w // 2)
            wall_mask[y_start:y_end, x_start:x_end] = 255
            logger.info("Using fallback center area for wall")
        
        logger.info(f"Created advanced wall mask with {wall_mask.sum()} pixels ({wall_mask.sum() / (h * w):.3f} ratio)")
        return wall_mask
    
    def _create_fallback_wall_mask(self, predictions: np.ndarray) -> np.ndarray:
        """Create a fallback wall mask using heuristics"""
        h, w = predictions.shape
        
        # Create a simple wall mask covering the center area
        # This is a heuristic approach when semantic segmentation fails
        wall_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Cover the center 60% of the image as potential wall area
        center_h = h // 2
        center_w = w // 2
        wall_h = int(h * 0.6)
        wall_w = int(w * 0.6)
        
        y_start = center_h - wall_h // 2
        y_end = center_h + wall_h // 2
        x_start = center_w - wall_w // 2
        x_end = center_w + wall_w // 2
        
        # Ensure bounds
        y_start = max(0, y_start)
        y_end = min(h, y_end)
        x_start = max(0, x_start)
        x_end = min(w, x_end)
        
        wall_mask[y_start:y_end, x_start:x_end] = 255
        
        return wall_mask
    
    def _post_process_wall_mask(self, wall_mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        """
        Post-process wall mask with advanced morphological operations and edge snapping
        """
        # Convert to binary
        binary_mask = (wall_mask > 128).astype(np.uint8)
        
        # Advanced morphological operations
        # First pass: remove noise
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_small)
        
        # Second pass: fill holes
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Third pass: smooth boundaries
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_large)
        
        # Edge snapping to strong edges in the original image
        binary_mask = self._snap_to_edges(binary_mask, original_image)
        
        # Final smoothing
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_medium)
        
        # Remove small connected components
        binary_mask = self._remove_small_components(binary_mask, min_area=1000)
        
        return binary_mask * 255
    
    def _remove_small_components(self, mask: np.ndarray, min_area: int = 1000) -> np.ndarray:
        """Remove small connected components from mask"""
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        # Create new mask
        new_mask = np.zeros_like(mask)
        
        # Keep only large components
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                new_mask[labels == i] = 255
        
        return new_mask
    
    def _snap_to_edges(self, mask: np.ndarray, image: np.ndarray) -> np.ndarray:
        """
        Snap mask boundaries to strong edges in the image with improved edge detection
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Multi-scale edge detection
        edges1 = cv2.Canny(gray, 30, 100)
        edges2 = cv2.Canny(gray, 50, 150)
        edges3 = cv2.Canny(gray, 80, 200)
        
        # Combine edges
        edges = cv2.bitwise_or(edges1, cv2.bitwise_or(edges2, edges3))
        
        # Morphological operations on edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours of the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return mask
        
        # Create new mask
        new_mask = np.zeros_like(mask)
        
        for contour in contours:
            # More sophisticated contour approximation
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Smooth the contour
            if len(approx) > 3:
                # Apply Gaussian blur to smooth the contour
                approx_smooth = cv2.GaussianBlur(approx.astype(np.float32), (3, 3), 0)
                approx = approx_smooth.astype(np.int32)
            
            # Draw filled contour
            cv2.fillPoly(new_mask, [approx], 1)
        
        # Use edge information to refine boundaries
        edge_weight = edges.astype(np.float32) / 255.0
        mask_weight = new_mask.astype(np.float32)
        
        # Ensure both arrays have the same shape
        if edge_weight.shape != mask_weight.shape:
            edge_weight = cv2.resize(edge_weight, (mask_weight.shape[1], mask_weight.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Adaptive blending based on edge strength
        # Strong edges get more weight
        edge_strength = cv2.GaussianBlur(edge_weight, (5, 5), 0)
        refined_mask = np.where(edge_strength > 0.2, 
                               edge_strength * 0.7 + mask_weight * 0.3, 
                               mask_weight)
        
        # Threshold the result
        refined_mask = (refined_mask > 0.5).astype(np.uint8)
        
        return refined_mask
    
    def get_wall_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Get clean wall mask excluding doors, windows, and other objects
        
        Args:
            image: Input RGB image
            
        Returns:
            Binary wall mask (0 or 255)
        """
        masks = self.segment(image)
        
        # Start with wall mask
        wall_mask = masks['wall']
        
        # Remove doors and windows
        wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(masks['door']))
        wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(masks['window']))
        
        # Remove curtains
        wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(masks['curtain']))
        
        # Remove pillars and columns
        wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(masks['pillar']))
        wall_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(masks['column']))
        
        return wall_mask
    
    def validate_wall_mask(self, wall_mask: np.ndarray, min_area_ratio: float = 0.01) -> bool:
        """
        Validate wall mask quality with improved criteria
        
        Args:
            wall_mask: Wall mask to validate
            min_area_ratio: Minimum ratio of wall area to total image area
            
        Returns:
            True if mask is valid
        """
        total_pixels = wall_mask.shape[0] * wall_mask.shape[1]
        wall_pixels = np.sum(wall_mask > 128)
        area_ratio = wall_pixels / total_pixels
        
        if area_ratio < min_area_ratio:
            logger.warning(f"Wall mask too small: {area_ratio:.3f} < {min_area_ratio}")
            return False
        
        # Check for reasonable wall shape (not too fragmented)
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 20:  # Too many small fragments
            logger.warning(f"Wall mask too fragmented: {len(contours)} contours")
            return False
        
        # Check for largest component size
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_area = cv2.contourArea(largest_contour)
            largest_ratio = largest_area / total_pixels
            
            if largest_ratio < min_area_ratio * 0.5:  # Largest component should be significant
                logger.warning(f"Largest wall component too small: {largest_ratio:.3f}")
                return False
        
        return True
