"""
Depth estimation and plane fitting for wall refinement
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.optimize import minimize
from sklearn.linear_model import RANSACRegressor

try:
    import transformers
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    transformers = None
    AutoImageProcessor = None
    AutoModel = None

logger = logging.getLogger(__name__)


class DepthPlaneRefiner:
    """
    Depth estimation and plane fitting for wall mask refinement
    """
    
    def __init__(
        self,
        model_name: str = "Intel/dpt-large",
        device: str = 'auto',
        use_depth: bool = True
    ):
        """
        Initialize depth plane refiner
        
        Args:
            model_name: Hugging Face model name for depth estimation
            device: Device to run on ('auto', 'cpu', 'cuda')
            use_depth: Whether to use depth estimation (can be disabled for speed)
        """
        self.device = self._get_device(device)
        self.use_depth = use_depth
        
        if use_depth and transformers is not None:
            try:
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                self.model.eval()
                logger.info(f"Loaded depth model {model_name} on {self.device}")
            except Exception as e:
                logger.warning(f"Failed to load depth model: {e}. Disabling depth estimation.")
                self.use_depth = False
        else:
            self.use_depth = False
            logger.info("Depth estimation disabled")
    
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
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map for the image
        
        Args:
            image: Input RGB image (H, W, 3)
            
        Returns:
            Depth map (H, W) normalized to [0, 1]
        """
        if not self.use_depth:
            # Fallback: create a simple depth map based on image gradients
            return self._create_simple_depth(image)
        
        try:
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
            
            # Convert to numpy and normalize
            depth = predicted_depth.squeeze().cpu().numpy().astype(np.float32)
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            
            # Resize to original image size
            original_shape = image.shape[:2]
            depth = cv2.resize(depth, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
            
            return depth
            
        except Exception as e:
            logger.warning(f"Depth estimation failed: {e}. Using simple depth.")
            return self._create_simple_depth(image)
    
    def _create_simple_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Create a simple depth map based on image gradients and structure
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize
        gradient_magnitude = (gradient_magnitude - gradient_magnitude.min()) / (gradient_magnitude.max() - gradient_magnitude.min())
        
        # Invert so that smooth areas (walls) have higher depth values
        depth = 1.0 - gradient_magnitude
        
        # Apply Gaussian blur to smooth the depth map
        depth = cv2.GaussianBlur(depth, (15, 15), 0)
        
        return depth
    
    def fit_plane_to_wall(self, depth_map: np.ndarray, wall_mask: np.ndarray) -> Dict:
        """
        Fit a plane to the wall region using RANSAC
        
        Args:
            depth_map: Depth map (H, W)
            wall_mask: Binary wall mask (H, W)
            
        Returns:
            Dictionary with plane parameters and inlier mask
        """
        # Get wall pixels
        wall_pixels = np.where(wall_mask > 128)
        if len(wall_pixels[0]) < 100:
            logger.warning("Too few wall pixels for plane fitting")
            return {
                'plane_params': None,
                'inlier_mask': wall_mask,
                'success': False
            }
        
        # Create 3D points (x, y, depth)
        points_3d = np.column_stack([
            wall_pixels[1].astype(np.float32),  # x coordinates
            wall_pixels[0].astype(np.float32),  # y coordinates
            depth_map[wall_pixels].astype(np.float32)  # depth values
        ])
        
        # Fit plane using RANSAC
        try:
            # Normalize coordinates for numerical stability
            points_centered = points_3d - points_3d.mean(axis=0)
            
            # Use RANSAC to fit plane
            ransac = RANSACRegressor(
                random_state=42,
                max_trials=1000,
                residual_threshold=0.1
            )
            
            # Fit plane: z = ax + by + c
            X = points_centered[:, :2]  # x, y
            y = points_centered[:, 2]   # z
            
            ransac.fit(X, y)
            
            # Get plane parameters
            a, b = ransac.estimator_.coef_
            c = ransac.estimator_.intercept_
            
            # Convert back to original coordinates
            center = points_3d.mean(axis=0)
            c = c - a * center[0] - b * center[1] + center[2]
            
            plane_params = {'a': a, 'b': b, 'c': c, 'center': center}
            
            # Create inlier mask
            inlier_mask = np.zeros_like(wall_mask)
            inlier_indices = ransac.inlier_mask_
            inlier_pixels = (wall_pixels[0][inlier_indices], wall_pixels[1][inlier_indices])
            inlier_mask[inlier_pixels] = 255
            
            return {
                'plane_params': plane_params,
                'inlier_mask': inlier_mask,
                'success': True,
                'n_inliers': np.sum(inlier_indices),
                'n_total': len(wall_pixels[0])
            }
            
        except Exception as e:
            logger.warning(f"Plane fitting failed: {e}")
            return {
                'plane_params': None,
                'inlier_mask': wall_mask,
                'success': False
            }
    
    def refine_wall_mask(self, wall_mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
        """
        Refine wall mask using depth and plane fitting
        
        Args:
            wall_mask: Initial wall mask
            depth_map: Depth map
            
        Returns:
            Refined wall mask
        """
        if not self.use_depth:
            # Simple morphological refinement
            return self._morphological_refine(wall_mask)
        
        # Fit plane to wall
        plane_result = self.fit_plane_to_wall(depth_map, wall_mask)
        
        if not plane_result['success']:
            logger.warning("Plane fitting failed, using morphological refinement")
            return self._morphological_refine(wall_mask)
        
        # Use inlier mask as refined wall mask
        refined_mask = plane_result['inlier_mask']
        
        # Additional morphological operations
        refined_mask = self._morphological_refine(refined_mask)
        
        logger.info(f"Wall mask refined: {plane_result['n_inliers']}/{plane_result['n_total']} pixels kept")
        
        return refined_mask
    
    def _morphological_refine(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to refine mask
        """
        # Convert to binary
        binary_mask = (mask > 128).astype(np.uint8)
        
        # Remove small holes and noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
        
        # Fill holes
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        
        return binary_mask * 255
    
    def estimate_wall_normal(self, plane_params: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Estimate wall normal vector from plane parameters
        
        Args:
            plane_params: Plane parameters from fit_plane_to_wall
            image_shape: Image shape (H, W)
            
        Returns:
            Normal vector (3,)
        """
        if plane_params is None:
            # Default normal (pointing towards camera)
            return np.array([0, 0, 1])
        
        # Plane equation: z = ax + by + c
        # Normal vector: (-a, -b, 1)
        a, b, c = plane_params['a'], plane_params['b'], plane_params['c']
        normal = np.array([-a, -b, 1])
        
        # Normalize
        normal = normal / np.linalg.norm(normal)
        
        return normal
    
    def get_wall_depth_consistency(self, depth_map: np.ndarray, wall_mask: np.ndarray) -> float:
        """
        Measure depth consistency within wall region
        
        Args:
            depth_map: Depth map
            wall_mask: Wall mask
            
        Returns:
            Depth consistency score (0-1, higher is better)
        """
        wall_pixels = depth_map[wall_mask > 128]
        
        if len(wall_pixels) < 10:
            return 0.0
        
        # Compute depth variance
        depth_std = np.std(wall_pixels)
        depth_mean = np.mean(wall_pixels)
        
        # Consistency score (lower std = higher consistency)
        consistency = 1.0 / (1.0 + depth_std)
        
        return min(consistency, 1.0)
