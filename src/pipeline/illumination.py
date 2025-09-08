"""
Illumination transfer and lightness matching for realistic wallpaper compositing
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging
from scipy import ndimage
from skimage import exposure, filters

logger = logging.getLogger(__name__)


class IlluminationTransfer:
    """
    Transfer illumination from original wall to wallpaper for realistic compositing
    """
    
    def __init__(
        self,
        bilateral_d: int = 9,
        bilateral_sigma_color: float = 75.0,
        bilateral_sigma_space: float = 75.0,
        guided_filter_radius: int = 8,
        guided_filter_eps: float = 0.01
    ):
        """
        Initialize illumination transfer
        
        Args:
            bilateral_d: Bilateral filter diameter
            bilateral_sigma_color: Bilateral filter color sigma
            bilateral_sigma_space: Bilateral filter space sigma
            guided_filter_radius: Guided filter radius
            guided_filter_eps: Guided filter epsilon
        """
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.guided_filter_radius = guided_filter_radius
        self.guided_filter_eps = guided_filter_eps
    
    def transfer_illumination(
        self,
        wallpaper: np.ndarray,
        room_image: np.ndarray,
        wall_mask: np.ndarray
    ) -> np.ndarray:
        """
        Transfer illumination from room wall to wallpaper
        
        Args:
            wallpaper: Wallpaper image (H, W, 3)
            room_image: Original room image (H, W, 3)
            wall_mask: Wall mask (H, W)
            
        Returns:
            Illumination-adjusted wallpaper (H, W, 3)
        """
        # Convert to Lab color space
        wallpaper_lab = cv2.cvtColor(wallpaper, cv2.COLOR_RGB2LAB).astype(np.float32)
        room_lab = cv2.cvtColor(room_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Extract L channels
        wallpaper_L = wallpaper_lab[:, :, 0]
        room_L = room_lab[:, :, 0]
        
        # Get wall region lightness
        wall_lightness = self._extract_wall_lightness(room_L, wall_mask)
        
        # Match wallpaper lightness to wall lightness
        matched_L = self._match_lightness(wallpaper_L, wall_lightness, wall_mask)
        
        # Extract shading map from original wall
        shading_map = self._extract_shading_map(room_L, wall_mask)
        
        # Apply shading to matched wallpaper
        shaded_L = self._apply_shading(matched_L, shading_map)
        
        # Combine with original a, b channels
        result_lab = wallpaper_lab.copy()
        result_lab[:, :, 0] = shaded_L
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result_rgb
    
    def _extract_wall_lightness(self, room_L: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        """
        Extract lightness distribution from wall region with fallback
        """
        # Get wall pixels
        wall_pixels = room_L[wall_mask > 128]
        
        if len(wall_pixels) == 0:
            logger.warning("No wall pixels found for lightness extraction, using fallback")
            return self._estimate_scene_tone(room_L, wall_mask)
        
        # Compute statistics
        mean_lightness = np.mean(wall_pixels)
        std_lightness = np.std(wall_pixels)
        
        logger.info(f"Wall lightness: mean={mean_lightness:.1f}, std={std_lightness:.1f}")
        
        return wall_pixels
    
    def _estimate_scene_tone(self, room_L: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        """
        Fallback tone estimation when no wall pixels available
        """
        import cv2
        
        # Try to find edge pixels as wall approximation
        edges = cv2.Canny(room_L.astype(np.uint8), 50, 150)
        edge_pixels = room_L[edges > 0]
        
        if len(edge_pixels) > 500:
            logger.info("Using edge pixels for tone estimation")
            return edge_pixels
        
        # Final fallback: use center region
        h, w = room_L.shape
        center_h, center_w = h // 2, w // 2
        center_region = room_L[center_h-h//8:center_h+h//8, center_w-w//8:center_w+w//8]
        logger.info("Using center region for tone estimation")
        return center_region.flatten()
    
    def _match_lightness(
        self,
        wallpaper_L: np.ndarray,
        wall_lightness: np.ndarray,
        wall_mask: np.ndarray
    ) -> np.ndarray:
        """
        Match wallpaper lightness to wall lightness using histogram matching
        """
        if len(wall_lightness) == 0:
            return wallpaper_L
        
        # Compute target histogram from wall lightness
        target_hist, target_bins = np.histogram(wall_lightness, bins=256, range=(0, 256))
        target_cdf = np.cumsum(target_hist).astype(np.float32)
        target_cdf = target_cdf / target_cdf[-1]  # Normalize
        
        # Compute source histogram from wallpaper
        source_hist, source_bins = np.histogram(wallpaper_L.flatten(), bins=256, range=(0, 256))
        source_cdf = np.cumsum(source_hist).astype(np.float32)
        source_cdf = source_cdf / source_cdf[-1]  # Normalize
        
        # Create mapping function
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            # Find closest target CDF value
            diff = np.abs(target_cdf - source_cdf[i])
            mapping[i] = np.argmin(diff)
        
        # Apply mapping
        matched_L = mapping[wallpaper_L.astype(np.uint8)]
        
        # Preserve some of the original wallpaper characteristics
        # Blend with original to avoid over-correction
        alpha = 0.7  # Weight for matched lightness
        matched_L = alpha * matched_L + (1 - alpha) * wallpaper_L
        
        # Clamp gain to prevent over-correction
        gain = matched_L / (wallpaper_L + 1e-6)
        gain = np.clip(gain, 0.85, 1.15)
        matched_L = wallpaper_L * gain
        
        return matched_L.astype(np.float32)
    
    def _extract_shading_map(self, room_L: np.ndarray, wall_mask: np.ndarray) -> np.ndarray:
        """
        Extract shading map from original wall using bilateral filtering
        """
        # Create wall-only lightness image
        wall_lightness = np.zeros_like(room_L)
        wall_lightness[wall_mask > 128] = room_L[wall_mask > 128]
        
        # Fill non-wall regions with mean wall lightness
        mean_wall_lightness = np.mean(room_L[wall_mask > 128]) if np.any(wall_mask > 128) else 128
        wall_lightness[wall_mask == 0] = mean_wall_lightness
        
        # Apply bilateral filter to extract large-scale shading
        shading_map = cv2.bilateralFilter(
            wall_lightness.astype(np.uint8),
            self.bilateral_d,
            self.bilateral_sigma_color,
            self.bilateral_sigma_space
        ).astype(np.float32)
        
        # Normalize shading map
        shading_map = shading_map / np.mean(shading_map)
        
        return shading_map
    
    def _apply_shading(self, matched_L: np.ndarray, shading_map: np.ndarray) -> np.ndarray:
        """
        Apply shading map to matched lightness
        """
        # Ensure same size
        if matched_L.shape != shading_map.shape:
            shading_map = cv2.resize(shading_map, (matched_L.shape[1], matched_L.shape[0]))
        
        # Apply shading
        shaded_L = matched_L * shading_map
        
        # Clamp to valid range
        shaded_L = np.clip(shaded_L, 0, 255)
        
        return shaded_L
    
    def enhance_lighting_consistency(
        self,
        wallpaper: np.ndarray,
        room_image: np.ndarray,
        wall_mask: np.ndarray
    ) -> np.ndarray:
        """
        Enhanced lighting consistency using Retinex-like decomposition
        """
        # Convert to Lab
        wallpaper_lab = cv2.cvtColor(wallpaper, cv2.COLOR_RGB2LAB).astype(np.float32)
        room_lab = cv2.cvtColor(room_image, cv2.COLOR_RGB2LAB).astype(np.float32)
        
        # Extract L channels
        wallpaper_L = wallpaper_lab[:, :, 0]
        room_L = room_lab[:, :, 0]
        
        # Decompose into reflectance and illumination
        wallpaper_reflectance, wallpaper_illumination = self._retinex_decompose(wallpaper_L)
        room_reflectance, room_illumination = self._retinex_decompose(room_L)
        
        # Extract wall illumination
        wall_illumination = room_illumination[wall_mask > 128]
        if len(wall_illumination) == 0:
            return wallpaper
        
        # Match wallpaper illumination to wall illumination
        matched_illumination = self._match_illumination_distribution(
            wallpaper_illumination, wall_illumination
        )
        
        # Reconstruct with matched illumination
        result_L = wallpaper_reflectance * matched_illumination
        
        # Combine with original a, b channels
        result_lab = wallpaper_lab.copy()
        result_lab[:, :, 0] = np.clip(result_L, 0, 255)
        
        # Convert back to RGB
        result_rgb = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        
        return result_rgb
    
    def _retinex_decompose(self, lightness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose lightness into reflectance and illumination using Retinex
        """
        # Estimate illumination using Gaussian blur
        illumination = cv2.GaussianBlur(lightness, (0, 0), sigmaX=50, sigmaY=50)
        
        # Avoid division by zero
        illumination = np.maximum(illumination, 1.0)
        
        # Compute reflectance
        reflectance = lightness / illumination
        
        # Normalize reflectance
        reflectance = np.clip(reflectance, 0, 1)
        
        return reflectance, illumination
    
    def _match_illumination_distribution(
        self,
        source_illumination: np.ndarray,
        target_illumination: np.ndarray
    ) -> np.ndarray:
        """
        Match illumination distribution using histogram matching
        """
        # Flatten arrays
        source_flat = source_illumination.flatten()
        target_flat = target_illumination.flatten()
        
        # Compute histograms
        source_hist, source_bins = np.histogram(source_flat, bins=256, range=(0, 256))
        target_hist, target_bins = np.histogram(target_flat, bins=256, range=(0, 256))
        
        # Compute CDFs
        source_cdf = np.cumsum(source_hist).astype(np.float32)
        source_cdf = source_cdf / source_cdf[-1]
        
        target_cdf = np.cumsum(target_hist).astype(np.float32)
        target_cdf = target_cdf / target_cdf[-1]
        
        # Create mapping
        mapping = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            diff = np.abs(target_cdf - source_cdf[i])
            mapping[i] = np.argmin(diff)
        
        # Apply mapping
        matched_illumination = mapping[source_illumination.astype(np.uint8)]
        
        return matched_illumination.astype(np.float32)
    
    def preserve_texture_details(
        self,
        original_wallpaper: np.ndarray,
        illumination_adjusted: np.ndarray,
        wall_mask: np.ndarray,
        strength: float = 0.3
    ) -> np.ndarray:
        """
        Preserve texture details from original wallpaper
        """
        # Convert to grayscale for texture analysis
        orig_gray = cv2.cvtColor(original_wallpaper, cv2.COLOR_RGB2GRAY)
        adj_gray = cv2.cvtColor(illumination_adjusted, cv2.COLOR_RGB2GRAY)
        
        # Compute texture difference
        texture_diff = orig_gray.astype(np.float32) - adj_gray.astype(np.float32)
        
        # Apply texture preservation
        result = illumination_adjusted.astype(np.float32)
        for c in range(3):
            result[:, :, c] += strength * texture_diff
        
        # Clamp to valid range
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
