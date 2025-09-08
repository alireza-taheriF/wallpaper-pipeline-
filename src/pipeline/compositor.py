"""
Main wallpaper compositor that combines all components
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .io_utils import load_image, save_image
from .seg_semantic import SemanticSegmenter
from .seg_instances import InstanceSegmenter
from .depth_plane import DepthPlaneRefiner
from .wall_polygon import WallPolygonizer
from .illumination import IlluminationTransfer
from .windows_optimizer import get_windows_optimizer

logger = logging.getLogger(__name__)


class WallpaperCompositor:
    """
    Main wallpaper compositor that orchestrates the entire pipeline
    """
    
    def __init__(
        self,
        use_depth: bool = True,
        device: str = 'auto',
        confidence_threshold: float = 0.5,
        feather_radius: int = 5,
        windows_optimized: bool = True
    ):
        """
        Initialize wallpaper compositor
        
        Args:
            use_depth: Whether to use depth estimation
            device: Device to run on
            confidence_threshold: Confidence threshold for object detection
            feather_radius: Radius for edge feathering
            windows_optimized: Enable Windows 32GB RAM optimizations
        """
        self.use_depth = use_depth
        self.feather_radius = feather_radius
        self.windows_optimized = windows_optimized
        
        # Initialize Windows optimizer if enabled
        if windows_optimized:
            self.windows_optimizer = get_windows_optimizer()
            logger.info("Windows 32GB RAM optimizations enabled")
        else:
            self.windows_optimizer = None
        
        # Initialize components
        logger.info("Initializing wallpaper compositor components...")
        
        # Use optimized model for Windows
        if windows_optimized:
            semantic_model = "efficientnet-b7"  # Larger model for better quality
        else:
            semantic_model = "efficientnet-b3"
        
        self.semantic_segmenter = SemanticSegmenter(
            device=device,
            encoder_name=semantic_model
        )
        self.instance_segmenter = InstanceSegmenter(
            device=device,
            confidence_threshold=confidence_threshold,
            mask_threshold=0.3
        )
        
        if use_depth:
            self.depth_refiner = DepthPlaneRefiner(device=device, use_depth=True)
        else:
            self.depth_refiner = None
        
        self.polygonizer = WallPolygonizer()
        self.illumination_transfer = IlluminationTransfer()
        
        logger.info("Wallpaper compositor initialized successfully")
    
    def composite_wallpaper(
        self,
        room_image: np.ndarray,
        wallpaper_image: np.ndarray,
        output_paths: Optional[Dict[str, Path]] = None
    ) -> Dict:
        """
        Composite wallpaper onto room image
        
        Args:
            room_image: Original room image (H, W, 3)
            wallpaper_image: Wallpaper image (H, W, 3)
            output_paths: Optional output paths for saving intermediate results
            
        Returns:
            Dictionary with compositing results and metadata
        """
        logger.info("Starting wallpaper compositing pipeline...")
        
        results = {
            'success': False,
            'wall_mask': None,
            'objects_mask': None,
            'polygon': None,
            'homography': None,
            'composite_image': None,
            'metadata': {}
        }
        
        try:
            # Step 1: Semantic segmentation for wall detection
            logger.info("Step 1: Performing semantic segmentation...")
            wall_mask = self.semantic_segmenter.get_wall_mask(room_image)
            
            # Validate wall mask
            if not self.semantic_segmenter.validate_wall_mask(wall_mask):
                logger.warning("Wall mask validation failed, but continuing...")
                # Don't return, just log warning
            
            results['wall_mask'] = wall_mask
            results['metadata']['wall_area_ratio'] = np.sum(wall_mask > 128) / (wall_mask.shape[0] * wall_mask.shape[1])
            
            # Step 2: Instance segmentation for object detection
            logger.info("Step 2: Performing instance segmentation...")
            objects_mask = self.instance_segmenter.get_objects_mask(room_image)
            objects_mask = self.instance_segmenter.filter_objects_by_size(objects_mask)
            
            # Validate objects mask
            if not self.instance_segmenter.validate_objects_mask(objects_mask):
                logger.warning("Objects mask validation failed, but continuing...")
            
            results['objects_mask'] = objects_mask
            results['metadata']['objects_area_ratio'] = np.sum(objects_mask > 128) / (objects_mask.shape[0] * objects_mask.shape[1])
            
            # Step 3: Depth estimation and plane fitting (optional)
            if self.use_depth and self.depth_refiner is not None:
                logger.info("Step 3: Performing depth estimation and plane fitting...")
                depth_map = self.depth_refiner.estimate_depth(room_image)
                wall_mask = self.depth_refiner.refine_wall_mask(wall_mask, depth_map)
                results['wall_mask'] = wall_mask
                results['metadata']['depth_consistency'] = self.depth_refiner.get_wall_depth_consistency(depth_map, wall_mask)
            else:
                logger.info("Step 3: Skipping depth estimation")
                results['metadata']['depth_consistency'] = 0.0
            
            # Step 4: Wall polygonization and homography estimation
            logger.info("Step 4: Performing wall polygonization...")
            polygon_result = self.polygonizer.extract_wall_polygon(wall_mask)
            
            if not polygon_result['success']:
                logger.error("Wall polygonization failed")
                return results
            
            results['polygon'] = polygon_result['polygon']
            results['homography'] = polygon_result['homography']
            results['metadata']['polygon_vertices'] = polygon_result['n_vertices']
            results['metadata']['polygon_area_ratio'] = polygon_result['area_ratio']
            results['metadata']['is_fallback'] = polygon_result.get('fallback', False)
            
            # Step 5: Illumination transfer
            logger.info("Step 5: Performing illumination transfer...")
            adjusted_wallpaper = self.illumination_transfer.transfer_illumination(
                wallpaper_image, room_image, wall_mask
            )
            
            # Step 6: Final compositing
            logger.info("Step 6: Performing final compositing...")
            composite_image = self._composite_images(
                room_image, adjusted_wallpaper, wall_mask, objects_mask, polygon_result['homography']
            )
            
            results['composite_image'] = composite_image
            results['success'] = True
            
            # Save outputs if paths provided
            if output_paths:
                self._save_outputs(results, output_paths)
            
            logger.info("Wallpaper compositing completed successfully")
            
        except Exception as e:
            logger.error(f"Wallpaper compositing failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _composite_images(
        self,
        room_image: np.ndarray,
        wallpaper_image: np.ndarray,
        wall_mask: np.ndarray,
        objects_mask: np.ndarray,
        homography: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Composite wallpaper onto room image
        """
        # Start with original room image
        composite = room_image.copy()
        
        # Create final blend mask (wall minus objects)
        blend_mask = cv2.bitwise_and(wall_mask, cv2.bitwise_not(objects_mask))
        
        # Apply feathering to blend mask
        blend_mask = self._feather_mask(blend_mask)
        
        if homography is not None:
            # Warp wallpaper using homography
            warped_wallpaper = self._warp_wallpaper(wallpaper_image, homography, room_image.shape)
        else:
            # Fallback: simple scaling
            warped_wallpaper = cv2.resize(wallpaper_image, (room_image.shape[1], room_image.shape[0]))
        
        # Alpha blend wallpaper onto room
        composite = self._alpha_blend(composite, warped_wallpaper, blend_mask)
        
        return composite
    
    def _warp_wallpaper(
        self,
        wallpaper: np.ndarray,
        homography: np.ndarray,
        target_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Warp wallpaper using homography
        """
        h, w = target_shape[:2]
        
        try:
            # Warp wallpaper
            warped = cv2.warpPerspective(wallpaper, homography, (w, h))
            return warped
        except Exception as e:
            logger.warning(f"Homography warping failed: {e}. Using simple resize.")
            return cv2.resize(wallpaper, (w, h))
    
    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply feathering to mask edges
        """
        # Convert to float for processing
        mask_float = mask.astype(np.float32) / 255.0
        
        # Apply Gaussian blur for feathering
        feathered = cv2.GaussianBlur(mask_float, (self.feather_radius*2+1, self.feather_radius*2+1), 0)
        
        # Convert back to uint8
        return (feathered * 255).astype(np.uint8)
    
    def _alpha_blend(
        self,
        background: np.ndarray,
        foreground: np.ndarray,
        alpha_mask: np.ndarray
    ) -> np.ndarray:
        """
        Alpha blend foreground onto background using alpha mask
        """
        # Normalize alpha mask
        alpha = alpha_mask.astype(np.float32) / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        
        # Blend
        blended = alpha * foreground + (1 - alpha) * background
        
        return blended.astype(np.uint8)
    
    def _save_outputs(self, results: Dict, output_paths: Dict[str, Path]) -> None:
        """
        Save compositing outputs
        """
        try:
            # Save composite image
            if results['composite_image'] is not None and 'composite' in output_paths:
                save_image(results['composite_image'], output_paths['composite'])
                logger.info(f"Saved composite image: {output_paths['composite']}")
            
            # Save wall mask
            if results['wall_mask'] is not None and 'wall_mask' in output_paths:
                save_image(results['wall_mask'], output_paths['wall_mask'])
                logger.info(f"Saved wall mask: {output_paths['wall_mask']}")
            
            # Save objects mask
            if results['objects_mask'] is not None and 'objects_mask' in output_paths:
                save_image(results['objects_mask'], output_paths['objects_mask'])
                logger.info(f"Saved objects mask: {output_paths['objects_mask']}")
            
        except Exception as e:
            logger.error(f"Failed to save outputs: {e}")
    
    def process_batch(
        self,
        room_path: Path,
        wallpaper_paths: List[Path],
        output_dir: Path,
        room_name: Optional[str] = None
    ) -> List[Dict]:
        """
        Process a batch of wallpapers for a single room
        
        Args:
            room_path: Path to room image
            wallpaper_paths: List of wallpaper paths
            output_dir: Output directory
            room_name: Optional room name (defaults to filename)
            
        Returns:
            List of processing results
        """
        if room_name is None:
            room_name = room_path.stem
        
        # Load room image
        logger.info(f"Loading room image: {room_path}")
        room_image = load_image(room_path)
        
        results = []
        
        for i, wallpaper_path in enumerate(wallpaper_paths):
            logger.info(f"Processing wallpaper {i+1}/{len(wallpaper_paths)}: {wallpaper_path.name}")
            
            try:
                # Load wallpaper
                wallpaper_image = load_image(wallpaper_path)
                
                # Create output paths
                wallpaper_name = wallpaper_path.stem
                output_paths = {
                    'composite': output_dir / 'composites' / f"{room_name}__{wallpaper_name}.png",
                    'wall_mask': output_dir / 'masks' / f"{wallpaper_name}.wall.png",
                    'objects_mask': output_dir / 'masks' / f"{wallpaper_name}.objects.png",
                    'debug_panel': output_dir / 'debug' / f"{room_name}__{wallpaper_name}.panel.png"
                }
                
                # Process wallpaper
                result = self.composite_wallpaper(room_image, wallpaper_image, output_paths)
                result['wallpaper_path'] = wallpaper_path
                result['room_path'] = room_path
                
                results.append(result)
                
                if result['success']:
                    logger.info(f"Successfully processed: {wallpaper_name}")
                else:
                    logger.error(f"Failed to process: {wallpaper_name}")
                
            except Exception as e:
                logger.error(f"Error processing {wallpaper_path}: {e}")
                results.append({
                    'success': False,
                    'error': str(e),
                    'wallpaper_path': wallpaper_path,
                    'room_path': room_path
                })
        
        return results
    
    def get_processing_stats(self, results: List[Dict]) -> Dict:
        """
        Get processing statistics from batch results
        """
        total = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        failed = total - successful
        
        # Collect metadata
        wall_area_ratios = [r.get('metadata', {}).get('wall_area_ratio', 0) for r in results if r.get('success', False)]
        objects_area_ratios = [r.get('metadata', {}).get('objects_area_ratio', 0) for r in results if r.get('success', False)]
        polygon_vertices = [r.get('metadata', {}).get('polygon_vertices', 0) for r in results if r.get('success', False)]
        fallbacks = sum(1 for r in results if r.get('metadata', {}).get('is_fallback', False))
        
        stats = {
            'total_wallpapers': total,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total if total > 0 else 0,
            'avg_wall_area_ratio': np.mean(wall_area_ratios) if wall_area_ratios else 0,
            'avg_objects_area_ratio': np.mean(objects_area_ratios) if objects_area_ratios else 0,
            'avg_polygon_vertices': np.mean(polygon_vertices) if polygon_vertices else 0,
            'fallback_count': fallbacks,
            'fallback_rate': fallbacks / successful if successful > 0 else 0
        }
        
        return stats
