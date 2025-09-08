"""
Debug visualization and panel rendering
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Optional, Tuple, Union
import logging
from pathlib import Path

from .io_utils import save_image

logger = logging.getLogger(__name__)


class DebugVisualizer:
    """
    Create debug visualizations and panels for wallpaper compositing
    """
    
    def __init__(
        self,
        panel_size: Tuple[int, int] = (1200, 800),
        font_scale: float = 0.7,
        font_thickness: int = 2
    ):
        """
        Initialize debug visualizer
        
        Args:
            panel_size: Size of debug panel (width, height)
            font_scale: Font scale for text
            font_thickness: Font thickness for text
        """
        self.panel_size = panel_size
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        
        # Colors for visualization
        self.colors = {
            'wall': (0, 255, 0),      # Green
            'objects': (255, 0, 0),   # Red
            'polygon': (0, 0, 255),   # Blue
            'text': (255, 255, 255),  # White
            'background': (0, 0, 0)   # Black
        }
    
    def create_debug_panel(
        self,
        original_image: np.ndarray,
        wall_mask: np.ndarray,
        objects_mask: np.ndarray,
        polygon: Optional[np.ndarray],
        composite_image: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Create debug panel with all intermediate results
        
        Args:
            original_image: Original room image
            wall_mask: Wall mask
            objects_mask: Objects mask
            polygon: Wall polygon
            composite_image: Final composite image
            metadata: Optional metadata for display
            
        Returns:
            Debug panel image
        """
        # Create panel canvas
        panel = np.zeros((self.panel_size[1], self.panel_size[0], 3), dtype=np.uint8)
        
        # Calculate grid layout (2x3)
        cell_width = self.panel_size[0] // 3
        cell_height = self.panel_size[1] // 2
        margin = 10
        
        # Resize images to fit cells
        cell_size = (cell_width - 2*margin, cell_height - 2*margin)
        
        # 1. Original image
        original_resized = cv2.resize(original_image, cell_size)
        panel[margin:margin+cell_height-2*margin, margin:margin+cell_width-2*margin] = original_resized
        self._add_text(panel, "Original Room", (margin, margin-5))
        
        # 2. Wall mask
        wall_vis = self._create_mask_visualization(wall_mask, self.colors['wall'])
        wall_vis_resized = cv2.resize(wall_vis, cell_size)
        panel[margin:margin+cell_height-2*margin, cell_width+margin:cell_width*2-margin] = wall_vis_resized
        self._add_text(panel, "Wall Mask", (cell_width+margin, margin-5))
        
        # 3. Objects mask
        objects_vis = self._create_mask_visualization(objects_mask, self.colors['objects'])
        objects_vis_resized = cv2.resize(objects_vis, cell_size)
        panel[margin:margin+cell_height-2*margin, cell_width*2+margin:cell_width*3-margin] = objects_vis_resized
        self._add_text(panel, "Objects Mask", (cell_width*2+margin, margin-5))
        
        # 4. Polygon overlay
        polygon_vis = self._create_polygon_visualization(original_image, polygon, wall_mask)
        polygon_vis_resized = cv2.resize(polygon_vis, cell_size)
        panel[cell_height+margin:cell_height*2-margin, margin:margin+cell_width-2*margin] = polygon_vis_resized
        self._add_text(panel, "Wall Polygon", (margin, cell_height+margin-5))
        
        # 5. Final composite
        composite_resized = cv2.resize(composite_image, cell_size)
        panel[cell_height+margin:cell_height*2-margin, cell_width+margin:cell_width*2-margin] = composite_resized
        self._add_text(panel, "Final Composite", (cell_width+margin, cell_height+margin-5))
        
        # 6. Metadata/Stats
        stats_vis = self._create_stats_visualization(metadata)
        stats_vis_resized = cv2.resize(stats_vis, cell_size)
        panel[cell_height+margin:cell_height*2-margin, cell_width*2+margin:cell_width*3-margin] = stats_vis_resized
        self._add_text(panel, "Processing Stats", (cell_width*2+margin, cell_height+margin-5))
        
        return panel
    
    def _create_mask_visualization(self, mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
        """
        Create colored mask visualization
        """
        # Convert mask to 3-channel
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        # Apply color
        colored_mask = np.zeros_like(mask_3ch)
        colored_mask[mask > 128] = color
        
        # Blend with original mask for transparency effect
        alpha = 0.6
        result = cv2.addWeighted(mask_3ch, 1-alpha, colored_mask, alpha, 0)
        
        return result
    
    def _create_polygon_visualization(
        self,
        original_image: np.ndarray,
        polygon: Optional[np.ndarray],
        wall_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create polygon overlay visualization
        """
        vis = original_image.copy()
        
        if polygon is not None and len(polygon) > 0:
            # Draw polygon
            polygon_int = polygon.astype(np.int32)
            cv2.polylines(vis, [polygon_int], True, self.colors['polygon'], 3)
            
            # Draw vertices
            for i, point in enumerate(polygon_int):
                cv2.circle(vis, tuple(point), 8, self.colors['polygon'], -1)
                cv2.putText(vis, str(i), (point[0]+10, point[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        # Overlay wall mask with transparency
        wall_overlay = self._create_mask_visualization(wall_mask, self.colors['wall'])
        alpha = 0.3
        vis = cv2.addWeighted(vis, 1-alpha, wall_overlay, alpha, 0)
        
        return vis
    
    def _create_stats_visualization(self, metadata: Optional[Dict]) -> np.ndarray:
        """
        Create statistics visualization
        """
        # Create black background
        stats_img = np.zeros((400, 300, 3), dtype=np.uint8)
        
        if metadata is None:
            cv2.putText(stats_img, "No metadata", (50, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['text'], 2)
            return stats_img
        
        y_pos = 30
        line_height = 25
        
        # Display key statistics
        stats_text = [
            f"Wall Area: {metadata.get('wall_area_ratio', 0):.1%}",
            f"Objects Area: {metadata.get('objects_area_ratio', 0):.1%}",
            f"Polygon Vertices: {metadata.get('polygon_vertices', 0)}",
            f"Depth Consistency: {metadata.get('depth_consistency', 0):.2f}",
            f"Fallback: {'Yes' if metadata.get('is_fallback', False) else 'No'}"
        ]
        
        for text in stats_text:
            cv2.putText(stats_img, text, (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 1)
            y_pos += line_height
        
        return stats_img
    
    def _add_text(self, image: np.ndarray, text: str, position: Tuple[int, int]) -> None:
        """
        Add text to image
        """
        cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   self.font_scale, self.colors['text'], self.font_thickness)
    
    def create_comparison_grid(
        self,
        images: List[np.ndarray],
        titles: List[str],
        grid_size: Tuple[int, int] = (2, 2)
    ) -> np.ndarray:
        """
        Create a grid comparison of multiple images
        
        Args:
            images: List of images to display
            titles: List of titles for each image
            grid_size: Grid size (rows, cols)
            
        Returns:
            Grid image
        """
        rows, cols = grid_size
        cell_width = self.panel_size[0] // cols
        cell_height = self.panel_size[1] // rows
        margin = 10
        
        grid = np.zeros((self.panel_size[1], self.panel_size[0], 3), dtype=np.uint8)
        
        for i, (image, title) in enumerate(zip(images, titles)):
            if i >= rows * cols:
                break
            
            row = i // cols
            col = i % cols
            
            # Calculate position
            x_start = col * cell_width + margin
            y_start = row * cell_height + margin
            x_end = (col + 1) * cell_width - margin
            y_end = (row + 1) * cell_height - margin
            
            # Resize image to fit cell
            cell_size = (x_end - x_start, y_end - y_start)
            resized_image = cv2.resize(image, cell_size)
            
            # Place image
            grid[y_start:y_end, x_start:x_end] = resized_image
            
            # Add title
            cv2.putText(grid, title, (x_start, y_start-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return grid
    
    def create_depth_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create depth map visualization
        """
        # Normalize depth map to 0-255
        depth_normalized = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        # Apply colormap
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        
        return depth_colored
    
    def create_homography_visualization(
        self,
        wallpaper: np.ndarray,
        homography: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Create homography transformation visualization
        """
        # Create a grid of points on the wallpaper
        h, w = wallpaper.shape[:2]
        grid_size = 20
        
        # Create grid points
        x_coords = np.linspace(0, w-1, grid_size, dtype=np.int32)
        y_coords = np.linspace(0, h-1, grid_size, dtype=np.int32)
        
        grid_points = []
        for y in y_coords:
            for x in x_coords:
                grid_points.append([x, y])
        
        grid_points = np.array(grid_points, dtype=np.float32)
        
        # Transform points
        if homography is not None:
            try:
                transformed_points = cv2.perspectiveTransform(
                    grid_points.reshape(-1, 1, 2), homography
                ).reshape(-1, 2)
            except:
                transformed_points = grid_points
        else:
            transformed_points = grid_points
        
        # Create visualization
        vis = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
        
        # Draw grid lines
        for i in range(grid_size):
            for j in range(grid_size-1):
                # Horizontal lines
                idx1 = i * grid_size + j
                idx2 = i * grid_size + j + 1
                if (idx1 < len(transformed_points) and idx2 < len(transformed_points) and
                    self._is_point_in_bounds(transformed_points[idx1], target_shape) and
                    self._is_point_in_bounds(transformed_points[idx2], target_shape)):
                    cv2.line(vis, 
                            tuple(transformed_points[idx1].astype(np.int32)),
                            tuple(transformed_points[idx2].astype(np.int32)),
                            (0, 255, 0), 1)
            
            for j in range(grid_size):
                for k in range(grid_size-1):
                    # Vertical lines
                    idx1 = j * grid_size + i
                    idx2 = (j + 1) * grid_size + i
                    if (idx1 < len(transformed_points) and idx2 < len(transformed_points) and
                        self._is_point_in_bounds(transformed_points[idx1], target_shape) and
                        self._is_point_in_bounds(transformed_points[idx2], target_shape)):
                        cv2.line(vis, 
                                tuple(transformed_points[idx1].astype(np.int32)),
                                tuple(transformed_points[idx2].astype(np.int32)),
                                (0, 255, 0), 1)
        
        return vis
    
    def _is_point_in_bounds(self, point: np.ndarray, shape: Tuple[int, int]) -> bool:
        """
        Check if point is within image bounds
        """
        return (0 <= point[0] < shape[1] and 0 <= point[1] < shape[0])
    
    def save_debug_panel(
        self,
        panel: np.ndarray,
        output_path: Union[str, Path],
        quality: int = 95
    ) -> None:
        """
        Save debug panel to file
        """
        save_image(panel, output_path, quality)
        logger.info(f"Saved debug panel: {output_path}")
    
    def create_processing_timeline(
        self,
        processing_times: Dict[str, float],
        total_time: float
    ) -> np.ndarray:
        """
        Create processing timeline visualization
        """
        # Create timeline image
        timeline_height = 200
        timeline_width = 800
        timeline = np.zeros((timeline_height, timeline_width, 3), dtype=np.uint8)
        
        # Calculate bar positions
        bar_width = timeline_width // len(processing_times)
        bar_height = timeline_height - 40
        
        x_pos = 0
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        
        for i, (step, time) in enumerate(processing_times.items()):
            # Calculate bar height proportional to time
            height = int((time / total_time) * bar_height)
            
            # Draw bar
            color = colors[i % len(colors)]
            cv2.rectangle(timeline, (x_pos, timeline_height - height - 20), 
                         (x_pos + bar_width - 10, timeline_height - 20), color, -1)
            
            # Add step name
            cv2.putText(timeline, step, (x_pos, timeline_height - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add time
            time_text = f"{time:.1f}s"
            cv2.putText(timeline, time_text, (x_pos, 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            x_pos += bar_width
        
        return timeline
