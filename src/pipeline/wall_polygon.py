"""
Wall polygonization and homography estimation
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class WallPolygonizer:
    """
    Convert wall mask to polygon and estimate homography for wallpaper projection
    """
    
    def __init__(
        self,
        min_polygon_area: float = 0.01,
        max_vertices: int = 6,
        min_vertices: int = 4
    ):
        """
        Initialize wall polygonizer
        
        Args:
            min_polygon_area: Minimum area ratio for valid polygon
            max_vertices: Maximum number of vertices
            min_vertices: Minimum number of vertices
        """
        self.min_polygon_area = min_polygon_area
        self.max_vertices = max_vertices
        self.min_vertices = min_vertices
    
    def extract_wall_polygon(self, wall_mask: np.ndarray) -> Dict:
        """
        Extract wall polygon from mask
        
        Args:
            wall_mask: Binary wall mask
            
        Returns:
            Dictionary with polygon info and homography
        """
        # Try morphology close to connect regions before finding contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask_connected = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            logger.warning("No contours found in wall mask")
            return self._create_fallback_polygon(wall_mask)
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Check if contour is large enough
        total_area = wall_mask.shape[0] * wall_mask.shape[1]
        contour_area = cv2.contourArea(largest_contour)
        
        if contour_area / total_area < self.min_polygon_area:
            logger.warning(f"Contour too small: {contour_area/total_area:.3f} < {self.min_polygon_area}")
            return self._create_fallback_polygon(wall_mask)
        
        # Approximate polygon
        polygon = self._approximate_polygon(largest_contour)
        
        if polygon is None or len(polygon) < self.min_vertices:
            logger.warning("Failed to create valid polygon, using fallback")
            return self._create_fallback_polygon(wall_mask)
        
        # Order polygon vertices
        ordered_polygon = self._order_polygon_vertices(polygon)
        
        # Estimate homography
        homography = self._estimate_homography(ordered_polygon, wall_mask.shape)
        
        return {
            'polygon': ordered_polygon,
            'homography': homography,
            'success': True,
            'area_ratio': contour_area / total_area,
            'n_vertices': len(ordered_polygon)
        }
    
    def _approximate_polygon(self, contour: np.ndarray) -> Optional[np.ndarray]:
        """
        Approximate contour to polygon with controlled number of vertices
        """
        # Try different epsilon values to get desired number of vertices
        for epsilon_factor in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            if self.min_vertices <= len(approx) <= self.max_vertices:
                return approx.reshape(-1, 2)
        
        # If no good approximation found, use the one with closest number of vertices
        best_approx = None
        best_diff = float('inf')
        
        for epsilon_factor in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            diff = abs(len(approx) - 4)  # Prefer 4 vertices
            if diff < best_diff:
                best_diff = diff
                best_approx = approx
        
        if best_approx is not None:
            return best_approx.reshape(-1, 2)
        
        return None
    
    def _order_polygon_vertices(self, polygon: np.ndarray) -> np.ndarray:
        """
        Order polygon vertices in clockwise order starting from top-left
        """
        # Find centroid
        centroid = np.mean(polygon, axis=0)
        
        # Calculate angles from centroid
        angles = np.arctan2(polygon[:, 1] - centroid[1], polygon[:, 0] - centroid[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        ordered_polygon = polygon[sorted_indices]
        
        # Ensure clockwise order
        if self._is_counterclockwise(ordered_polygon):
            ordered_polygon = ordered_polygon[::-1]
        
        return ordered_polygon
    
    def _is_counterclockwise(self, polygon: np.ndarray) -> bool:
        """Check if polygon is counterclockwise"""
        total = 0
        n = len(polygon)
        for i in range(n):
            j = (i + 1) % n
            total += (polygon[j, 0] - polygon[i, 0]) * (polygon[j, 1] + polygon[i, 1])
        return total < 0
    
    def _estimate_homography(self, polygon: np.ndarray, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Estimate homography from wallpaper rectangle to wall polygon
        """
        if len(polygon) < 4:
            logger.warning("Need at least 4 points for homography")
            return None
        
        # Define source points (wallpaper rectangle)
        h, w = image_shape[:2]
        wallpaper_w, wallpaper_h = w, h  # Use full image size for wallpaper
        
        # Create a rectangle that fits the wall polygon's aspect ratio
        wall_bbox = cv2.boundingRect(polygon.astype(np.int32))
        wall_w, wall_h = wall_bbox[2], wall_bbox[3]
        
        # Scale wallpaper to match wall aspect ratio
        aspect_ratio = wall_w / wall_h
        if aspect_ratio > 1:
            # Wall is wider than tall
            src_w = int(wallpaper_h * aspect_ratio)
            src_h = wallpaper_h
        else:
            # Wall is taller than wide
            src_w = wallpaper_w
            src_h = int(wallpaper_w / aspect_ratio)
        
        # Center the source rectangle
        src_x = (wallpaper_w - src_w) // 2
        src_y = (wallpaper_h - src_h) // 2
        
        src_points = np.array([
            [src_x, src_y],
            [src_x + src_w, src_y],
            [src_x + src_w, src_y + src_h],
            [src_x, src_y + src_h]
        ], dtype=np.float32)
        
        # Use first 4 points of polygon as destination
        dst_points = polygon[:4].astype(np.float32)
        
        try:
            # Estimate homography
            homography, mask = cv2.findHomography(
                src_points, dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=5.0
            )
            
            if homography is None:
                logger.warning("Failed to estimate homography")
                return None
            
            # Validate homography
            if self._validate_homography(homography, src_points, dst_points):
                return homography
            else:
                logger.warning("Homography validation failed")
                return None
                
        except Exception as e:
            logger.warning(f"Homography estimation failed: {e}")
            return None
    
    def _validate_homography(self, homography: np.ndarray, src_points: np.ndarray, dst_points: np.ndarray) -> bool:
        """
        Validate homography by checking reprojection error
        """
        try:
            # Transform source points
            src_homogeneous = np.column_stack([src_points, np.ones(len(src_points))])
            projected = homography @ src_homogeneous.T
            projected = projected[:2] / projected[2]
            projected = projected.T
            
            # Calculate reprojection error
            error = np.mean(np.linalg.norm(projected - dst_points, axis=1))
            
            # Accept if error is reasonable
            return error < 50.0  # pixels
            
        except Exception:
            return False
    
    def _create_fallback_polygon(self, wall_mask: np.ndarray) -> Dict:
        """
        Create fallback polygon using minimal area rectangle
        """
        logger.info("Creating fallback polygon using minimal area rectangle")
        
        # Find contours
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # Create a simple rectangle covering the entire image
            h, w = wall_mask.shape[:2]
            polygon = np.array([
                [w*0.1, h*0.1],
                [w*0.9, h*0.1],
                [w*0.9, h*0.9],
                [w*0.1, h*0.9]
            ], dtype=np.float32)
        else:
            # Use minimal area rectangle
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            polygon = box.astype(np.float32)
        
        # Try to estimate homography
        homography = self._estimate_homography(polygon, wall_mask.shape)
        
        return {
            'polygon': polygon,
            'homography': homography,
            'success': homography is not None,
            'area_ratio': 1.0,
            'n_vertices': len(polygon),
            'fallback': True
        }
    
    def detect_vanishing_points(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect vanishing points for perspective correction
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (horizontal_vanishing_point, vertical_vanishing_point)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return None, None
        
        # Separate horizontal and vertical lines
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            
            # Classify lines as horizontal or vertical
            if 80 < angle < 100:  # Horizontal lines
                horizontal_lines.append(line[0])
            elif angle < 10 or angle > 170:  # Vertical lines
                vertical_lines.append(line[0])
        
        # Find vanishing points
        h_vp = self._find_vanishing_point(horizontal_lines) if horizontal_lines else None
        v_vp = self._find_vanishing_point(vertical_lines) if vertical_lines else None
        
        return h_vp, v_vp
    
    def _find_vanishing_point(self, lines: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Find vanishing point from a set of lines
        """
        if len(lines) < 2:
            return None
        
        # Convert lines to homogeneous coordinates
        line_coords = []
        for rho, theta in lines:
            a = np.cos(theta)
            b = np.sin(theta)
            c = -rho
            line_coords.append([a, b, c])
        
        line_coords = np.array(line_coords)
        
        # Find intersection point (vanishing point)
        # Solve system of equations: A * vp = 0
        _, _, vh = np.linalg.svd(line_coords)
        vp_homogeneous = vh[-1]
        
        if abs(vp_homogeneous[2]) < 1e-6:
            return None
        
        # Convert to Cartesian coordinates
        vp = vp_homogeneous[:2] / vp_homogeneous[2]
        
        return vp
    
    def align_polygon_to_vanishing_points(self, polygon: np.ndarray, h_vp: Optional[np.ndarray], v_vp: Optional[np.ndarray]) -> np.ndarray:
        """
        Align polygon edges to vanishing points for better perspective
        """
        if h_vp is None and v_vp is None:
            return polygon
        
        aligned_polygon = polygon.copy()
        
        # Align horizontal edges to horizontal vanishing point
        if h_vp is not None:
            aligned_polygon = self._align_edges_to_vp(aligned_polygon, h_vp, 'horizontal')
        
        # Align vertical edges to vertical vanishing point
        if v_vp is not None:
            aligned_polygon = self._align_edges_to_vp(aligned_polygon, v_vp, 'vertical')
        
        return aligned_polygon
    
    def _align_edges_to_vp(self, polygon: np.ndarray, vp: np.ndarray, direction: str) -> np.ndarray:
        """
        Align polygon edges to vanishing point
        """
        aligned = polygon.copy()
        n = len(polygon)
        
        for i in range(n):
            j = (i + 1) % n
            
            # Get edge direction
            edge = polygon[j] - polygon[i]
            
            # Check if edge is roughly in the target direction
            if direction == 'horizontal' and abs(edge[1]) > abs(edge[0]):
                continue
            elif direction == 'vertical' and abs(edge[0]) > abs(edge[1]):
                continue
            
            # Extend edge through vanishing point
            direction_vec = vp - polygon[i]
            direction_vec = direction_vec / np.linalg.norm(direction_vec)
            
            # Update vertex j to be on the line through vp
            edge_length = np.linalg.norm(edge)
            aligned[j] = polygon[i] + direction_vec * edge_length
        
        return aligned
