"""
Sanity check script for wallpaper pipeline validation
"""

import argparse
import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple
import json

from ..pipeline import (
    SemanticSegmenter,
    InstanceSegmenter,
    DepthPlaneRefiner,
    WallPolygonizer,
    WallpaperCompositor,
    DebugVisualizer,
    load_image,
    get_image_files,
    set_seed
)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


class SanityChecker:
    """
    Sanity checker for wallpaper pipeline components
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize sanity checker
        
        Args:
            device: Device to run models on
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.semantic_segmenter = SemanticSegmenter(device=device)
        self.instance_segmenter = InstanceSegmenter(device=device)
        self.depth_refiner = DepthPlaneRefiner(device=device, use_depth=True)
        self.polygonizer = WallPolygonizer()
        self.compositor = WallpaperCompositor(device=device, use_depth=True)
        self.debug_visualizer = DebugVisualizer()
    
    def check_semantic_segmentation(self, image: np.ndarray) -> Dict:
        """
        Check semantic segmentation quality
        """
        self.logger.info("Checking semantic segmentation...")
        
        # Get wall mask
        wall_mask = self.semantic_segmenter.get_wall_mask(image)
        
        # Validate mask
        is_valid = self.semantic_segmenter.validate_wall_mask(wall_mask)
        
        # Compute metrics
        total_pixels = image.shape[0] * image.shape[1]
        wall_pixels = np.sum(wall_mask > 128)
        wall_area_ratio = wall_pixels / total_pixels
        
        # Check for reasonable wall shape
        contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        n_contours = len(contours)
        
        # Check wall connectivity
        largest_contour_area = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            largest_contour_area = cv2.contourArea(largest_contour)
            largest_contour_ratio = largest_contour_area / total_pixels
        
        return {
            'is_valid': is_valid,
            'wall_area_ratio': wall_area_ratio,
            'n_contours': n_contours,
            'largest_contour_ratio': largest_contour_ratio if contours else 0,
            'wall_mask': wall_mask
        }
    
    def check_instance_segmentation(self, image: np.ndarray) -> Dict:
        """
        Check instance segmentation quality
        """
        self.logger.info("Checking instance segmentation...")
        
        # Get objects mask
        objects_mask = self.instance_segmenter.get_objects_mask(image)
        
        # Get detailed detections
        detections = self.instance_segmenter.get_detailed_objects(image)
        
        # Validate mask
        is_valid = self.instance_segmenter.validate_objects_mask(objects_mask)
        
        # Compute metrics
        total_pixels = image.shape[0] * image.shape[1]
        object_pixels = np.sum(objects_mask > 128)
        objects_area_ratio = object_pixels / total_pixels
        
        # Count unique objects
        unique_classes = set(det['class'] for det in detections)
        n_unique_objects = len(unique_classes)
        
        # Check object sizes
        object_areas = []
        for detection in detections:
            bbox = detection['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            object_areas.append(area)
        
        avg_object_area = np.mean(object_areas) if object_areas else 0
        
        return {
            'is_valid': is_valid,
            'objects_area_ratio': objects_area_ratio,
            'n_detections': len(detections),
            'n_unique_classes': n_unique_objects,
            'unique_classes': list(unique_classes),
            'avg_object_area': avg_object_area,
            'objects_mask': objects_mask,
            'detections': detections
        }
    
    def check_depth_estimation(self, image: np.ndarray, wall_mask: np.ndarray) -> Dict:
        """
        Check depth estimation quality
        """
        self.logger.info("Checking depth estimation...")
        
        # Estimate depth
        depth_map = self.depth_refiner.estimate_depth(image)
        
        # Fit plane to wall
        plane_result = self.depth_refiner.fit_plane_to_wall(depth_map, wall_mask)
        
        # Get depth consistency
        depth_consistency = self.depth_refiner.get_wall_depth_consistency(depth_map, wall_mask)
        
        return {
            'depth_map': depth_map,
            'plane_fitting_success': plane_result['success'],
            'depth_consistency': depth_consistency,
            'n_inliers': plane_result.get('n_inliers', 0),
            'n_total': plane_result.get('n_total', 0),
            'plane_params': plane_result.get('plane_params', None)
        }
    
    def check_polygonization(self, wall_mask: np.ndarray) -> Dict:
        """
        Check wall polygonization quality
        """
        self.logger.info("Checking wall polygonization...")
        
        # Extract polygon
        polygon_result = self.polygonizer.extract_wall_polygon(wall_mask)
        
        # Check polygon quality
        polygon_quality = self._assess_polygon_quality(polygon_result)
        
        return {
            'success': polygon_result['success'],
            'n_vertices': polygon_result.get('n_vertices', 0),
            'area_ratio': polygon_result.get('area_ratio', 0),
            'is_fallback': polygon_result.get('fallback', False),
            'polygon': polygon_result.get('polygon', None),
            'homography': polygon_result.get('homography', None),
            'quality_score': polygon_quality
        }
    
    def _assess_polygon_quality(self, polygon_result: Dict) -> float:
        """
        Assess polygon quality (0-1, higher is better)
        """
        if not polygon_result['success']:
            return 0.0
        
        score = 1.0
        
        # Penalize fallback
        if polygon_result.get('fallback', False):
            score *= 0.5
        
        # Prefer 4 vertices
        n_vertices = polygon_result.get('n_vertices', 0)
        if n_vertices == 4:
            score *= 1.0
        elif n_vertices in [3, 5, 6]:
            score *= 0.8
        else:
            score *= 0.6
        
        # Prefer reasonable area ratio
        area_ratio = polygon_result.get('area_ratio', 0)
        if 0.1 <= area_ratio <= 0.8:
            score *= 1.0
        elif 0.05 <= area_ratio <= 0.9:
            score *= 0.8
        else:
            score *= 0.6
        
        return score
    
    def check_homography_quality(self, homography: np.ndarray, image_shape: Tuple[int, int]) -> Dict:
        """
        Check homography quality
        """
        if homography is None:
            return {'is_valid': False, 'condition_number': float('inf'), 'determinant': 0}
        
        # Check determinant
        det = np.linalg.det(homography)
        
        # Check condition number
        try:
            condition_number = np.linalg.cond(homography)
        except:
            condition_number = float('inf')
        
        # Check if homography is well-conditioned
        is_valid = (abs(det) > 1e-6 and condition_number < 1e6)
        
        return {
            'is_valid': is_valid,
            'determinant': det,
            'condition_number': condition_number
        }
    
    def check_edge_consistency(self, image: np.ndarray, wall_mask: np.ndarray) -> Dict:
        """
        Check consistency between wall mask and image edges
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        
        # Find wall mask edges
        wall_edges = cv2.Canny(wall_mask, 50, 150)
        
        # Compute edge overlap
        edge_overlap = cv2.bitwise_and(edges, wall_edges)
        overlap_ratio = np.sum(edge_overlap > 0) / max(np.sum(edges > 0), 1)
        
        return {
            'edge_overlap_ratio': overlap_ratio,
            'image_edges': edges,
            'wall_edges': wall_edges,
            'overlap_edges': edge_overlap
        }
    
    def run_full_sanity_check(self, image_path: Path) -> Dict:
        """
        Run full sanity check on an image
        """
        self.logger.info(f"Running sanity check on: {image_path}")
        
        # Load image
        image = load_image(image_path)
        
        results = {
            'image_path': str(image_path),
            'image_shape': image.shape,
            'semantic_segmentation': {},
            'instance_segmentation': {},
            'depth_estimation': {},
            'polygonization': {},
            'homography': {},
            'edge_consistency': {},
            'overall_score': 0.0
        }
        
        try:
            # Check semantic segmentation
            results['semantic_segmentation'] = self.check_semantic_segmentation(image)
            
            if not results['semantic_segmentation']['is_valid']:
                self.logger.warning("Semantic segmentation failed, skipping other checks")
                return results
            
            # Check instance segmentation
            results['instance_segmentation'] = self.check_instance_segmentation(image)
            
            # Check depth estimation
            wall_mask = results['semantic_segmentation']['wall_mask']
            results['depth_estimation'] = self.check_depth_estimation(image, wall_mask)
            
            # Check polygonization
            results['polygonization'] = self.check_polygonization(wall_mask)
            
            # Check homography
            if results['polygonization']['homography'] is not None:
                results['homography'] = self.check_homography_quality(
                    results['polygonization']['homography'], image.shape
                )
            
            # Check edge consistency
            results['edge_consistency'] = self.check_edge_consistency(image, wall_mask)
            
            # Compute overall score
            results['overall_score'] = self._compute_overall_score(results)
            
        except Exception as e:
            self.logger.error(f"Error during sanity check: {e}")
            results['error'] = str(e)
        
        return results
    
    def _compute_overall_score(self, results: Dict) -> float:
        """
        Compute overall quality score (0-1, higher is better)
        """
        score = 0.0
        
        # Semantic segmentation (30%)
        if results['semantic_segmentation']['is_valid']:
            wall_ratio = results['semantic_segmentation']['wall_area_ratio']
            if 0.1 <= wall_ratio <= 0.7:
                score += 0.3
            else:
                score += 0.15
        
        # Instance segmentation (20%)
        if results['instance_segmentation']['is_valid']:
            objects_ratio = results['instance_segmentation']['objects_area_ratio']
            if 0.05 <= objects_ratio <= 0.5:
                score += 0.2
            else:
                score += 0.1
        
        # Depth estimation (15%)
        depth_consistency = results['depth_estimation']['depth_consistency']
        score += 0.15 * depth_consistency
        
        # Polygonization (20%)
        if results['polygonization']['success']:
            score += 0.2 * results['polygonization']['quality_score']
        
        # Homography (10%)
        if results['homography'].get('is_valid', False):
            score += 0.1
        
        # Edge consistency (5%)
        edge_overlap = results['edge_consistency']['edge_overlap_ratio']
        score += 0.05 * min(edge_overlap * 2, 1.0)
        
        return min(score, 1.0)
    
    def create_sanity_report(self, results: Dict, output_path: Path) -> None:
        """
        Create sanity check report
        """
        # Create visualization
        image = load_image(results['image_path'])
        
        # Create comparison grid
        images = [image]
        titles = ["Original"]
        
        if 'wall_mask' in results['semantic_segmentation']:
            wall_vis = self.debug_visualizer._create_mask_visualization(
                results['semantic_segmentation']['wall_mask'], (0, 255, 0)
            )
            images.append(wall_vis)
            titles.append("Wall Mask")
        
        if 'objects_mask' in results['instance_segmentation']:
            objects_vis = self.debug_visualizer._create_mask_visualization(
                results['instance_segmentation']['objects_mask'], (255, 0, 0)
            )
            images.append(objects_vis)
            titles.append("Objects Mask")
        
        if 'depth_map' in results['depth_estimation']:
            depth_vis = self.debug_visualizer.create_depth_visualization(
                results['depth_estimation']['depth_map']
            )
            images.append(depth_vis)
            titles.append("Depth Map")
        
        # Create grid
        grid = self.debug_visualizer.create_comparison_grid(images, titles, (2, 2))
        
        # Save visualization
        self.debug_visualizer.save_debug_panel(grid, output_path)
        
        # Save JSON report
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            # Remove numpy arrays and other non-serializable objects
            clean_results = self._clean_results_for_json(results)
            json.dump(clean_results, f, indent=2)
        
        self.logger.info(f"Sanity report saved: {output_path}")
        self.logger.info(f"JSON report saved: {json_path}")
    
    def _clean_results_for_json(self, results: Dict) -> Dict:
        """
        Clean results for JSON serialization
        """
        clean_results = {}
        
        for key, value in results.items():
            if key in ['wall_mask', 'objects_mask', 'depth_map', 'image_edges', 'wall_edges', 'overlap_edges']:
                # Skip numpy arrays
                continue
            elif isinstance(value, dict):
                clean_results[key] = self._clean_results_for_json(value)
            elif isinstance(value, (list, tuple)):
                clean_results[key] = [self._clean_results_for_json(item) if isinstance(item, dict) else item for item in value]
            elif isinstance(value, np.ndarray):
                # Convert numpy arrays to lists
                clean_results[key] = value.tolist()
            else:
                clean_results[key] = value
        
        return clean_results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Sanity check for wallpaper pipeline")
    
    parser.add_argument(
        "--rooms-dir",
        type=str,
        default="/Users/alireza/Documents/wallpaper_pipeline/src/data/rooms",
        help="Directory containing room images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/Users/alireza/Documents/wallpaper_pipeline/src/data/out/sanity",
        help="Output directory for sanity check results"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=3,
        help="Number of room samples to check"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to run models on"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Validate input directory
    rooms_dir = Path(args.rooms_dir)
    if not rooms_dir.exists():
        logger.error(f"Rooms directory does not exist: {rooms_dir}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get room images
    room_files = get_image_files(rooms_dir)
    if not room_files:
        logger.error("No room images found")
        return 1
    
    # Select samples
    sample_files = room_files[:args.num_samples]
    logger.info(f"Running sanity check on {len(sample_files)} room samples")
    
    # Initialize sanity checker
    checker = SanityChecker(device=args.device)
    
    # Run sanity checks
    all_results = []
    for i, room_path in enumerate(sample_files):
        logger.info(f"Checking sample {i+1}/{len(sample_files)}: {room_path.name}")
        
        try:
            # Run sanity check
            results = checker.run_full_sanity_check(room_path)
            all_results.append(results)
            
            # Create report
            report_path = output_dir / f"sanity_check_{room_path.stem}.png"
            checker.create_sanity_report(results, report_path)
            
            # Print summary
            logger.info(f"Overall score: {results['overall_score']:.3f}")
            logger.info(f"Wall area ratio: {results['semantic_segmentation']['wall_area_ratio']:.3f}")
            logger.info(f"Objects area ratio: {results['instance_segmentation']['objects_area_ratio']:.3f}")
            logger.info(f"Polygon vertices: {results['polygonization']['n_vertices']}")
            logger.info(f"Depth consistency: {results['depth_estimation']['depth_consistency']:.3f}")
            
        except Exception as e:
            logger.error(f"Error checking {room_path}: {e}")
            continue
    
    # Print overall summary
    if all_results:
        avg_score = np.mean([r['overall_score'] for r in all_results])
        logger.info("=" * 60)
        logger.info("SANITY CHECK SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Average overall score: {avg_score:.3f}")
        logger.info(f"Checked {len(all_results)} samples")
        
        # Save overall report
        overall_report = {
            'summary': {
                'avg_overall_score': avg_score,
                'n_samples': len(all_results),
                'scores': [r['overall_score'] for r in all_results]
            },
            'detailed_results': all_results
        }
        
        overall_report_path = output_dir / "overall_sanity_report.json"
        with open(overall_report_path, 'w') as f:
            json.dump(checker._clean_results_for_json(overall_report), f, indent=2)
        
        logger.info(f"Overall report saved: {overall_report_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
