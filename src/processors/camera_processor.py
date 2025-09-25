"""Camera feed processor with OpenCV for vehicle detection."""

import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from ..models.vehicle_detection import VehicleDetection

logger = logging.getLogger(__name__)


@dataclass
class DetectionBox:
    """Represents a detection bounding box."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    class_id: int


class CameraProcessor:
    """Processes camera feeds for vehicle detection using OpenCV."""
    
    def __init__(self, camera_config: Dict, detection_threshold: float = 0.8):
        self.camera_config = camera_config
        self.detection_threshold = detection_threshold
        self.vehicle_counter = 0
        
        # YOLO class IDs for vehicles (COCO dataset)
        self.vehicle_class_ids = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50
        )
        
        # Tracking variables
        self.tracked_vehicles: Dict[str, Dict] = {}
        self.next_vehicle_id = 1
        
        logger.info("CameraProcessor initialized")
    
    def process_frame(self, frame: np.ndarray, timestamp: datetime, 
                     camera_position: str) -> List[VehicleDetection]:
        """Process a single frame and return vehicle detections."""
        if frame is None or frame.size == 0:
            logger.warning("Empty frame received")
            return []
        
        try:
            # For simulation, we'll use background subtraction and contour detection
            # In a real implementation, this would use YOLO or similar
            detections = self._detect_vehicles_simulation(frame, timestamp, camera_position)
            
            logger.debug(f"Processed frame with {len(detections)} detections")
            return detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []
    
    def _detect_vehicles_simulation(self, frame: np.ndarray, timestamp: datetime,
                                  camera_position: str) -> List[VehicleDetection]:
        """Simulate vehicle detection using background subtraction."""
        detections = []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size to identify potential vehicles
        min_area = 500  # Minimum area for a vehicle
        max_area = 5000  # Maximum area for a vehicle
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if min_area < area < max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center position
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Convert to world coordinates (simplified)
                world_pos = self._convert_to_world_coordinates(
                    center_x, center_y, camera_position
                )
                
                # Determine vehicle type based on size
                vehicle_type = self._classify_vehicle_by_size(w, h)
                
                # Generate vehicle ID
                vehicle_id = f"cam_{camera_position}_vehicle_{self.next_vehicle_id:04d}"
                self.next_vehicle_id += 1
                
                # Determine lane based on position
                lane = self._determine_lane(world_pos, camera_position)
                
                # Create detection
                detection = VehicleDetection(
                    vehicle_id=vehicle_id,
                    vehicle_type=vehicle_type,
                    position=world_pos,
                    lane=lane,
                    confidence=min(0.95, 0.7 + (area / max_area) * 0.25),  # Confidence based on size
                    timestamp=timestamp
                )
                
                detections.append(detection)
        
        return detections
    
    def _convert_to_world_coordinates(self, pixel_x: int, pixel_y: int, 
                                    camera_position: str) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates."""
        # Simplified conversion - in reality this would use camera calibration
        # Assume camera covers a 100m x 100m area
        
        frame_width = 640  # Assumed frame width
        frame_height = 480  # Assumed frame height
        
        # Convert to normalized coordinates (-1 to 1)
        norm_x = (pixel_x / frame_width) * 2 - 1
        norm_y = (pixel_y / frame_height) * 2 - 1
        
        # Scale to world coordinates (meters)
        world_scale = 50  # 50 meters from center
        
        if camera_position == 'north':
            return (norm_x * world_scale, norm_y * world_scale + 50)
        elif camera_position == 'south':
            return (norm_x * world_scale, norm_y * world_scale - 50)
        elif camera_position == 'east':
            return (norm_x * world_scale + 50, norm_y * world_scale)
        elif camera_position == 'west':
            return (norm_x * world_scale - 50, norm_y * world_scale)
        else:
            return (norm_x * world_scale, norm_y * world_scale)
    
    def _classify_vehicle_by_size(self, width: int, height: int) -> str:
        """Classify vehicle type based on bounding box size."""
        area = width * height
        aspect_ratio = width / height if height > 0 else 1
        
        if area < 1000:
            return 'motorcycle'
        elif area > 3000 or aspect_ratio > 2.5:
            if aspect_ratio > 3.0:
                return 'truck'
            else:
                return 'bus'
        else:
            return 'car'
    
    def _determine_lane(self, world_pos: Tuple[float, float], 
                       camera_position: str) -> str:
        """Determine which lane a vehicle is in based on position."""
        x, y = world_pos
        
        # Simplified lane determination
        if camera_position == 'north':
            if x < -2:
                return 'north_lane_1'
            elif x > 2:
                return 'north_lane_2'
            else:
                return 'north_lane_1'
        elif camera_position == 'south':
            if x < -2:
                return 'south_lane_1'
            elif x > 2:
                return 'south_lane_2'
            else:
                return 'south_lane_1'
        elif camera_position == 'east':
            if y < -2:
                return 'east_lane_1'
            elif y > 2:
                return 'east_lane_2'
            else:
                return 'east_lane_1'
        elif camera_position == 'west':
            if y < -2:
                return 'west_lane_1'
            elif y > 2:
                return 'west_lane_2'
            else:
                return 'west_lane_1'
        else:
            return 'unknown_lane'
    
    def create_synthetic_frame(self, width: int = 640, height: int = 480,
                             num_vehicles: int = 5) -> np.ndarray:
        """Create a synthetic frame with simulated vehicles for testing."""
        # Create a base frame (road background)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray
        
        # Add road markings
        cv2.line(frame, (0, height//2), (width, height//2), (255, 255, 255), 2)
        cv2.line(frame, (width//2, 0), (width//2, height), (255, 255, 255), 2)
        
        # Add random vehicle-like rectangles
        for _ in range(num_vehicles):
            x = np.random.randint(50, width - 100)
            y = np.random.randint(50, height - 80)
            w = np.random.randint(40, 80)
            h = np.random.randint(20, 40)
            
            # Random vehicle color
            color = (
                np.random.randint(100, 255),
                np.random.randint(100, 255),
                np.random.randint(100, 255)
            )
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
            
            # Add some noise/details
            cv2.rectangle(frame, (x + 5, y + 5), (x + w - 5, y + h - 5), 
                         (color[0] - 20, color[1] - 20, color[2] - 20), 2)
        
        return frame
    
    def process_video_stream(self, video_source: str, duration_seconds: int = 30) -> List[VehicleDetection]:
        """Process a video stream for a specified duration."""
        all_detections = []
        
        try:
            # For simulation, we'll generate synthetic frames
            # In reality, this would open cv2.VideoCapture(video_source)
            
            fps = 10  # Simulate 10 FPS
            total_frames = duration_seconds * fps
            
            for frame_num in range(total_frames):
                # Create synthetic frame
                frame = self.create_synthetic_frame()
                
                # Add some temporal variation
                if frame_num > 0:
                    # Add some noise to simulate motion
                    noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
                    frame = cv2.add(frame, noise)
                
                timestamp = datetime.now()
                detections = self.process_frame(frame, timestamp, 'north')
                all_detections.extend(detections)
            
            logger.info(f"Processed {total_frames} frames, found {len(all_detections)} detections")
            
        except Exception as e:
            logger.error(f"Error processing video stream: {e}")
        
        return all_detections
    
    def get_detection_statistics(self, detections: List[VehicleDetection]) -> Dict[str, any]:
        """Get statistics about detections."""
        if not detections:
            return {
                'total_detections': 0,
                'average_confidence': 0.0,
                'detections_by_type': {},
                'detections_by_lane': {}
            }
        
        detections_by_type = {}
        detections_by_lane = {}
        total_confidence = 0.0
        
        for detection in detections:
            # Count by type
            detections_by_type[detection.vehicle_type] = detections_by_type.get(detection.vehicle_type, 0) + 1
            
            # Count by lane
            detections_by_lane[detection.lane] = detections_by_lane.get(detection.lane, 0) + 1
            
            # Sum confidence
            total_confidence += detection.confidence
        
        return {
            'total_detections': len(detections),
            'average_confidence': total_confidence / len(detections),
            'detections_by_type': detections_by_type,
            'detections_by_lane': detections_by_lane
        }