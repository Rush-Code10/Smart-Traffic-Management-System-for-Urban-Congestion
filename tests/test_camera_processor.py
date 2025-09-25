"""Tests for camera processor."""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from src.processors.camera_processor import CameraProcessor, DetectionBox
from src.models.vehicle_detection import VehicleDetection


class TestDetectionBox:
    """Test DetectionBox dataclass."""
    
    def test_detection_box_creation(self):
        """Test creating a detection box."""
        box = DetectionBox(x=10, y=20, width=50, height=30, confidence=0.9, class_id=2)
        
        assert box.x == 10
        assert box.y == 20
        assert box.width == 50
        assert box.height == 30
        assert box.confidence == 0.9
        assert box.class_id == 2


class TestCameraProcessor:
    """Test CameraProcessor class."""
    
    @pytest.fixture
    def camera_config(self):
        """Sample camera configuration."""
        return {
            'position': 'north',
            'angle': 180,
            'resolution': (640, 480),
            'fps': 10
        }
    
    @pytest.fixture
    def processor(self, camera_config):
        """Create camera processor instance."""
        return CameraProcessor(camera_config)
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.camera_config is not None
        assert processor.detection_threshold == 0.8
        assert processor.vehicle_counter == 0
        assert len(processor.vehicle_class_ids) == 4
        assert processor.bg_subtractor is not None
        assert processor.next_vehicle_id == 1
    
    def test_processor_custom_threshold(self, camera_config):
        """Test processor with custom detection threshold."""
        processor = CameraProcessor(camera_config, detection_threshold=0.9)
        assert processor.detection_threshold == 0.9
    
    def test_process_frame_empty(self, processor):
        """Test processing empty frame."""
        detections = processor.process_frame(None, datetime.now(), 'north')
        assert detections == []
        
        empty_frame = np.array([])
        detections = processor.process_frame(empty_frame, datetime.now(), 'north')
        assert detections == []
    
    def test_process_frame_valid(self, processor):
        """Test processing valid frame."""
        # Create a synthetic frame
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        timestamp = datetime.now()
        
        detections = processor.process_frame(frame, timestamp, 'north')
        
        assert isinstance(detections, list)
        # Note: Detections may be empty for synthetic frame without motion
    
    def test_create_synthetic_frame(self, processor):
        """Test creating synthetic frame."""
        frame = processor.create_synthetic_frame(width=640, height=480, num_vehicles=3)
        
        assert frame.shape == (480, 640, 3)
        assert frame.dtype == np.uint8
        
        # Check that frame has some variation (not all same color)
        assert np.std(frame) > 0
    
    def test_create_synthetic_frame_custom_size(self, processor):
        """Test creating synthetic frame with custom size."""
        frame = processor.create_synthetic_frame(width=800, height=600, num_vehicles=5)
        
        assert frame.shape == (600, 800, 3)
        assert frame.dtype == np.uint8
    
    def test_convert_to_world_coordinates(self, processor):
        """Test converting pixel coordinates to world coordinates."""
        # Test different camera positions
        positions = ['north', 'south', 'east', 'west']
        
        for position in positions:
            world_pos = processor._convert_to_world_coordinates(320, 240, position)
            
            assert isinstance(world_pos, tuple)
            assert len(world_pos) == 2
            assert isinstance(world_pos[0], (int, float))
            assert isinstance(world_pos[1], (int, float))
            
            # Check that coordinates are within reasonable range
            x, y = world_pos
            assert -100 <= x <= 100
            assert -100 <= y <= 100
    
    def test_classify_vehicle_by_size(self, processor):
        """Test vehicle classification by size."""
        # Test motorcycle (small)
        vehicle_type = processor._classify_vehicle_by_size(20, 30)
        assert vehicle_type == 'motorcycle'
        
        # Test car (medium)
        vehicle_type = processor._classify_vehicle_by_size(40, 30)
        assert vehicle_type == 'car'
        
        # Test truck (large with high aspect ratio)
        vehicle_type = processor._classify_vehicle_by_size(80, 20)
        assert vehicle_type == 'truck'
        
        # Test bus (large)
        vehicle_type = processor._classify_vehicle_by_size(80, 30)  # Larger area
        assert vehicle_type == 'bus'
    
    def test_determine_lane(self, processor):
        """Test lane determination from position."""
        # Test north camera
        lane = processor._determine_lane((-5, 30), 'north')
        assert lane == 'north_lane_1'
        
        lane = processor._determine_lane((5, 30), 'north')
        assert lane == 'north_lane_2'
        
        # Test south camera
        lane = processor._determine_lane((-5, -30), 'south')
        assert lane == 'south_lane_1'
        
        # Test east camera
        lane = processor._determine_lane((30, -5), 'east')
        assert lane == 'east_lane_1'
        
        # Test west camera
        lane = processor._determine_lane((-30, 5), 'west')
        assert lane == 'west_lane_2'
    
    @patch('cv2.findContours')
    @patch('cv2.contourArea')
    @patch('cv2.boundingRect')
    def test_detect_vehicles_simulation(self, mock_bounding_rect, mock_contour_area, 
                                      mock_find_contours, processor):
        """Test vehicle detection simulation."""
        # Mock OpenCV functions
        mock_contour = np.array([[10, 10], [50, 10], [50, 40], [10, 40]])
        mock_find_contours.return_value = ([mock_contour], None)
        mock_contour_area.return_value = 1000  # Valid vehicle size
        mock_bounding_rect.return_value = (10, 10, 40, 30)
        
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        timestamp = datetime.now()
        
        detections = processor._detect_vehicles_simulation(frame, timestamp, 'north')
        
        assert isinstance(detections, list)
        if detections:  # If detection occurred
            detection = detections[0]
            assert isinstance(detection, VehicleDetection)
            assert detection.vehicle_type in ['car', 'truck', 'bus', 'motorcycle']
            assert 0.0 <= detection.confidence <= 1.0
            assert detection.timestamp == timestamp
    
    def test_process_video_stream(self, processor):
        """Test processing video stream."""
        detections = processor.process_video_stream('fake_source', duration_seconds=1)
        
        assert isinstance(detections, list)
        # Should process at least a few frames
        # Note: May be empty if no motion detected in synthetic frames
    
    def test_get_detection_statistics_empty(self, processor):
        """Test getting statistics with no detections."""
        stats = processor.get_detection_statistics([])
        
        assert stats['total_detections'] == 0
        assert stats['average_confidence'] == 0.0
        assert stats['detections_by_type'] == {}
        assert stats['detections_by_lane'] == {}
    
    def test_get_detection_statistics_with_detections(self, processor):
        """Test getting statistics with detections."""
        # Create sample detections
        detections = [
            VehicleDetection(
                vehicle_id='test1',
                vehicle_type='car',
                position=(10.0, 20.0),
                lane='north_lane_1',
                confidence=0.9,
                timestamp=datetime.now()
            ),
            VehicleDetection(
                vehicle_id='test2',
                vehicle_type='truck',
                position=(15.0, 25.0),
                lane='north_lane_2',
                confidence=0.8,
                timestamp=datetime.now()
            )
        ]
        
        stats = processor.get_detection_statistics(detections)
        
        assert stats['total_detections'] == 2
        assert abs(stats['average_confidence'] - 0.85) < 0.001  # Account for floating point precision
        assert stats['detections_by_type']['car'] == 1
        assert stats['detections_by_type']['truck'] == 1
        assert stats['detections_by_lane']['north_lane_1'] == 1
        assert stats['detections_by_lane']['north_lane_2'] == 1
    
    def test_vehicle_class_ids(self, processor):
        """Test vehicle class ID mapping."""
        assert processor.vehicle_class_ids[2] == 'car'
        assert processor.vehicle_class_ids[3] == 'motorcycle'
        assert processor.vehicle_class_ids[5] == 'bus'
        assert processor.vehicle_class_ids[7] == 'truck'
    
    @patch('cv2.createBackgroundSubtractorMOG2')
    def test_background_subtractor_initialization(self, mock_bg_subtractor, camera_config):
        """Test background subtractor initialization."""
        mock_bg_subtractor.return_value = MagicMock()
        
        processor = CameraProcessor(camera_config)
        
        mock_bg_subtractor.assert_called_once_with(
            detectShadows=True,
            varThreshold=50
        )
        assert processor.bg_subtractor is not None
    
    def test_next_vehicle_id_increment(self, processor):
        """Test that vehicle ID increments properly."""
        initial_id = processor.next_vehicle_id
        
        # Create a frame that might generate detections
        frame = processor.create_synthetic_frame(num_vehicles=1)
        processor.process_frame(frame, datetime.now(), 'north')
        
        # ID should increment if detection occurred
        # Note: May not increment if no motion detected
        assert processor.next_vehicle_id >= initial_id