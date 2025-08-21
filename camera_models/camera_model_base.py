# camera_models/camera_model_base.py
# Copy this to: single_camera_test/camera_models/camera_model_base.py

import cv2
import numpy as np
import os
import uuid
import time
import threading
from datetime import datetime

# Import with fallback for single camera setup
try:
    from camera_models.kalman_track import Sort
except ImportError:
    from kalman_track import Sort

try:
    from logger import setup_datacenter_logger
except ImportError:
    import logging
    def setup_datacenter_logger(name, filename):
        return logging.getLogger(name)

try:
    from config import Config
except ImportError:
    # Fallback config for testing
    class Config:
        DETECTION_CONFIDENCE_THRESHOLD = 0.5
        FRAMES_OUTPUT_DIR = 'outputs/frames'

# For single camera testing - simplified database interface
class SimpleDatabaseInterface:
    """Simplified database interface for single camera testing"""
    def execute_query(self, query, params=None):
        return []

class DatacenterCameraModelBase:
    """
    Base class for all datacenter camera-specific models.
    Modified for single camera testing setup.
    """
    
    def __init__(self, camera_id, zones=None, rules=None, settings=None, db=None, 
                 db_writer=None, frames_base_dir='outputs/frames', camera_manager=None):
        """
        Initialize the datacenter camera model base class
        """
        # Initialize logger
        try:
            self.logger = setup_datacenter_logger(f'camera_model_{camera_id}', f'camera_model_{camera_id}.log')
        except:
            import logging
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(f'camera_model_{camera_id}')
        
        self.logger.info(f"Initializing camera model base for camera {camera_id}")
        
        # Store parameters
        self.camera_id = camera_id
        self.zones = zones or {}
        self.rules = rules or {}
        self.settings = settings or {}
        self.camera_manager = camera_manager
        
        # For single camera testing
        self.datacenter_id = 1  # Default datacenter ID
        
        # Initialize database interface
        self.db = db if db is not None else SimpleDatabaseInterface()
        
        # Storage settings
        self.db_writer = db_writer
        self.frames_base_dir = frames_base_dir
        
        # Create directory for this camera
        self.camera_output_dir = os.path.join(frames_base_dir, f"camera_{camera_id}")
        os.makedirs(self.camera_output_dir, exist_ok=True)
        
        # Initialize object tracking
        try:
            self.object_tracker = Sort(max_age=30, min_hits=3)
        except:
            # Fallback if Sort import fails
            self.object_tracker = None
        
        # Tracking state
        self.tracked_objects = {}
        self.total_object_count = 0
        
        # Performance statistics
        self.stats = {
            'frames_processed': 0,
            'events_detected': 0,
            'videos_saved': 0,
            'start_time': time.time(),
            'last_processed_time': None
        }
        
        # Event recording settings
        self.auto_recording_enabled = getattr(Config, 'AUTO_RECORDING_ENABLED', True)
        self.event_cooldown = getattr(Config, 'EVENT_COOLDOWN', 30)
        self.media_preference = getattr(Config, 'MEDIA_PREFERENCE', 'both')
        
        # Zone colors for visualization
        self.zone_colors = {
            'counting': (0, 255, 255),     # Yellow
            'ppe_zone': (255, 165, 0),     # Orange
            'entry': (0, 255, 255),        # Cyan
            'intrusion': (0, 0, 255),      # Red
            'loitering': (255, 0, 255),    # Magenta
            'unauthorized': (255, 0, 0)    # Red
        }
        
        self.logger.info(f"Camera model base initialized for camera {camera_id}")
    
    def _get_datacenter_id(self):
        """Get datacenter_id for this camera"""
        return 1  # Default for single camera testing
    
    def load_zones_from_database(self):
        """Load zones from database - simplified for single camera"""
        try:
            # For single camera testing, return empty zones if none provided
            if not self.zones:
                self.logger.warning(f"No zones found for camera {self.camera_id}")
                return {}
            return self.zones
        except Exception as e:
            self.logger.error(f"Error loading zones: {e}")
            return {}
        
    def process_frame(self, frame, timestamp, detection_result=None):
        """
        Process a frame with detection results.
        This is the main entry point called from the video processor.
        """
        # Update statistics
        self.stats['frames_processed'] += 1
        self.stats['last_processed_time'] = timestamp
        
        # Perform custom processing in subclasses
        annotated_frame, detections = self._process_frame_impl(frame, timestamp, detection_result)
        
        return annotated_frame, detections
    
    def _process_frame_impl(self, frame, timestamp, detection_result):
        """
        Implement frame processing in subclasses.
        This should be overridden by each camera model.
        """
        # Default implementation - should be overridden
        return frame, []
    
    def update_tracker(self, detection_array):
        """Update object tracker with new detections"""
        try:
            if self.object_tracker is None:
                return []
            
            if len(detection_array) > 0:
                tracked_objects_array, object_count = self.object_tracker.update(detection_array)
            else:
                tracked_objects_array, object_count = self.object_tracker.update()
            
            self.total_object_count = object_count
            
            # Convert tracker output to list of dictionaries
            updated_detections = []
            
            for i in range(tracked_objects_array.shape[0]):
                track = tracked_objects_array[i]
                x, y, aspect_ratio, height, track_id = track
                track_id = int(track_id)
                
                # Convert back to bbox format
                width = aspect_ratio * height
                x1 = int(x - width/2)
                y1 = int(y - height/2)
                x2 = int(x + width/2)
                y2 = int(y + height/2)
                
                detection_dict = {
                    'track_id': track_id,
                    'bbox': [x1, y1, x2, y2],
                    'center': (x, y),
                    'confidence': 0.5  # Default confidence
                }
                
                updated_detections.append(detection_dict)
                self.tracked_objects[track_id] = detection_dict
            
            return updated_detections
            
        except Exception as e:
            self.logger.error(f"Error updating tracker: {e}")
            return []
    
    def is_in_zone(self, point, zone):
        """
        Check if a point is inside a zone
        """
        try:
            if isinstance(zone, dict) and 'coordinates' in zone:
                polygon = np.array(zone['coordinates'])
            elif isinstance(zone, list):
                polygon = np.array(zone)
            else:
                return False
            
            # Use OpenCV point in polygon test
            result = cv2.pointPolygonTest(polygon.astype(np.int32), point, False)
            return result >= 0
            
        except Exception as e:
            self.logger.error(f"Error checking point in zone: {e}")
            return False
    
    def calculate_zone_overlap(self, bbox, zone):
        """
        Calculate overlap ratio between bounding box and zone
        """
        try:
            x1, y1, x2, y2 = bbox
            
            # Create bounding box mask
            width = x2 - x1
            height = y2 - y1
            bbox_mask = np.zeros((height, width), dtype=np.uint8)
            bbox_mask.fill(255)
            
            # Get zone polygon
            if isinstance(zone, dict) and 'coordinates' in zone:
                polygon = np.array(zone['coordinates'])
            elif isinstance(zone, list):
                polygon = np.array(zone)
            else:
                return 0.0
            
            # Adjust polygon coordinates relative to bbox
            min_x, min_y = x1, y1
            polygon_adj = polygon - np.array([min_x, min_y])
            
            # Create zone mask
            zone_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(zone_mask, [polygon_adj.astype(np.int32)], 255)
            
            # Calculate overlap
            overlap = cv2.bitwise_and(bbox_mask, zone_mask)
            overlap_area = cv2.countNonZero(overlap)
            bbox_area = cv2.countNonZero(bbox_mask)
            
            if bbox_area == 0:
                return 0.0
                
            return overlap_area / bbox_area
            
        except Exception as e:
            self.logger.error(f"Error calculating zone overlap: {e}")
            return 0.0
    
    def get_current_people_count(self):
        """Get current number of people detected by this camera"""
        return len(self.tracked_objects)
    
    def get_stats(self):
        """Get statistics for this camera model"""
        uptime = time.time() - self.stats.get('start_time', time.time())
        frames_processed = self.stats.get('frames_processed', 0)
        fps = frames_processed / max(1, uptime)
        
        stats = {
            'camera_id': self.camera_id,
            'datacenter_id': self.datacenter_id,
            'camera_type': self.__class__.__name__,
            'frames_processed': frames_processed,
            'events_detected': self.stats.get('events_detected', 0),
            'tracked_objects': len(self.tracked_objects),
            'total_object_count': self.total_object_count,
            'uptime_seconds': uptime,
            'fps': fps,
            'last_processed': self.stats.get('last_processed_time')
        }
        
        stats.update(self.stats)
        return stats

# Legacy compatibility
CameraModelBase = DatacenterCameraModelBase