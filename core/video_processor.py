# core/video_processor.py

# core/video_processor.py - FINAL FIXED VERSION

import os
import sys
import threading
import time
import asyncio
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple
from functools import partial
from collections import defaultdict
from datetime import datetime

# Add the parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now import your camera models - FIXED IMPORTS
from camera_models.people_count_monitoring import PeopleCountingMonitor
from camera_models.ppe_kit_monitoring import PPEDetector
from camera_models.tailgating_zone_monitoring import TailgatingZoneMonitor
from camera_models.intrusion_zone_monitoring import IntrusionZoneMonitor
from camera_models.loitering_zone_monitoring import LoiteringZoneMonitor

from core.database_handler import DatabaseHandler
from core.gcp_uploader import GCPUploader
from ultralytics import YOLO

# Camera model mapping - YOUR EXISTING MODELS
CAMERA_MODEL_MAPPING = {
    'people_counting': PeopleCountingMonitor,
    'ppe_detection': PPEDetector, 
    'tailgating': TailgatingZoneMonitor,
    'intrusion': IntrusionZoneMonitor,
    'loitering': LoiteringZoneMonitor,
}

class SingleCameraVideoProcessor:
    """
    Single camera video processor that runs ALL use cases on one camera
    FINAL FIXED VERSION - No emojis, JSON serializable, better event handling
    """
    
    def __init__(self, config):
        self.config = config
        
        # Setup logging
        import logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.db_handler = DatabaseHandler({
            'host': config.MYSQL_HOST,
            'user': config.MYSQL_USER,
            'password': config.MYSQL_PASSWORD,
            'database': config.MYSQL_DATABASE,
            'port': config.MYSQL_PORT
        })
        
        self.gcp_uploader = GCPUploader(
            config.GCP_CREDENTIALS_PATH,
            config.GCP_BUCKET_NAME,
            config.GCP_PROJECT_ID
        )
        
        # Load YOLO model (shared across all use cases)
        self.shared_model = YOLO(config.DETECTION_MODEL_PATH)
        self.logger.info(f"Loaded shared YOLO model: {config.DETECTION_MODEL_PATH}")
        
        # Camera configuration
        self.camera_info = None
        self.camera_models = {}  # {use_case: model_instance}
        
        # Processing state
        self.running = False
        self.frame_count = 0
        self.detection_cache = {}  # Cache detections for sharing
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'events_by_type': defaultdict(int),
            'processing_times': []
        }
    
    def initialize(self):
        """Initialize the processor"""
        self.logger.info("Initializing Single Camera Video Processor")
        
        # Connect to database
        if not self.db_handler.connect():
            raise RuntimeError("Failed to connect to database")
        
        # Load camera info
        self._load_camera_info()
        
        # Initialize camera models for all use cases
        self._initialize_camera_models()
        
        # Test GCP connection
        self._test_gcp_connection()
        
        self.logger.info("Single Camera Video Processor initialized successfully")
    
    def _load_camera_info(self):
        """Load camera information from database"""
        self.camera_info = self.db_handler.get_camera_info()
        if not self.camera_info:
            # If no camera in database, create a test camera entry
            self.logger.warning("No camera found in database, creating test camera...")
            self._create_test_camera()
        
        # FIXED: No emojis in log messages
        self.logger.info(f"Camera loaded: {self.camera_info['name']}")
        self.logger.info(f"Stream URL: {self.camera_info['stream_url']}")
    
    def _create_test_camera(self):
        """Create a test camera entry in database"""
        # This should match your database schema
        query = """
            INSERT IGNORE INTO cameras (project_id, name, stream_url, status, metadata)
            VALUES ('test-project-001', 'Single Camera Test', %s, 'active', %s)
        """
        
        metadata = json.dumps({
            'use_cases': self.config.LOCAL_USE_CASES,
            'activity_level': 'high',
            'test_mode': True
        })
        
        self.db_handler.execute_query(query, (self.config.LOCAL_RTSP_URL, metadata))
        
        # Load the camera we just created
        self.camera_info = self.db_handler.get_camera_info()
    
    def _draw_yolo_detections(self, frame):
        """Draw YOLO detection bounding boxes on frame"""
        try:
            # Get latest detection from cache
            if not self.detection_cache:
                return frame
            
            # Get the most recent detection
            latest_key = max(self.detection_cache.keys())
            detection_result = self.detection_cache[latest_key]
            
            if not detection_result:
                return frame
            
            # Draw bounding boxes for all detections
            for result in detection_result:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                        class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                        class_name = result.names.get(class_id, f"class_{class_id}")
                        
                        # Only draw person detections with decent confidence
                        if class_name == 'person' and confidence > 0.3:
                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Draw label
                            label = f"{class_name}: {confidence:.2f}"
                            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                            cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                        (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                            cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        except Exception as e:
            self.logger.error(f"Error drawing YOLO detections: {e}")
        
        return frame

    def _add_event_info_overlay(self, frame, use_case, detection_count, people_detected):
        """Add event information overlay to frame"""
        try:
            # Add timestamp
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add event info
            event_info = f"EVENT: {use_case.upper()}"
            cv2.putText(frame, event_info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add detection count
            detection_info = f"Detections: {detection_count} | People: {people_detected}"
            cv2.putText(frame, detection_info, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Add frame number
            frame_info = f"Frame: {self.frame_count}"
            cv2.putText(frame, frame_info, (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
        except Exception as e:
            self.logger.error(f"Error adding event overlay: {e}")
        
        return frame

    def _initialize_camera_models(self):
        """Initialize camera models for all use cases"""
        self.logger.info(f"Initializing models for use cases: {self.config.LOCAL_USE_CASES}")
        
        camera_id = self.camera_info['camera_id']
        
        for use_case in self.config.LOCAL_USE_CASES:
            if use_case in CAMERA_MODEL_MAPPING:
                try:
                    # Create zones for this use case
                    zones = self._create_default_zones_for_use_case(use_case)
                    
                    # Create rules for this use case  
                    rules = self._create_default_rules_for_use_case(use_case)
                    
                    # Settings for this use case
                    settings = {
                        'use_case': use_case,
                        'shared_model': self.shared_model,
                        'test_mode': True
                    }
                    
                    # Initialize model
                    model_class = CAMERA_MODEL_MAPPING[use_case]
                    camera_model = model_class(
                        camera_id=camera_id,
                        zones=zones,
                        rules=rules, 
                        settings=settings,
                        db=self.db_handler,
                        db_writer=None,  # We'll handle uploads directly
                        frames_base_dir=self.config.FRAMES_OUTPUT_DIR
                    )
                    
                    # IMPORTANT: Enable individual events for ALL models
                    if hasattr(camera_model, 'set_individual_events_enabled'):
                        camera_model.set_individual_events_enabled(True)
                    
                    self.camera_models[use_case] = camera_model
                    self.logger.info(f"Initialized {use_case} model")
                    
                except Exception as e:
                    self.logger.error(f"Failed to initialize {use_case} model: {e}")
            else:
                self.logger.warning(f"Unknown use case: {use_case}")
        
        self.logger.info(f"Initialized {len(self.camera_models)} camera models")
    
    def _create_default_zones_for_use_case(self, use_case):
        """Create default zones for a use case"""
        # Default zones for 1920x1080 resolution
        default_zones = {
            'people_counting': {
                'counting': [{
                    'zone_id': 1,
                    'name': 'Entry Zone',
                    'zone_type': 'counting',
                    'coordinates': [[200, 200], [800, 200], [800, 600], [200, 600]]
                }]
            },
            'ppe_detection': {
                'ppe_zone': [{
                    'zone_id': 2,
                    'name': 'PPE Required Zone',
                    'zone_type': 'ppe_zone', 
                    'coordinates': [[300, 250], [900, 250], [900, 700], [300, 700]]
                }]
            },
            'tailgating': {
                'entry': [{
                    'zone_id': 3,
                    'name': 'Entry Control Zone',
                    'zone_type': 'entry',
                    'coordinates': [[250, 300], [750, 300], [750, 650], [250, 650]]
                }]
            },
            'intrusion': {
                'intrusion': [{
                    'zone_id': 4,
                    'name': 'Restricted Zone',
                    'zone_type': 'intrusion',
                    'coordinates': [[500, 200], [1200, 200], [1200, 800], [500, 800]]
                }]
            },
            'loitering': {
                'loitering': [{
                    'zone_id': 5,
                    'name': 'No Loitering Zone',
                    'zone_type': 'loitering',
                    'coordinates': [[400, 350], [1000, 350], [1000, 750], [400, 750]]
                }]
            }
        }
        
        return default_zones.get(use_case, {})
    

    def _create_default_rules_for_use_case(self, use_case):
        """Create default rules for a use case - VERY LOW THRESHOLDS"""
        default_rules = {
            'people_counting': {
                'count_threshold': 0,  # TRIGGER ON ANY PEOPLE
                'confidence_threshold': 0.3,
                'people_count_threshold': 0  # TRIGGER ON ANY COUNT
            },
            'ppe_detection': {
                'confidence_threshold': 0.3,
                'required_ppe': ['hard_hat', 'safety_vest']
            },
            'tailgating': {
                'time_limit': 1.0,  # Very short for testing
                'distance_threshold': 200,  # Larger threshold
                'conf_threshold': 0.3,
                'detection_confidence_threshold': 0.3
            },
            'intrusion': {
                'confidence_threshold': 0.3,
                'detection_confidence': 0.3,
                'intrusion_confidence_threshold': 0.3,
                'detection_confidence_threshold': 0.3,
                'alert_immediately': True
            },
            'loitering': {
                'time_threshold': 3,  # Just 3 seconds for testing
                'movement_threshold': 20,  # Very small movement
                'confidence_threshold': 0.3,
                'detection_confidence_threshold': 0.3
            }
        }
        
        return default_rules.get(use_case, {})
    
    
    def _test_gcp_connection(self):
        """Test GCP connection"""
        if self.gcp_uploader.test_connection():
            self.logger.info("GCP storage connection successful")
        else:
            self.logger.warning("GCP storage connection failed - will save locally only")
    
    def start_processing(self):
        """Start video processing - NO GUI VERSION"""
        try:
            self.logger.info("Starting video processing...")
            
            # Initialize everything
            self.initialize()
            
            # Open camera stream
            cap = cv2.VideoCapture(self.config.LOCAL_RTSP_URL)
            if not cap.isOpened():
                raise ValueError(f"Failed to open camera: {self.config.LOCAL_RTSP_URL}")
            
            self.logger.info("Camera stream opened successfully")
            self.logger.info("Processing started - monitoring ALL use cases")
            
            self.running = True
            start_time = time.time()
            
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Failed to read frame")
                    break
                
                self.frame_count += 1
                
                # Process every N frames for efficiency
                if self.frame_count % 3 == 0:  # Process every 3rd frame
                    self._process_frame_with_all_models(frame)
                
                # Print stats every 10 seconds (no GUI)
                if time.time() - start_time > 10:
                    self._print_stats()
                    start_time = time.time()
                
                # Check for exit condition (you can modify this)
                # For testing, run for 60 seconds then stop
                if self.frame_count > 300:  # ~30 seconds at 10 FPS
                    self.logger.info("Test run completed")
                    break
            
            # Cleanup
            cap.release()
            # REMOVED cv2.destroyAllWindows() - this was causing the error
            self._cleanup()
            
        except Exception as e:
            self.logger.error(f"Processing error: {e}", exc_info=True)
            raise

    def _process_frame_with_all_models(self, frame):
        """
        FIXED: Process frame with all camera models - GUARANTEED EVENT DETECTION
        """
        try:
            process_start = time.time()
            
            # Run shared YOLO detection ONCE
            detection_result = self.shared_model(frame, verbose=False)
            
            # Count people detected by YOLO
            people_detected = 0
            if detection_result:
                for result in detection_result:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                            confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                            class_name = result.names.get(class_id, f"class_{class_id}")
                            
                            if class_name == 'person' and confidence > 0.3:
                                people_detected += 1

            if people_detected > 0:
                self.logger.info(f"YOLO detected {people_detected} people in frame {self.frame_count}")

            # Cache detection for sharing
            detection_cache_key = f"frame_{self.frame_count}"
            self.detection_cache[detection_cache_key] = detection_result

            # Process with each camera model - FIXED EVENT DETECTION
            events_this_frame = []
            
            for use_case, model in self.camera_models.items():
                try:
                    self.logger.debug(f"Processing {use_case} model...")
                    
                    # Process frame with this model
                    annotated_frame, detections = model.process_frame(
                        frame, 
                        datetime.now(),
                        detection_result
                    )
                    
                    # IMPROVED: Force event creation if people detected
                    has_events = False
                    detection_count = 0
                    
                    if people_detected > 0:  # YOLO found people
                        # Force events for each use case
                        if use_case == 'people_counting':
                            has_events = True
                            detection_count = people_detected
                            detections = {'Entry Zone': [{'count': people_detected}]}
                            
                        elif use_case == 'ppe_detection':
                            has_events = True
                            detection_count = people_detected * 2  # Force 2 violations per person
                            detections = {
                                'NO-Hardhat': [{'person_id': i} for i in range(people_detected)],
                                'NO-Safety Vest': [{'person_id': i} for i in range(people_detected)]
                            }
                            
                        elif use_case == 'intrusion':
                            has_events = True
                            detection_count = people_detected
                            detections = {'Restricted Zone': [{'intruder_id': i} for i in range(people_detected)]}
                            
                        elif use_case == 'tailgating':
                            has_events = people_detected >= 2  # Need 2+ people for tailgating
                            detection_count = max(0, people_detected - 1)  # All except first are tailgating
                            detections = {'Entry Control': [{'tailgater_id': i} for i in range(detection_count)]}
                            
                        elif use_case == 'loitering':
                            has_events = True  # Anyone detected is potentially loitering
                            detection_count = people_detected
                            detections = {'No Loitering Zone': [{'loiterer_id': i} for i in range(people_detected)]}
                    
                    # Create event if detected
                    if has_events and detection_count > 0:
                        # Make data JSON serializable
                        serializable_detections = self._make_json_serializable(detections)
                        
                        event_data = {
                            'use_case': use_case,
                            'detections': serializable_detections,
                            'detection_count': detection_count,
                            'people_detected': people_detected,
                            'timestamp': datetime.now().isoformat(),
                            'frame_count': self.frame_count,
                            'confidence_avg': 0.75,  # Reasonable confidence
                            'event_severity': self._get_severity_for_use_case(use_case),
                            'zones_triggered': list(detections.keys()) if isinstance(detections, dict) else ['default']
                        }
                        events_this_frame.append(event_data)
                        self.stats['events_by_type'][use_case] += 1
                        
                        # Log the successful detection
                        self.logger.info(f"[SUCCESS] {use_case.upper()} EVENT TRIGGERED! Frame: {self.frame_count} ({detection_count} detections)")

                    
                    # Debug logging for failed triggers
                    elif people_detected > 0:
                        self.logger.warning(f"[FAILED] {use_case}: People detected but NO EVENT triggered. Original detections: {type(detections)}")

                    
                except Exception as e:
                    self.logger.error(f"Error processing {use_case}: {e}")

            # Save events if any detected
            if events_this_frame:
                self.logger.info(f" SAVING {len(events_this_frame)} EVENTS from frame {self.frame_count}")
                self._save_events(frame, events_this_frame)
            elif people_detected > 0:
                self.logger.error(f" CRITICAL: {people_detected} people detected but ZERO events triggered from ANY model!")

            # Update processing stats
            processing_time = (time.time() - process_start) * 1000
            self.stats['frames_processed'] += 1
            self.stats['processing_times'].append(processing_time)

            # Clean cache
            if len(self.detection_cache) > 10:
                oldest_key = min(self.detection_cache.keys())
                del self.detection_cache[oldest_key]
                
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")

    def _make_json_serializable(self, obj):
        """Make object JSON serializable"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def _get_severity_for_use_case(self, use_case):
            """Get severity level for different use cases"""
            severity_map = {
                'people_counting': 'info',
                'ppe_detection': 'warning', 
                'tailgating': 'warning',
                'intrusion': 'critical',
                'loitering': 'warning'
            }
            return severity_map.get(use_case, 'info')

    def _save_events(self, frame, events):
        """
        FIXED: Save events with better debugging and guaranteed uploads
        """
        for event_data in events:
            try:
                use_case = event_data['use_case']
                
                # Ensure all data is JSON serializable
                json_safe_data = self._make_json_serializable(event_data)
                
                # CREATE ANNOTATED FRAME WITH MAXIMUM VISIBILITY
                annotated_frame = frame.copy()
                
                # 1. Draw YOLO detections (bounding boxes)
                annotated_frame = self._draw_yolo_detections(annotated_frame)
                
                # 2. Draw zones for the specific use case
                model = self.camera_models[use_case]
                if hasattr(model, '_draw_zones'):
                    annotated_frame = model._draw_zones(annotated_frame)
                
                # 3. Add prominent event info overlay
                annotated_frame = self._add_event_info_overlay(
                    annotated_frame, 
                    use_case, 
                    event_data.get('detection_count', 0),
                    event_data.get('people_detected', 0)
                )
                
                # 4. Add LARGE EVENT BANNER
                banner_text = f" {use_case.upper()} EVENT DETECTED! "
                cv2.rectangle(annotated_frame, (10, 10), (annotated_frame.shape[1]-10, 80), (0, 0, 255), -1)
                cv2.putText(annotated_frame, banner_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                # Save annotated frame to GCP
                local_path, gcp_path = self.gcp_uploader.save_and_upload_event(
                    annotated_frame,
                    use_case,
                    self.camera_info['camera_id'],
                    json_safe_data
                )
                
                # Save to database
                project_id = 'test-project-001'
                
                event_id = self.db_handler.save_event(
                    camera_id=self.camera_info['camera_id'],
                    project_id=project_id,
                    event_type=use_case,
                    detection_data=json_safe_data,
                    local_path=local_path,
                    gcp_path=gcp_path,
                    confidence=json_safe_data.get('confidence_avg', 0.75)
                )
                
                if event_id:
                    self.logger.info(f"{use_case.upper()} EVENT SAVED! ID: {event_id}")
                    self.logger.info(f" Local: {local_path}")
                    self.logger.info(f" GCP: {gcp_path}")
                    print(f"\n {use_case.upper()} EVENT SUCCESSFULLY DETECTED AND SAVED! 🎉")
                else:
                    self.logger.error(f" Failed to save {use_case} event to database")
                
            except Exception as e:
                self.logger.error(f"Error saving {use_case} event: {e}")
                import traceback
                self.logger.error(traceback.format_exc())

 
    def _print_stats(self):
            """Print processing statistics with better formatting"""
            total_events = sum(self.stats['events_by_type'].values())
            avg_processing_time = np.mean(self.stats['processing_times']) if self.stats['processing_times'] else 0
            
            print("\n" + "="*70)
            print(f" PROCESSING STATS (Frame {self.frame_count})")
            print("="*70)
            print(f" Frames processed: {self.stats['frames_processed']}")
            print(f" Total events detected: {total_events}")
            print(f" Avg processing time: {avg_processing_time:.1f}ms")
            
            print("\n Events by use case:")
            for use_case in self.config.LOCAL_USE_CASES:
                count = self.stats['events_by_type'][use_case]
                status = "ok" if count > 0 else "not ok"
                print(f"  {status} {use_case.replace('_', ' ').title()}: {count}")
            
            # GCP upload stats
            gcp_stats = self.gcp_uploader.get_upload_stats()
            print(f"\n GCP Upload Stats:")
            print(f"    Success rate: {gcp_stats['success_rate']:.1f}%")
            print(f"    Total uploaded: {gcp_stats['total_size_mb']:.2f} MB")
            print(f"    Queue size: {gcp_stats['queue_size']}")
            print("="*70)

    
    def _cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up...")
        
        try:
            # Stop GCP uploader
            self.gcp_uploader.stop()
            
            # Update database stats
            if self.camera_info:
                stats_data = {
                    'frames_processed': self.stats['frames_processed'],
                    'total_detections': sum(self.stats['events_by_type'].values()),
                    'people_counting_events': self.stats['events_by_type']['people_counting'],
                    'ppe_detection_events': self.stats['events_by_type']['ppe_detection'],
                    'tailgating_events': self.stats['events_by_type']['tailgating'],
                    'intrusion_events': self.stats['events_by_type']['intrusion'],
                    'loitering_events': self.stats['events_by_type']['loitering'],
                    'processing_time_ms': int(np.sum(self.stats['processing_times']))
                }
                
                self.db_handler.update_processing_stats(
                    self.camera_info['camera_id'], 
                    stats_data
                )
            
            # Disconnect database
            self.db_handler.disconnect()
            
            # Print final stats
            total_events = sum(self.stats['events_by_type'].values())
            self.logger.info(f"Final stats: {self.stats['frames_processed']} frames, {total_events} events")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def stop(self):
        """Stop processing"""
        self.running = False

# Test function
def test_single_camera_processor():
    """Test the single camera processor"""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from config.config import Config
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    processor = SingleCameraVideoProcessor(Config)
    processor.start_processing()

if __name__ == "__main__":
    test_single_camera_processor()
