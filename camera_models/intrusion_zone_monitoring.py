# camera_models/intrusion_zone_monitoring.py - BUG FIX FOR EVENT PROCESSING

import cv2
import numpy as np
import time
from utils import _draw_zone

try:
    from .camera_model_base import CameraModelBase
except ImportError:
    from camera_model_base import CameraModelBase

try:
    from utils import draw_text_with_background
except ImportError:
    def draw_text_with_background(img, text, position, color):
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

class IntrusionZoneMonitor(CameraModelBase):
    """
    Use Case: Intrusion Zone Monitoring
    BUG FIX VERSION - Fixed event processing error
    """

    def __init__(self, camera_id, zones=None, rules=None, settings=None, db=None, 
                 db_writer=None, frames_base_dir='outputs/frames', camera_manager=None):
        super().__init__(
            camera_id=camera_id,
            zones=zones,
            rules=rules,
            settings=settings,
            db=db,
            db_writer=db_writer,
            frames_base_dir=frames_base_dir,
            camera_manager=camera_manager
        )
        self.logger.info("Initializing IntrusionZoneMonitor - BUG FIX VERSION")

        # Get intrusion zones
        self.intrusion_zones = self._get_intrusion_zones()
        if not self.intrusion_zones:
            self.logger.warning("No intrusion zones defined, using default zone")
            self.intrusion_zones = [{
                'name': 'Restricted Access Zone',
                'coordinates': [[500, 200], [1200, 200], [1200, 800], [500, 800]]
            }]

        # Zone colors
        self.zone_colors = {"intrusion": (0, 0, 255)}  # Red for danger

        # State tracking
        self.intrusion_alerts = set()
        self.current_people_count = 0
        self.current_intrusion_count = 0
        self.enable_individual_events = True

        self.logger.info(f"IntrusionZoneMonitor initialized with {len(self.intrusion_zones)} zones")

    def _get_intrusion_zones(self):
        """Extract intrusion zones from configuration"""
        intrusion_zones = []
        try:
            if not self.zones or len(self.zones) == 0:
                self.logger.info(f"Loading zones from database for camera {self.camera_id}")
                self.zones = self.load_zones_from_database()
            
            if 'intrusion' in self.zones:
                intrusion_zones = self.zones['intrusion']
                self.logger.info(f"Found {len(intrusion_zones)} intrusion zones")
            else:
                self.logger.warning("No intrusion zones found in database")
        except Exception as e:
            self.logger.error(f"Error extracting intrusion zones: {e}")
            
        return intrusion_zones

    def set_individual_events_enabled(self, enabled: bool):
        """Enable/disable individual camera event triggering"""
        self.enable_individual_events = enabled

    def get_current_people_count(self):
        return self.current_people_count

    def get_current_intrusion_count(self):
        return self.current_intrusion_count

    def _draw_zones(self, frame):
        """Draw intrusion zones on frame"""
        try:
            for zone in self.intrusion_zones:
                if 'coordinates' in zone:
                    points = zone['coordinates']
                    zone_name = zone.get('name', 'Restricted Zone')
                    color = self.zone_colors.get('intrusion', (0, 0, 255))
                    
                    frame = _draw_zone(
                        frame=frame,
                        points=points,
                        color=color,
                        label=zone_name,
                        alpha=0.3
                    )
        except Exception as e:
            self.logger.error(f"Error drawing intrusion zones: {e}")
        
        return frame

    def detect_people(self, frame, detection_result):
        """
        FIXED: Intrusion detection with proper event data structure
        """
        people_detections = []
        
        # Extract people from shared detection
        if detection_result:
            try:
                for result in detection_result:
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        for box in result.boxes:
                            class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                            confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                            class_name = result.names.get(class_id, f"class_{class_id}")
                            
                            if class_name == 'person' and confidence > 0.3:
                                x1, y1, x2, y2 = map(float, box.xyxy[0])
                                center_x = (x1 + x2) / 2
                                center_y = (y1 + y2) / 2
                                
                                person_detection = {
                                    'bbox': [x1, y1, x2, y2],
                                    'center': (center_x, center_y),
                                    'confidence': confidence,
                                    'track_id': f"intruder_{len(people_detections) + 1}"
                                }
                                people_detections.append(person_detection)
                                
            except Exception as e:
                self.logger.error(f"Error parsing detection result: {e}")

        # Create annotated frame with zones
        annotated_frame = frame.copy()
        annotated_frame = self._draw_zones(annotated_frame)

        people_count = len(people_detections)
        self.current_people_count = people_count
        
        # FIXED: Create proper event data structure
        intrusion_count = 0
        # CHANGED: Return simple list instead of complex nested structure
        intrusion_detections = []
        
        if people_count > 0:
            self.logger.info(f"INTRUSION: {people_count} people detected - ALL ARE INTRUDERS!")
            
            for i, person in enumerate(people_detections):
                track_id = person['track_id']
                bbox = person['bbox']
                
                # FORCE INTRUSION for all detected people
                zone_name = 'Restricted Access Zone'
                
                # Create simple intrusion detection object
                intrusion_detection = {
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': person['confidence'],
                    'zone_name': zone_name,
                    'alert_level': 'CRITICAL',
                    'timestamp': time.time()
                }
                
                # Add to alerts tracking
                if track_id not in self.intrusion_alerts:
                    self.intrusion_alerts.add(track_id)
                
                intrusion_count += 1
                intrusion_detections.append(intrusion_detection)
                
                # Draw critical alert
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Thick red border
                
                # Alert labels
                alert_text = f"INTRUDER {track_id}!"
                draw_text_with_background(annotated_frame, alert_text, (x1, y1 - 30), color=(255, 255, 255))
                
                # Zone info
                info_text = f"Zone: {zone_name}"
                draw_text_with_background(annotated_frame, info_text, (x1, y1 - 10), color=(255, 255, 255))

        self.current_intrusion_count = intrusion_count

        # ALWAYS trigger events if intrusions detected
        if self.enable_individual_events and intrusion_detections:
            self.logger.warning(f"CRITICAL: {len(intrusion_detections)} INTRUSION ALERTS triggered!")
            self._handle_individual_camera_events(intrusion_detections, annotated_frame)

        # Add alert overlay to frame
        if intrusion_count > 0:
            alert_text = f"SECURITY ALERT: {intrusion_count} INTRUDERS DETECTED"
            cv2.rectangle(annotated_frame, (10, 10), (800, 60), (0, 0, 255), -1)
            cv2.putText(annotated_frame, alert_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # FIXED: Return simple list structure that matches other models
        return annotated_frame, people_count, intrusion_count, intrusion_detections

    def _handle_individual_camera_events(self, intrusion_events, annotated_frame):
        """Handle intrusion events - FIXED VERSION"""
        if intrusion_events:
            self.logger.warning(f"INTRUSION EVENT: {len(intrusion_events)} critical alerts!")
            self.stats['events_detected'] = self.stats.get('events_detected', 0) + len(intrusion_events)
            self.stats['intrusion_events'] = self.stats.get('intrusion_events', 0) + len(intrusion_events)
            
            for event in intrusion_events:
                track_id = event.get('track_id', 'unknown')
                zone_name = event.get('zone_name', 'unknown')
                self.logger.warning(f"  ALERT: {track_id} breached {zone_name}")
        else:
            self.logger.warning("Intrusion handler called but no events found!")

    def _process_frame_impl(self, frame, timestamp, detection_result):
        """Process frame for intrusion detection"""
        self.logger.debug(f"Processing intrusion frame at timestamp {timestamp}")
        
        annotated_frame, people_count, intrusion_count, intrusion_detections = self.detect_people(frame, detection_result)
        
        self.stats['frames_processed'] = self.stats.get('frames_processed', 0) + 1
        self.stats['people_detected'] = people_count
        
        # FIXED: Return detections in simple format expected by video processor
        return annotated_frame, intrusion_detections

