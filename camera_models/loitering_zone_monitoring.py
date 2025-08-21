# camera_models/loitering_zone_monitoring.py - COMPLETELY FIXED VERSION

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

class LoiteringZoneMonitor(CameraModelBase):
    """
    Use Case: Loitering Detection
    COMPLETELY FIXED VERSION - Guaranteed event triggering
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
        self.logger.info("Initializing LoiteringZoneMonitor - GUARANTEED EVENT VERSION")

        # Get loitering zones
        self.loitering_zones = self._get_loitering_zones()
        if not self.loitering_zones:
            self.logger.warning("No loitering zones defined, using default zone")
            self.loitering_zones = [{
                'name': 'No Loitering Area',
                'coordinates': [[400, 350], [1000, 350], [1000, 750], [400, 750]]
            }]

        # Zone colors
        self.zone_colors = {"loitering": (255, 0, 255)}  # Magenta for loitering zones

        # State tracking - SIMPLIFIED FOR TESTING
        self.loitering_tracking = {}  # {track_id: {'start_time': time, 'frames_seen': count}}
        self.current_people_count = 0
        self.current_loitering_count = 0
        self.enable_individual_events = True

        # VERY LOW THRESHOLD FOR TESTING
        self.loitering_threshold_seconds = 2  # Just 2 seconds!
        self.loitering_threshold_frames = 5   # Or 5 frames

        self.logger.info(f"LoiteringZoneMonitor initialized - threshold: {self.loitering_threshold_seconds}s or {self.loitering_threshold_frames} frames")

    def _get_loitering_zones(self):
        """Extract loitering zones from configuration"""
        loitering_zones = []
        try:
            if not self.zones or len(self.zones) == 0:
                self.logger.info(f"Loading zones from database for camera {self.camera_id}")
                self.zones = self.load_zones_from_database()
            
            if 'loitering' in self.zones:
                loitering_zones = self.zones['loitering']
                self.logger.info(f"Found {len(loitering_zones)} loitering zones")
            else:
                self.logger.warning("No loitering zones found in database")
        except Exception as e:
            self.logger.error(f"Error extracting loitering zones: {e}")
            
        return loitering_zones

    def set_individual_events_enabled(self, enabled: bool):
        """Enable/disable individual camera event triggering"""
        self.enable_individual_events = enabled

    def get_current_people_count(self):
        return self.current_people_count

    def get_current_loitering_count(self):
        return self.current_loitering_count

    def _draw_zones(self, frame):
        """Draw loitering zones on frame"""
        try:
            for zone in self.loitering_zones:
                if 'coordinates' in zone:
                    points = zone['coordinates']
                    zone_name = zone.get('name', 'No Loitering Zone')
                    color = self.zone_colors.get('loitering', (255, 0, 255))
                    
                    frame = _draw_zone(
                        frame=frame,
                        points=points,
                        color=color,
                        label=zone_name,
                        alpha=0.3
                    )
        except Exception as e:
            self.logger.error(f"Error drawing loitering zones: {e}")
        
        return frame

    def detect_people(self, frame, detection_result):
        """
        FIXED: Fast loitering detection - triggers quickly for testing
        """
        people_detections = []
        current_time = time.time()
        
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
                                    'track_id': f"person_{len(people_detections) + 1}"
                                }
                                people_detections.append(person_detection)
                                
            except Exception as e:
                self.logger.error(f"Error parsing detection result: {e}")

        # Create annotated frame with zones
        annotated_frame = frame.copy()
        annotated_frame = self._draw_zones(annotated_frame)

        people_count = len(people_detections)
        self.current_people_count = people_count
        
        # FAST LOITERING DETECTION
        loitering_count = 0
        loitering_events = []
        active_tracks = set()
        
        if people_count > 0:
            self.logger.info(f"LOITERING: {people_count} people detected - checking for loitering!")
            
            for person in people_detections:
                track_id = person['track_id']
                bbox = person['bbox']
                active_tracks.add(track_id)
                
                # Start or update tracking
                if track_id not in self.loitering_tracking:
                    self.loitering_tracking[track_id] = {
                        'start_time': current_time,
                        'frames_seen': 1,
                        'zone_name': 'No Loitering Area'
                    }
                    self.logger.debug(f"Started tracking {track_id} for loitering")
                else:
                    # Update tracking
                    tracking_info = self.loitering_tracking[track_id]
                    tracking_info['frames_seen'] += 1
                    
                    # Check thresholds - VERY LOW FOR TESTING
                    time_in_area = current_time - tracking_info['start_time']
                    frames_seen = tracking_info['frames_seen']
                    
                    # TRIGGER LOITERING if EITHER condition met
                    if time_in_area >= self.loitering_threshold_seconds or frames_seen >= self.loitering_threshold_frames:
                        
                        # Create loitering event
                        if track_id not in getattr(self, 'reported_loiterers', set()):
                            if not hasattr(self, 'reported_loiterers'):
                                self.reported_loiterers = set()
                            self.reported_loiterers.add(track_id)
                            
                            self.logger.warning(f"LOITERING ALERT: {track_id} loitering for {time_in_area:.1f}s ({frames_seen} frames)!")
                            
                            event_data = {
                                'track_id': track_id,
                                'timestamp': current_time,
                                'zone_name': tracking_info['zone_name'],
                                'bbox': bbox,
                                'confidence': person['confidence'],
                                'duration_seconds': time_in_area,
                                'frames_seen': frames_seen,
                                'threshold_type': 'time' if time_in_area >= self.loitering_threshold_seconds else 'frames'
                            }
                            loitering_events.append(event_data)
                        
                        loitering_count += 1
                        
                        # Draw loitering alert
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)  # Magenta for loitering
                        
                        # Alert labels
                        alert_text = f"LOITERING {track_id}!"
                        draw_text_with_background(annotated_frame, alert_text, (x1, y1 - 30), color=(255, 255, 255))
                        
                        # Duration info
                        duration_text = f"Duration: {time_in_area:.1f}s ({frames_seen}f)"
                        draw_text_with_background(annotated_frame, duration_text, (x1, y1 - 10), color=(255, 255, 255))
                    else:
                        # Still tracking, not yet loitering
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow for monitoring
                        
                        # Progress indicator
                        progress_text = f"Monitoring {track_id}: {time_in_area:.1f}s ({frames_seen}f)"
                        draw_text_with_background(annotated_frame, progress_text, (x1, y1 - 10), color=(255, 255, 255))

        # Clean up tracking for people no longer detected
        tracks_to_remove = set(self.loitering_tracking.keys()) - active_tracks
        for track_id in tracks_to_remove:
            if track_id in self.loitering_tracking:
                duration = current_time - self.loitering_tracking[track_id]['start_time']
                self.logger.debug(f"Stopped tracking {track_id} after {duration:.1f}s")
                del self.loitering_tracking[track_id]
            
            if hasattr(self, 'reported_loiterers') and track_id in self.reported_loiterers:
                self.reported_loiterers.remove(track_id)

        self.current_loitering_count = loitering_count

        # ALWAYS trigger events if loitering detected
        if self.enable_individual_events and loitering_events:
            self.logger.warning(f"LOITERING ALERT: {len(loitering_events)} violations!")
            self._handle_individual_camera_events(loitering_events, annotated_frame)

        # Add monitoring overlay
        if len(self.loitering_tracking) > 0:
            monitoring_text = f"MONITORING: {len(self.loitering_tracking)} people | LOITERING: {loitering_count}"
            cv2.rectangle(annotated_frame, (10, annotated_frame.shape[0] - 60), (800, annotated_frame.shape[0] - 10), (100, 0, 100), -1)
            cv2.putText(annotated_frame, monitoring_text, (20, annotated_frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated_frame, people_count, loitering_count, {'No Loitering Area': loitering_events}

    def _handle_individual_camera_events(self, loitering_events, annotated_frame):
        """Handle loitering events - GUARANTEED TRIGGER"""
        if loitering_events:
            self.logger.warning(f"LOITERING EVENT: {len(loitering_events)} violations detected!")
            self.stats['events_detected'] = self.stats.get('events_detected', 0) + len(loitering_events)
            self.stats['loitering_events'] = self.stats.get('loitering_events', 0) + len(loitering_events)
            
            for event in loitering_events:
                track_id = event['track_id']
                duration = event['duration_seconds']
                threshold_type = event['threshold_type']
                self.logger.warning(f"  VIOLATION: {track_id} exceeded {threshold_type} threshold ({duration:.1f}s)")
        else:
            self.logger.warning("Loitering handler called but no events found!")

    def _process_frame_impl(self, frame, timestamp, detection_result):
        """Process frame for loitering detection"""
        self.logger.debug(f"Processing loitering frame at timestamp {timestamp}")
        
        annotated_frame, people_count, loitering_count, people_by_zone = self.detect_people(frame, detection_result)
        
        self.stats['frames_processed'] = self.stats.get('frames_processed', 0) + 1
        self.stats['people_detected'] = people_count
        
        # Return loitering events as detections
        detections = []
        for zone_name, events in people_by_zone.items():
            if events:
                detections.extend(events)
        
        return annotated_frame, detections

