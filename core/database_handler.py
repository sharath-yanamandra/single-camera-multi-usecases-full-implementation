# core/database_handler.py

import mysql.connector
from mysql.connector import Error
import json
import uuid
from datetime import datetime
import logging
from typing import Optional, Dict, Any, List

class DatabaseHandler:
    """Simple database handler for single camera testing"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None
        self.logger = logging.getLogger(__name__)
        
    def connect(self) -> bool:
        """Establish database connection"""
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            if self.connection.is_connected():
                self.logger.info(" Database connected successfully")
                return True
        except Error as e:
            self.logger.error(f" Database connection error: {e}")
            return False
        
        return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: tuple = None) -> Optional[List[Dict]]:
        """Execute a query and return results"""
        try:
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return None
            
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params)
            
            if query.strip().upper().startswith('SELECT'):
                result = cursor.fetchall()
                cursor.close()
                return result
            else:
                self.connection.commit()
                cursor.close()
                return [{'affected_rows': cursor.rowcount}]
                
        except Error as e:
            self.logger.error(f" Query execution error: {e}")
            return None
    
    def get_camera_info(self) -> Optional[Dict[str, Any]]:
        """Get camera information"""
        query = """
            SELECT c.camera_id, c.name, c.stream_url, c.metadata, p.project_id
            FROM cameras c
            JOIN projects p ON c.project_id = p.project_id
            WHERE c.status = 'active'
            LIMIT 1
        """
        
        result = self.execute_query(query)
        if result and len(result) > 0:
            camera_info = result[0]
            # Parse metadata if it exists
            if camera_info.get('metadata'):
                try:
                    camera_info['metadata'] = json.loads(camera_info['metadata'])
                except json.JSONDecodeError:
                    camera_info['metadata'] = {}
            
            return camera_info
        
        return None
    
    def save_event(self, camera_id: int, project_id: str, event_type: str, 
                   detection_data: Dict[str, Any], local_path: str = None, 
                   gcp_path: str = None, confidence: float = None) -> str:
        """Save detection event to database"""
        
        event_id = str(uuid.uuid4())
        
        # Determine severity based on event type
        severity_map = {
            'people_counting': 'info',
            'ppe_detection': 'warning', 
            'tailgating': 'warning',
            'intrusion': 'critical',
            'loitering': 'warning'
        }
        
        severity = severity_map.get(event_type, 'info')
        
        query = """
            INSERT INTO events (
                event_id, camera_id, project_id, event_type, severity,
                detection_data, local_image_path, gcp_image_path, 
                confidence_score, status
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'new')
        """
        
        params = (
            event_id,
            camera_id,
            project_id,
            event_type,
            severity,
            json.dumps(detection_data),
            local_path,
            gcp_path,
            confidence
        )
        
        result = self.execute_query(query, params)
        if result:
            self.logger.info(f" Event saved: {event_type} -> {event_id}")
            return event_id
        else:
            self.logger.error(f" Failed to save event: {event_type}")
            return None
    
    def update_processing_stats(self, camera_id: int, stats: Dict[str, int]):
        """Update processing statistics"""
        
        # Check if stats record exists
        check_query = "SELECT stat_id FROM processing_stats WHERE camera_id = %s ORDER BY timestamp DESC LIMIT 1"
        existing = self.execute_query(check_query, (camera_id,))
        
        if existing:
            # Update existing record
            update_query = """
                UPDATE processing_stats SET
                    frames_processed = frames_processed + %s,
                    total_detections = total_detections + %s,
                    people_counting_events = people_counting_events + %s,
                    ppe_detection_events = ppe_detection_events + %s,
                    tailgating_events = tailgating_events + %s,
                    intrusion_events = intrusion_events + %s,
                    loitering_events = loitering_events + %s,
                    processing_time_ms = processing_time_ms + %s,
                    timestamp = CURRENT_TIMESTAMP
                WHERE stat_id = %s
            """
            
            params = (
                stats.get('frames_processed', 0),
                stats.get('total_detections', 0),
                stats.get('people_counting_events', 0),
                stats.get('ppe_detection_events', 0),
                stats.get('tailgating_events', 0),
                stats.get('intrusion_events', 0),
                stats.get('loitering_events', 0),
                stats.get('processing_time_ms', 0),
                existing[0]['stat_id']
            )
            
            self.execute_query(update_query, params)
        else:
            # Create new record
            insert_query = """
                INSERT INTO processing_stats (
                    camera_id, frames_processed, total_detections,
                    people_counting_events, ppe_detection_events, tailgating_events,
                    intrusion_events, loitering_events, processing_time_ms
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                camera_id,
                stats.get('frames_processed', 0),
                stats.get('total_detections', 0),
                stats.get('people_counting_events', 0),
                stats.get('ppe_detection_events', 0),
                stats.get('tailgating_events', 0),
                stats.get('intrusion_events', 0),
                stats.get('loitering_events', 0),
                stats.get('processing_time_ms', 0)
            )
            
            self.execute_query(insert_query, params)
    
    def get_event_stats(self, camera_id: int, hours: int = 24) -> Dict[str, Any]:
        """Get event statistics for the last N hours"""
        
        query = """
            SELECT 
                event_type,
                COUNT(*) as count,
                MAX(timestamp) as latest_event
            FROM events 
            WHERE camera_id = %s 
                AND timestamp >= DATE_SUB(NOW(), INTERVAL %s HOUR)
            GROUP BY event_type
            ORDER BY count DESC
        """
        
        result = self.execute_query(query, (camera_id, hours))
        
        stats = {
            'total_events': 0,
            'by_type': {},
            'latest_event': None
        }
        
        if result:
            for row in result:
                event_type = row['event_type']
                count = row['count']
                latest = row['latest_event']
                
                stats['by_type'][event_type] = count
                stats['total_events'] += count
                
                if not stats['latest_event'] or latest > stats['latest_event']:
                    stats['latest_event'] = latest
        
        return stats
    
    def initialize_database(self) -> bool:
        """Initialize database with schema from SQL file"""
        try:
            # Read SQL file
            sql_file = "config/database_setup.sql"
            with open(sql_file, 'r', encoding='utf-8') as file:
                sql_content = file.read()
            
            # Split into individual statements
            statements = [stmt.strip() for stmt in sql_content.split(';') if stmt.strip()]
            
            if not self.connection or not self.connection.is_connected():
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            for statement in statements:
                if statement and not statement.startswith('--'):
                    try:
                        cursor.execute(statement)
                        self.connection.commit()
                    except Error as e:
                        self.logger.warning(f"Statement execution warning: {e}")
            
            cursor.close()
            self.logger.info(" Database initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f" Database initialization error: {e}")
            return False

# Test function
def test_database_connection():
    """Test database connection and operations"""
    from config.config import Config
    
    logging.basicConfig(level=logging.INFO)
    
    db = DatabaseHandler(Config.DB_CONFIG)
    
    if db.connect():
        print(" Database connection successful")
        
        # Test getting camera info
        camera_info = db.get_camera_info()
        if camera_info:
            print(f" Camera found: {camera_info['name']}")
            print(f" Stream URL: {camera_info['stream_url']}")
            
            # Test saving an event
            test_detection = {
                'test': True,
                'detection_count': 1,
                'timestamp': datetime.now().isoformat()
            }
            
            event_id = db.save_event(
                camera_info['camera_id'],
                camera_info['project_id'],
                'test_detection',
                test_detection
            )
            
            if event_id:
                print(f" Test event saved: {event_id}")
            
            # Get stats
            stats = db.get_event_stats(camera_info['camera_id'])
            print(f" Event stats: {stats}")
        
        db.disconnect()
    else:
        print("Database connection failed")

if __name__ == "__main__":
    test_database_connection()