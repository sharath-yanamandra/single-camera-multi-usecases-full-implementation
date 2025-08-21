# main.py
# Copy this to: single_camera_test/main.py

import os
import sys
import argparse
import logging
import time
import asyncio
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.config import Config, validate_config
from core.database_handler import DatabaseHandler
from core.gcp_uploader import GCPUploader  
from core.video_processor import SingleCameraVideoProcessor

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory
    os.makedirs(Config.LOGS_DIR, exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = os.path.join(Config.LOGS_DIR, 'single_camera.log')
    
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL),
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def print_banner():
    """Print application banner"""
    print("="*70)
    print(" SINGLE CAMERA MULTI-USE CASE MONITORING SYSTEM")
    print("="*70)
    print(f" Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Camera: {Config.LOCAL_RTSP_URL}")
    print(f" Use Cases: {', '.join(Config.LOCAL_USE_CASES)}")
    print(f" GCP Bucket: {Config.GCP_BUCKET_NAME}")
    print(f" Database: {Config.MYSQL_HOST}/{Config.MYSQL_DATABASE}")
    print("="*70)

def test_database_connection():
    """Test database connectivity"""
    print("🔍 Testing database connection...")
    
    db = DatabaseHandler({
        'host': Config.MYSQL_HOST,
        'user': Config.MYSQL_USER, 
        'password': Config.MYSQL_PASSWORD,
        'database': Config.MYSQL_DATABASE,
        'port': Config.MYSQL_PORT
    })
    
    if db.connect():
        print(" Database connection successful")
        
        # Test getting camera info
        camera_info = db.get_camera_info()
        if camera_info:
            print(f" Camera found: {camera_info['name']}")
        else:
            print(" No camera found in database (will create test camera)")
        
        db.disconnect()
        return True
    else:
        print(" Database connection failed")
        return False

def test_gcp_connection():
    """Test GCP storage connectivity"""
    print(" Testing GCP storage connection...")
    
    uploader = GCPUploader(
        Config.GCP_CREDENTIALS_PATH,
        Config.GCP_BUCKET_NAME,
        Config.GCP_PROJECT_ID
    )
    
    if uploader.test_connection():
        print(" GCP storage connection successful")
        uploader.stop()
        return True
    else:
        print("GCP storage connection failed")
        uploader.stop()
        return False

def test_camera_stream():
    """Test camera stream connectivity"""
    print("Testing camera stream...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(Config.LOCAL_RTSP_URL)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                print("Camera stream accessible")
                return True
            else:
                print("Camera stream not providing frames")
                return False
        else:
            print("Cannot open camera stream")
            return False
    except Exception as e:
        print(f"Camera test failed: {e}")
        return False

def initialize_database():
    """Initialize database with schema"""
    print(" Initializing database...")
    
    db = DatabaseHandler({
        'host': Config.MYSQL_HOST,
        'user': Config.MYSQL_USER,
        'password': Config.MYSQL_PASSWORD, 
        'database': Config.MYSQL_DATABASE,
        'port': Config.MYSQL_PORT
    })
    
    if db.initialize_database():
        print("Database initialized successfully")
        return True
    else:
        print(" Database initialization failed")
        return False

def run_system_check():
    """Run complete system health check"""
    print(" Running system health check...")
    print("-" * 50)
    
    checks_passed = 0
    total_checks = 6
    
    # 1. Configuration validation
    print("1. Validating configuration...")
    if validate_config():
        checks_passed += 1
    
    # 2. Database connection
    print("2. Testing database connection...")
    if test_database_connection():
        checks_passed += 1
    
    # 3. GCP connection
    print("3. Testing GCP storage...")
    if test_gcp_connection():
        checks_passed += 1
    
    # 4. Camera stream
    print("4. Testing camera stream...")
    if test_camera_stream():
        checks_passed += 1
    
    # 5. Model files
    print("5. Checking model files...")
    models_ok = True
    models = [
        Config.DETECTION_MODEL_PATH,
        Config.PPE_DETECTION_MODEL_PATH,
        Config.POSE_ESTIMATION_MODEL_PATH
    ]
    
    for model_path in models:
        if not os.path.exists(model_path):
            print(f" Model not found: {model_path}")
            models_ok = False
        else:
            print(f"Model found: {os.path.basename(model_path)}")
    
    if models_ok:
        checks_passed += 1
    
    # 6. Camera models import
    print("6. Testing camera models import...")
    try:
        from camera_models.people_count_monitoring import PeopleCountingMonitor
        from camera_models.ppe_kit_monitoring import PPEDetector
        from camera_models.tailgating_zone_monitoring import TailgatingZoneMonitor
        from camera_models.intrusion_zone_monitoring import IntrusionZoneMonitor
        from camera_models.loitering_zone_monitoring import LoiteringZoneMonitor
        print(" All camera models imported successfully")
        checks_passed += 1
    except ImportError as e:
        print(f"Camera models import failed: {e}")
    
    print("-" * 50)
    print(f" System Check: {checks_passed}/{total_checks} passed")
    
    if checks_passed == total_checks:
        print("System ready for monitoring!")
        return True
    else:
        print(" Some checks failed. Please fix issues before running.")
        return False

def start_monitoring():
    """Start the main monitoring process"""
    try:
        print_banner()
        
        # Validate configuration
        if not validate_config():
            print("Configuration validation failed")
            return False
        
        print("Starting Single Camera Video Processor...")
        
        # Create and start processor
        processor = SingleCameraVideoProcessor(Config)
        processor.start_processing()
        
        return True
        
    except KeyboardInterrupt:
        print("\n Monitoring stopped by user")
        return True
    except Exception as e:
        print(f"Monitoring error: {e}")
        logging.getLogger(__name__).error(f"Monitoring error: {e}", exc_info=True)
        return False

def test_camera_only():
    """Test camera stream with preview window"""
    print(" Testing camera stream with preview...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(Config.LOCAL_RTSP_URL)
        
        if not cap.isOpened():
            print("Failed to open camera")
            return False
        
        print(" Camera opened successfully")
        print(" Preview window opened - press 'q' to quit")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            frame_count += 1
            
            # Add frame counter
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Camera Test - Press Q to Quit', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"Camera test completed - processed {frame_count} frames")
        return True
        
    except Exception as e:
        print(f"Camera test failed: {e}")
        return False

def show_stats():
    """Show system statistics"""
    print(" System Statistics")
    print("-" * 40)
    
    # Database stats
    db = DatabaseHandler({
        'host': Config.MYSQL_HOST,
        'user': Config.MYSQL_USER,
        'password': Config.MYSQL_PASSWORD,
        'database': Config.MYSQL_DATABASE,
        'port': Config.MYSQL_PORT
    })
    
    if db.connect():
        try:
            camera_info = db.get_camera_info()
            if camera_info:
                stats = db.get_event_stats(camera_info['camera_id'], 24)
                print(f"Camera: {camera_info['name']}")
                print(f"Events in last 24 hours: {stats['total_events']}")
                
                if stats['by_type']:
                    print("Events by type:")
                    for event_type, count in stats['by_type'].items():
                        print(f"  • {event_type.replace('_', ' ').title()}: {count}")
                
                if stats['latest_event']:
                    print(f"Latest event: {stats['latest_event']}")
            else:
                print("No camera found in database")
        except Exception as e:
            print(f"Error getting stats: {e}")
        
        db.disconnect()
    
    # GCP stats
    print("\nGCP Storage:")
    try:
        import subprocess
        result = subprocess.run([
            'gsutil', 'du', '-sh', f'gs://{Config.GCP_BUCKET_NAME}/single_camera_test/'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            print(f"Total storage used: {result.stdout.strip()}")
        else:
            print("Unable to get GCP storage stats")
    except Exception:
        print("gsutil not available for storage stats")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Single Camera Multi-Use Case Monitoring System')
    
    parser.add_argument('command', nargs='?', default='monitor', 
                       choices=['monitor', 'init-db', 'health-check', 'stats', 'test-camera'],
                       help='Command to execute')
    
    parser.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    parser.add_argument('--camera-url', 
                       help='Override camera RTSP URL')
    
    args = parser.parse_args()
    
    # Override config based on arguments
    if args.log_level:
        Config.LOG_LEVEL = args.log_level
    
    if args.camera_url:
        Config.LOCAL_RTSP_URL = args.camera_url
        print(f"Using camera URL: {args.camera_url}")
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'monitor':
            success = start_monitoring()
            return 0 if success else 1
            
        elif args.command == 'init-db':
            success = initialize_database()
            return 0 if success else 1
            
        elif args.command == 'health-check':
            success = run_system_check()
            return 0 if success else 1
            
        elif args.command == 'stats':
            show_stats()
            return 0
            
        elif args.command == 'test-camera':
            success = test_camera_only()
            return 0 if success else 1
        
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        print(f"Application error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)