# config/config.py  
# Copy this to: single_camera_test/config/config.py

import os
from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict, Any

class Config:
    """Configuration class for single camera multi-use case testing"""
    
    # Database Configuration - KEEP YOUR EXISTING DATABASE
    MYSQL_HOST = os.getenv('MYSQL_HOST', '34.93.87.255')
    MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))
    MYSQL_USER = os.getenv('MYSQL_USER', 'insighteye')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'insighteye0411')
    MYSQL_ROOT_PASSWORD = os.getenv('MYSQL_ROOT_PASSWORD', 'insighteye0411')
    MYSQL_DATABASE = os.getenv('MYSQL_DATABASE', 'dc_test')
    
    # Connection pooling
    MYSQL_POOL_SIZE = int(os.getenv('DB_POOL_SIZE', '10'))
    MYSQL_MAX_OVERFLOW = int(os.getenv('MYSQL_MAX_OVERFLOW', '5'))
    MYSQL_POOL_TIMEOUT = int(os.getenv('MYSQL_POOL_TIMEOUT', '30'))
    MYSQL_POOL_RECYCLE = int(os.getenv('MYSQL_POOL_RECYCLE', '3600'))
    
    # AI Model Configuration - ALL 3 MODELS
    DETECTION_MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', 'models/yolov11l.pt')
    POSE_ESTIMATION_MODEL_PATH = os.getenv('POSE_ESTIMATION_MODEL_PATH', 'models/yolov11l-pose.pt')
    PPE_DETECTION_MODEL_PATH = os.getenv('PPE_DETECTION_MODEL_PATH', 'models/ppe_detection.pt')
    POSE_ENABLED = os.getenv('POSE_ENABLED', 'false').lower() == 'true'

    # Detection Thresholds - FROM YOUR ORIGINAL CONFIG
    INTRUSION_THRESHOLD = int(os.getenv('INTRUSION_THRESHOLD', '1'))
    LOITERING_THRESHOLD = int(os.getenv('LOITERING_THRESHOLD', '300'))  # 5 minutes in seconds
    TAILGATING_TIME_LIMIT = float(os.getenv('TAILGATING_TIME_LIMIT', '2.0'))  # 2 seconds
    MOVEMENT_THRESHOLD = float(os.getenv('MOVEMENT_THRESHOLD', '1.0'))
    PEOPLE_COUNT_THRESHOLD = int(os.getenv('PEOPLE_COUNT_THRESHOLD', '10'))

    # Tracking Configuration
    TRACKING_THRESHOLD = float(os.getenv('TRACKING_THRESHOLD', '0.3'))
    MAX_AGE = int(os.getenv('MAX_AGE', '30'))
    MIN_HITS = int(os.getenv('MIN_HITS', '3'))

    # PPE Detection
    PPE_DETECTION_ENABLED = os.getenv('PPE_DETECTION_ENABLED', 'true').lower() == 'true'
    PPE_CONFIDENCE_THRESHOLD = float(os.getenv('PPE_CONFIDENCE_THRESHOLD', '0.7'))

    # Camera and RTSP settings
    RTSP_TIMEOUT = int(os.getenv('RTSP_TIMEOUT', '30'))
    CONNECTION_TIMEOUT = int(os.getenv('CONNECTION_TIMEOUT', '30'))
    
    # Model settings
    DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv('DETECTION_CONFIDENCE_THRESHOLD', '0.5'))
    NMS_THRESHOLD = float(os.getenv('NMS_THRESHOLD', '0.4'))

    # Processing Configuration - OPTIMIZED FOR SINGLE CAMERA
    BATCH_SIZE = 1  # Single camera
    BATCH_TIMEOUT = float(os.getenv('BATCH_TIMEOUT', '0.1'))
    MAX_PARALLEL_CAMERAS = 1  # Single camera
    READER_FPS_LIMIT = 5  # Reasonable FPS for testing
    
    # Video Configuration
    VIDEO_CODEC = os.getenv('VIDEO_CODEC', 'mp4v')
    VIDEO_FPS = int(os.getenv('VIDEO_FPS', '10'))
    VIDEO_BUFFER_PAST_SECONDS = int(os.getenv('VIDEO_BUFFER_PAST_SECONDS', '10'))
    VIDEO_BUFFER_FUTURE_SECONDS = int(os.getenv('VIDEO_BUFFER_FUTURE_SECONDS', '50'))
    VIDEO_BUFFER_SIZE = int(os.getenv('VIDEO_BUFFER_SIZE', '100'))
    VIDEO_DURATION = int(os.getenv('VIDEO_DURATION', '30'))  # seconds

    # Activity level based FPS settings - SIMPLIFIED
    ACTIVITY_LEVEL_HIGH = 5
    ACTIVITY_LEVEL_MEDIUM = 3  
    ACTIVITY_LEVEL_LOW = 2

    # Event Detection Configuration
    EVENT_COOLDOWN = int(os.getenv('EVENT_COOLDOWN', '10'))  # Shorter for testing
    
    # Storage Configuration
    FRAMES_OUTPUT_DIR = os.getenv('FRAMES_OUTPUT_DIR', 'outputs/frames')
    VIDEOS_OUTPUT_DIR = os.getenv('VIDEOS_OUTPUT_DIR', 'outputs/videos')
    LOGS_DIR = os.getenv('LOGS_DIR', 'logs')
    
    # Auto Recording Settings
    AUTO_RECORDING_ENABLED = os.getenv('AUTO_RECORDING_ENABLED', 'true').lower() == 'true'
    MEDIA_PREFERENCE = os.getenv('MEDIA_PREFERENCE', 'both')  # 'frames', 'videos', 'both'
    
    # Cloud Storage (GCP) - YOUR SETTINGS
    GCP_BUCKET_NAME = os.getenv('GCP_BUCKET_NAME', 'dc_bucket_test')
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'shining-env-447008-e1')
    BUCKET_NAME = GCP_BUCKET_NAME
    GCP_CREDENTIALS_PATH = os.getenv('GCP_CREDENTIALS_PATH', 'secrets/gcp-credentials.json')
    GOOGLE_APPLICATION_CREDENTIALS = GCP_CREDENTIALS_PATH
    
    # Single Camera Configuration - NEW
    SINGLE_CAMERA_MODE = True
    LOCAL_CAMERA_ID = "test_camera_001"  
    LOCAL_RTSP_URL = "rtsp://admin:password@192.168.29.213:554/ch0_0.264"  # YOUR CAMERA
    
    # Multi-use case settings for single camera - ALL YOUR USE CASES
    LOCAL_USE_CASES = [
        'people_counting',
        'ppe_detection', 
        'tailgating',
        'intrusion',
        'loitering'
    ]
    
    # Performance optimization for multi-use case
    SHARED_MODEL_DETECTION = True  # Use same YOLO detection for all use cases
    MAX_USE_CASES_PER_CAMERA = 5
    
    # Database Writer Configuration
    DB_WRITER_BATCH_SIZE = int(os.getenv('DB_WRITER_BATCH_SIZE', '10'))
    DB_WRITER_FLUSH_INTERVAL = int(os.getenv('DB_WRITER_FLUSH_INTERVAL', '5'))  # seconds
    
    # Logging Configuration  
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE_MAX_SIZE = int(os.getenv('LOG_FILE_MAX_SIZE', '10485760'))  # 10MB
    LOG_FILE_BACKUP_COUNT = int(os.getenv('LOG_FILE_BACKUP_COUNT', '5'))
    
    # Motion Detection Settings - DISABLED FOR SINGLE CAMERA TESTING
    MOTION_DETECTION_ENABLED = False  # Force processing without motion detection
    MOTION_AREA_THRESHOLD = float(os.getenv('MOTION_AREA_THRESHOLD', '0.001'))  # Very low threshold
    MOTION_IDLE_TIMEOUT = int(os.getenv('MOTION_IDLE_TIMEOUT', '5'))  # Short timeout

    # Monitoring and Health Checks
    HEALTH_CHECK_INTERVAL = int(os.getenv('HEALTH_CHECK_INTERVAL', '60'))  # seconds
    PERFORMANCE_LOG_INTERVAL = int(os.getenv('PERFORMANCE_LOG_INTERVAL', '300'))  # seconds

# Event types from your original system
class DatacenterEventTypes:
    """Event types for datacenter monitoring"""
    PEOPLE_COUNTING = "people_counting"
    PPE_VIOLATION = "ppe_violation"
    TAILGATING = "tailgating"
    INTRUSION = "intrusion"
    LOITERING = "loitering"
    CAMERA_TAMPER = "camera_tamper"
    MOTION_DETECTION = "motion_detection"
    ACCESS_VIOLATION = "access_violation"

# Camera types from your original system
class DatacenterCameraTypes:
    """Camera types for datacenter monitoring"""
    ENTRY_MONITORING = "entry_monitoring"
    PPE_DETECTION = "ppe_detection"
    PEOPLE_COUNTING = "people_counting"
    INTRUSION_DETECTION = "intrusion_detection"
    TAILGATING_DETECTION = "tailgating_detection"
    LOITERING_DETECTION = "loitering_detection"
    GENERAL_SURVEILLANCE = "general_surveillance"

# Create directories if they don't exist
os.makedirs(Config.FRAMES_OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.VIDEOS_OUTPUT_DIR, exist_ok=True)
os.makedirs(Config.LOGS_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('secrets', exist_ok=True)

# Validation function
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check model files exist
    if not os.path.exists(Config.DETECTION_MODEL_PATH):
        errors.append(f"YOLO detection model not found: {Config.DETECTION_MODEL_PATH}")
    
    if not os.path.exists(Config.PPE_DETECTION_MODEL_PATH):
        errors.append(f"PPE detection model not found: {Config.PPE_DETECTION_MODEL_PATH}")
        
    if not os.path.exists(Config.POSE_ESTIMATION_MODEL_PATH):
        errors.append(f"Pose estimation model not found: {Config.POSE_ESTIMATION_MODEL_PATH}")
    
    # Check credentials file exists
    if not os.path.exists(Config.GCP_CREDENTIALS_PATH):
        errors.append(f"GCP credentials not found: {Config.GCP_CREDENTIALS_PATH}")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        return False
    
    print(" Configuration validated successfully")
    return True

if __name__ == "__main__":
    validate_config()