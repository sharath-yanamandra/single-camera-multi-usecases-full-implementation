-- config/database_setup.sql

-- Simple database setup for single camera testing

USE dc_test;

-- Drop existing tables if they exist (for clean setup)
-- SET FOREIGN_KEY_CHECKS = 0;
-- DROP TABLE IF EXISTS events;
-- DROP TABLE IF EXISTS cameras;
-- DROP TABLE IF EXISTS projects;
-- SET FOREIGN_KEY_CHECKS = 1;

-- Projects table (simplified)
CREATE TABLE IF NOT EXISTS projects (
    project_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL DEFAULT 'test-user',
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(100) DEFAULT 'single_camera_test',
    location VARCHAR(255),
    status ENUM('active', 'inactive') DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Cameras table (simplified) 
CREATE TABLE IF NOT EXISTS cameras (
    camera_id INT AUTO_INCREMENT PRIMARY KEY,
    project_id VARCHAR(36) NOT NULL,
    name VARCHAR(255) NOT NULL,
    stream_url VARCHAR(255) NOT NULL,
    camera_type VARCHAR(100) DEFAULT 'multi_use_case',
    status ENUM('active', 'inactive', 'maintenance') DEFAULT 'active',
    metadata JSON DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
);

-- Events table (stores all detections from all use cases)
CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR(36) PRIMARY KEY,
    camera_id INT NOT NULL,
    project_id VARCHAR(36) NOT NULL,
    event_type VARCHAR(100) NOT NULL,
    event_subtype VARCHAR(100) DEFAULT NULL,
    severity ENUM('info', 'warning', 'critical') DEFAULT 'info',
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    detection_data JSON DEFAULT NULL,
    local_image_path VARCHAR(500) DEFAULT NULL,
    gcp_image_path VARCHAR(500) DEFAULT NULL,
    status ENUM('new', 'processed', 'archived') DEFAULT 'new',
    confidence_score DECIMAL(5,4) DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
    INDEX idx_event_type (event_type),
    INDEX idx_timestamp (timestamp),
    INDEX idx_status (status),
    INDEX idx_camera_timestamp (camera_id, timestamp)
);

-- Processing stats table (optional - for monitoring)
CREATE TABLE IF NOT EXISTS processing_stats (
    stat_id INT AUTO_INCREMENT PRIMARY KEY,
    camera_id INT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frames_processed INT DEFAULT 0,
    total_detections INT DEFAULT 0,
    people_counting_events INT DEFAULT 0,
    ppe_detection_events INT DEFAULT 0,
    tailgating_events INT DEFAULT 0,
    intrusion_events INT DEFAULT 0,
    loitering_events INT DEFAULT 0,
    processing_time_ms INT DEFAULT 0,
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
);

-- Insert test project
INSERT IGNORE INTO projects (project_id, user_id, name, description, location, status)
VALUES 
('single-cam-test-001', 'test-user-001', 'Single Camera Multi-Use Case Test', 
 'Testing all 5 use cases on single camera with GCP storage', 'Home Office', 'active');

-- Insert test camera with your RTSP URL
INSERT IGNORE INTO cameras (project_id, name, stream_url, camera_type, status, metadata)
VALUES 
('single-cam-test-001', 'Office Test Camera', 'rtsp://admin:password@192.168.29.213:554/ch0_0.264', 
 'multi_use_case', 'active', 
 JSON_OBJECT(
    'use_cases', JSON_ARRAY('people_counting', 'ppe_detection', 'tailgating', 'intrusion', 'loitering'),
    'activity_level', 'high',
    'resolution', '1920x1080',
    'fps', 25,
    'location_details', JSON_OBJECT(
        'room', 'Office',
        'floor', 'Ground Floor', 
        'description', 'Main office area with desk and entry door'
    )
 ));

-- Verify setup
SELECT 'Database setup completed successfully!' AS Status;

-- Show created data
SELECT 'Projects:' AS Info;
SELECT project_id, name, type, status FROM projects;

SELECT 'Cameras:' AS Info;  
SELECT camera_id, name, camera_type, status FROM cameras;

-- Create initial processing stats entry
INSERT IGNORE INTO processing_stats (camera_id, frames_processed, total_detections)
SELECT camera_id, 0, 0 FROM cameras WHERE name = 'Office Test Camera';

SELECT 'Ready for single camera multi-use case testing!' AS Ready;
SELECT 'All 5 use cases will store events in the events table' AS Note;
SELECT 'GCP uploads will go to: gs://dc_bucket_test/single_camera_test/' AS GCP_Info;