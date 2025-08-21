import mysql.connector
import json

def fixed_database_setup():
    """Fixed database setup script - handles foreign key constraints properly"""
    
    config = {
        'host': '34.93.87.255',
        'user': 'insighteye',
        'password': 'insighteye0411',
        'database': 'dc_test',
        'port': 3306
    }
    
    try:
        print(" Connecting to database...")
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # STEP 1: Drop existing tables if they exist (to start fresh)
        print("Cleaning up existing tables...")
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
        
        tables_to_drop = ['processing_stats', 'events', 'cameras', 'projects']
        for table in tables_to_drop:
            try:
                cursor.execute(f"DROP TABLE IF EXISTS {table}")
                print(f" Dropped {table} (if existed)")
            except:
                pass
        
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
        conn.commit()
        
        # STEP 2: Create tables in correct order
        print("Creating tables in correct order...")
        
        # Create projects table FIRST
        print("  1/4 Creating projects table...")
        cursor.execute("""
            CREATE TABLE projects (
                project_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(36) NOT NULL DEFAULT 'test-user',
                name VARCHAR(255) NOT NULL,
                description TEXT,
                status ENUM('active', 'inactive') DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Create cameras table SECOND
        print("  2/4 Creating cameras table...")
        cursor.execute("""
            CREATE TABLE cameras (
                camera_id INT AUTO_INCREMENT PRIMARY KEY,
                project_id VARCHAR(36) NOT NULL,
                name VARCHAR(255) NOT NULL,
                stream_url VARCHAR(255) NOT NULL,
                status ENUM('active', 'inactive') DEFAULT 'active',
                metadata JSON DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        
        # Create events table THIRD
        print("  3/4 Creating events table...")
        cursor.execute("""
            CREATE TABLE events (
                event_id VARCHAR(36) PRIMARY KEY,
                camera_id INT NOT NULL,
                project_id VARCHAR(36) NOT NULL,
                event_type VARCHAR(100) NOT NULL,
                event_subtype VARCHAR(100) DEFAULT NULL,
                severity ENUM('info', 'warning', 'critical') DEFAULT 'info',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detection_data JSON DEFAULT NULL,
                local_image_path VARCHAR(500),
                gcp_image_path VARCHAR(500),
                status ENUM('new', 'processed', 'archived') DEFAULT 'new',
                confidence_score DECIMAL(5,4) DEFAULT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
                FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE,
                INDEX idx_event_type (event_type),
                INDEX idx_timestamp (timestamp),
                INDEX idx_status (status),
                INDEX idx_camera_timestamp (camera_id, timestamp)
            )
        """)
        # print("  3/4 Creating events table...")
        # cursor.execute("""
        #     CREATE TABLE events (
        #         event_id VARCHAR(36) PRIMARY KEY,
        #         camera_id INT NOT NULL,
        #         project_id VARCHAR(36) NOT NULL,
        #         event_type VARCHAR(100) NOT NULL,
        #         timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        #         detection_data JSON DEFAULT NULL,
        #         local_image_path VARCHAR(500),
        #         gcp_image_path VARCHAR(500),
        #         status ENUM('new', 'processed') DEFAULT 'new',
        #         FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE,
        #         FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE CASCADE
        #     )
        # """)
        conn.commit()
        
        # Create processing_stats table FOURTH
        print("  4/4 Creating processing_stats table...")
        cursor.execute("""
            CREATE TABLE processing_stats (
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
                FOREIGN KEY (camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        
        # STEP 3: Insert sample data
        print("Inserting sample data...")
        
        # Insert project FIRST
        cursor.execute("""
            INSERT INTO projects (project_id, name, description, status)
            VALUES ('test-project-001', 'Single Camera Test', 'Multi-use case testing', 'active')
        """)
        conn.commit()
        
        # Insert camera SECOND
        metadata = json.dumps({
            "use_cases": ["people_counting", "ppe_detection", "tailgating", "intrusion", "loitering"],
            "activity_level": "high"
        })
        
        cursor.execute("""
            INSERT INTO cameras (project_id, name, stream_url, status, metadata)
            VALUES ('test-project-001', 'Single Camera Test', 'rtsp://admin:password@192.168.29.213:554/ch0_0.264', 'active', %s)
        """, (metadata,))
        conn.commit()
        
        # Verify setup
        cursor.execute("SELECT camera_id, name FROM cameras")
        camera_info = cursor.fetchone()
        
        print(f"Database setup complete!")
        print(f"Camera ID: {camera_info[0]}, Name: {camera_info[1]}")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f" Database setup failed: {e}")
        return False

if __name__ == "__main__":
    fixed_database_setup()