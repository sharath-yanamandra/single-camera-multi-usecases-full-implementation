# core/gcp_uploader.py
# core/gcp_uploader.py - FIXED VERSION
# core/gcp_uploader.py - FINAL FIXED VERSION

import os
import cv2
import time
import uuid
import json
import logging
import numpy as np
from datetime import datetime
from threading import Thread
from queue import Queue, Empty
from typing import Optional, Dict, Any
from google.cloud import storage
from google.oauth2 import service_account

class GCPUploader:
    """GCP Storage uploader for camera events - FINAL FIXED VERSION"""
    
    def __init__(self, credentials_path: str, bucket_name: str, project_id: str):
        self.credentials_path = credentials_path
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize GCP client
        self.storage_client = None
        self.bucket = None
        self._init_gcp_client()
        
        # Upload queue for background processing
        self.upload_queue = Queue()
        self.running = True
        
        # Start background upload worker
        self.upload_thread = Thread(target=self._upload_worker, daemon=True)
        self.upload_thread.start()
        
        # Statistics
        self.stats = {
            'total_uploads': 0,
            'successful_uploads': 0,
            'failed_uploads': 0,
            'total_size_bytes': 0
        }
    
    def _init_gcp_client(self) -> bool:
        """Initialize GCP storage client"""
        try:
            if not os.path.exists(self.credentials_path):
                self.logger.error(f"GCP credentials file not found: {self.credentials_path}")
                return False
            
            # Initialize client with service account credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path
            )
            
            self.storage_client = storage.Client(
                credentials=credentials,
                project=self.project_id
            )
            
            # Get bucket
            self.bucket = self.storage_client.bucket(self.bucket_name)
            
            # Test bucket access
            if self.bucket.exists():
                self.logger.info(f"GCP Storage connected: {self.bucket_name}")
                return True
            else:
                self.logger.error(f"GCP bucket does not exist: {self.bucket_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"GCP initialization error: {e}")
            return False
    
    def _make_json_serializable(self, obj):
        """Make object JSON serializable - RECURSIVE"""
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
        elif hasattr(obj, '__dict__'):
            return self._make_json_serializable(obj.__dict__)
        else:
            return obj
    
    def save_and_upload_event(self, frame, event_type: str, camera_id: int, 
                            detection_data: Dict[str, Any] = None) -> tuple:
        """Save frame locally and queue for GCP upload"""
        try:
            # Generate unique identifiers
            event_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            # Create organized directory structure
            date_str = timestamp.strftime("%Y-%m-%d")
            hour_str = timestamp.strftime("%H")
            
            # Local path structure
            local_dir = os.path.join(
                "outputs", "frames", event_type, date_str, hour_str
            )
            os.makedirs(local_dir, exist_ok=True)
            
            # Generate filename
            filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{event_id[:8]}_{event_type}.jpg"
            local_path = os.path.join(local_dir, filename)
            
            # Save frame locally
            success = cv2.imwrite(local_path, frame)
            if not success:
                self.logger.error(f"Failed to save frame locally: {local_path}")
                return None, None
            
            # Generate GCP path
            gcp_path = f"single_camera_test/camera_{camera_id}/{event_type}/events/{timestamp.strftime('%Y/%m/%d/%H')}/{filename}"
            
            # Queue for upload if GCP is available
            if self.bucket:
                # FIXED: Make detection_data JSON serializable before queuing
                safe_detection_data = self._make_json_serializable(detection_data) if detection_data else None
                
                upload_item = {
                    'local_path': local_path,
                    'gcp_path': gcp_path,
                    'event_type': event_type,
                    'event_id': event_id,
                    'timestamp': timestamp.isoformat(),  # FIXED: Convert to string
                    'detection_data': safe_detection_data,
                    'camera_id': camera_id
                }
                
                self.upload_queue.put(upload_item)
                self.logger.info(f"Saved and queued: {event_type} -> {filename}")
            else:
                self.logger.warning(f"GCP not available, saved locally only: {filename}")
            
            return local_path, f"gs://{self.bucket_name}/{gcp_path}" if self.bucket else None
            
        except Exception as e:
            self.logger.error(f"Error saving event frame: {e}")
            return None, None
    
    def _upload_worker(self):
        """Background worker for uploading files to GCP - FIXED VERSION"""
        self.logger.info("GCP upload worker started")
        
        while self.running:
            try:
                # FIXED: Use timeout and catch Empty exception properly
                upload_item = self.upload_queue.get(timeout=2.0)
                
                # Perform upload
                success = self._upload_single_file(upload_item)
                
                # Update statistics
                self.stats['total_uploads'] += 1
                if success:
                    self.stats['successful_uploads'] += 1
                else:
                    self.stats['failed_uploads'] += 1
                
                # Mark task as done
                self.upload_queue.task_done()
                
            except Empty:
                # FIXED: This is normal when queue is empty, don't log as error
                continue
            except Exception as e:
                self.logger.error(f"Upload worker error: {e}")
                # Continue processing even if there's an error
                continue
        
        self.logger.info("GCP upload worker stopped")
    
    def _upload_single_file(self, upload_item: Dict[str, Any]) -> bool:
        """Upload a single file to GCP"""
        try:
            if not self.bucket:
                self.logger.warning("No GCP bucket available for upload")
                return False
            
            local_path = upload_item['local_path']
            gcp_path = upload_item['gcp_path']
            
            # Check if local file exists
            if not os.path.exists(local_path):
                self.logger.error(f"Local file not found: {local_path}")
                return False
            
            # Get file size for statistics
            file_size = os.path.getsize(local_path)
            
            # Create blob and upload
            blob = self.bucket.blob(gcp_path)
            
            # Set metadata - FIXED: Ensure all values are strings
            metadata = {
                'event_type': str(upload_item['event_type']),
                'event_id': str(upload_item['event_id']),
                'camera_id': str(upload_item['camera_id']),
                'timestamp': str(upload_item['timestamp']),
                'uploaded_at': datetime.now().isoformat(),
                'source_system': 'single_camera_test'
            }
            
            # FIXED: Only add detection_data if it exists and can be serialized
            if upload_item.get('detection_data'):
                try:
                    # Convert to JSON string safely
                    detection_json = json.dumps(upload_item['detection_data'])
                    metadata['detection_data'] = detection_json
                except Exception as e:
                    self.logger.warning(f"Could not serialize detection_data: {e}")
                    metadata['detection_data'] = str(upload_item['detection_data'])
            
            blob.metadata = metadata
            
            # Upload file
            start_time = time.time()
            blob.upload_from_filename(local_path)
            upload_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_size_bytes'] += file_size
            
            self.logger.info(f"Uploaded: {upload_item['event_type']} -> gs://{self.bucket_name}/{gcp_path} ({file_size} bytes, {upload_time:.2f}s)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Upload failed for {upload_item.get('local_path', 'unknown')}: {e}")
            return False
    
    def get_upload_stats(self) -> Dict[str, Any]:
        """Get upload statistics"""
        queue_size = self.upload_queue.qsize()
        
        return {
            'total_uploads': self.stats['total_uploads'],
            'successful_uploads': self.stats['successful_uploads'],
            'failed_uploads': self.stats['failed_uploads'],
            'success_rate': (self.stats['successful_uploads'] / max(1, self.stats['total_uploads'])) * 100,
            'total_size_mb': self.stats['total_size_bytes'] / (1024 * 1024),
            'queue_size': queue_size,
            'is_connected': self.bucket is not None
        }
    
    def test_connection(self) -> bool:
        """Test GCP connection by uploading a test file"""
        try:
            if not self.bucket:
                return False
            
            # Create test file
            test_content = f"Test upload from single camera system at {datetime.now()}"
            test_path = "test_connection.txt"
            
            with open(test_path, 'w') as f:
                f.write(test_content)
            
            # Upload test file
            test_blob_path = f"single_camera_test/test/connection_test_{int(time.time())}.txt"
            blob = self.bucket.blob(test_blob_path)
            blob.upload_from_filename(test_path)
            
            # Verify upload by downloading
            downloaded_content = blob.download_as_text()
            
            # Cleanup
            blob.delete()
            os.remove(test_path)
            
            if test_content == downloaded_content:
                self.logger.info("GCP connection test successful")
                return True
            else:
                self.logger.error("GCP connection test failed - content mismatch")
                return False
                
        except Exception as e:
            self.logger.error(f"GCP connection test failed: {e}")
            return False
    
    def stop(self):
        """Stop the uploader and wait for pending uploads"""
        self.logger.info("Stopping GCP uploader...")
        
        # Wait for pending uploads (with timeout)
        if not self.upload_queue.empty():
            self.logger.info(f"Waiting for {self.upload_queue.qsize()} pending uploads...")
            try:
                # Wait up to 10 seconds for queue to empty
                for _ in range(10):
                    if self.upload_queue.empty():
                        break
                    time.sleep(1)
                else:
                    self.logger.warning("Timeout waiting for uploads to complete")
            except:
                pass
        
        # Stop worker
        self.running = False
        
        # Wait for worker thread to finish
        if self.upload_thread.is_alive():
            self.upload_thread.join(timeout=3.0)
        
        # Print final statistics
        stats = self.get_upload_stats()
        self.logger.info(f"Final upload stats: {stats['successful_uploads']}/{stats['total_uploads']} successful, {stats['total_size_mb']:.2f} MB total")

# Test function
def test_gcp_uploader():
    """Test GCP uploader functionality"""
    import numpy as np
    from config.config import Config
    
    logging.basicConfig(level=logging.INFO)
    
    # Create test uploader
    uploader = GCPUploader(
        Config.GCP_CREDENTIALS_PATH,
        Config.GCP_BUCKET_NAME,
        Config.GCP_PROJECT_ID
    )
    
    # Test connection
    if uploader.test_connection():
        print("GCP connection test passed")
        
        # Create test image
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_frame, "TEST FRAME", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # Test upload
        local_path, gcp_path = uploader.save_and_upload_event(
            test_frame, 
            'test_event', 
            1, 
            {'test': True, 'timestamp': datetime.now().isoformat()}
        )
        
        if local_path and gcp_path:
            print(f"Test upload successful:")
            print(f"   Local: {local_path}")
            print(f"   GCP: {gcp_path}")
        
        # Wait a moment for upload to complete
        time.sleep(2)
        
        # Print stats
        stats = uploader.get_upload_stats()
        print(f"Upload stats: {stats}")
        
        # Stop uploader
        uploader.stop()
    else:
        print("GCP connection test failed")

if __name__ == "__main__":
    test_gcp_uploader()

