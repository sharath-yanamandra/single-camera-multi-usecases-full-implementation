# debug_detection.py - Test what the camera actually sees

import cv2
import numpy as np
from ultralytics import YOLO
import time

def test_camera_and_detection():
    """Test camera feed and YOLO detection"""
    
    print(" DEBUGGING CAMERA AND DETECTION")
    print("="*50)
    
    # Camera URL
    camera_url = "rtsp://admin:password@192.168.29.213:554/ch0_0.264"
    
    print(f" Connecting to camera: {camera_url}")
    
    # Test camera connection
    cap = cv2.VideoCapture(camera_url)
    if not cap.isOpened():
        print(" Cannot connect to camera!")
        return False
    
    print(" Camera connected successfully")
    
    # Load YOLO model
    print("Loading YOLO model...")
    try:
        model = YOLO("models/yolov11l.pt")
        print(" YOLO model loaded successfully")
    except Exception as e:
        print(f" Failed to load YOLO model: {e}")
        return False
    
    # Test detection on frames
    print("\n Testing detection on camera frames...")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    people_detected_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to read frame")
            break
            
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Check for people detections
        people_count = 0
        all_detections = []
        
        if results:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0]) if box.cls.numel() > 0 else -1
                        confidence = float(box.conf[0]) if box.conf.numel() > 0 else 0
                        class_name = result.names.get(class_id, f"class_{class_id}")
                        
                        all_detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'class_id': class_id
                        })
                        
                        if class_name == 'person' and confidence > 0.3:  # Lower threshold
                            people_count += 1
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Draw bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Person {confidence:.2f}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if people_count > 0:
            people_detected_frames += 1
        
        # Add info to frame
        info_text = [
            f"Frame: {frame_count}",
            f"People: {people_count}",
            f"Total detections: {len(all_detections)}",
            f"People frames: {people_detected_frames}"
        ]
        
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, 30 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show frame (if possible)
        try:
            cv2.imshow('Camera Debug - Press Q to quit', frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"debug_frame_{frame_count}_{people_count}people.jpg"
                cv2.imwrite(filename, frame)
                print(f" Saved frame: {filename}")
                
        except Exception as e:
            # If display doesn't work, just print stats
            if frame_count % 30 == 0:  # Every 30 frames
                print(f"Frame {frame_count}: {people_count} people, {len(all_detections)} total detections")
        
        # Auto-exit after 100 frames for testing
        if frame_count >= 100:
            break
    
    # Cleanup
    cap.release()
    try:
        cv2.destroyAllWindows()
    except:
        pass
    
    # Summary
    print(f"\n DETECTION SUMMARY:")
    print(f"Total frames: {frame_count}")
    print(f"Frames with people: {people_detected_frames}")
    print(f"People detection rate: {(people_detected_frames/max(1,frame_count))*100:.1f}%")
    
    if people_detected_frames == 0:
        print("\n  NO PEOPLE DETECTED!")
        print("Possible issues:")
        print("1. No people in camera view")
        print("2. Camera angle/lighting issues")
        print("3. YOLO model confidence too high")
        print("4. Camera feed quality issues")
        print("\nSuggestions:")
        print("- Move in front of the camera")
        print("- Check camera angle and lighting")
        print("- Lower detection confidence threshold")
    else:
        print(f"People detected successfully!")
        return True
    
    return people_detected_frames > 0

def test_zone_positions():
    """Test if zones are positioned correctly"""
    print("\n TESTING ZONE POSITIONS")
    print("="*50)
    
    # Default zones from your config
    zones = {
        'people_counting': [[200, 200], [800, 200], [800, 600], [200, 600]],
        'ppe_detection': [[300, 250], [900, 250], [900, 700], [300, 700]],
        'tailgating': [[250, 300], [750, 300], [750, 650], [250, 650]],
        'intrusion': [[500, 200], [1200, 200], [1200, 800], [500, 800]],
        'loitering': [[400, 350], [1000, 350], [1000, 750], [400, 750]]
    }
    
    print("Current zone coordinates:")
    for use_case, coords in zones.items():
        print(f"  {use_case}: {coords}")
    
    print(f"\nThese zones assume camera resolution around 1280x720 or higher")
    print(f"If your camera has different resolution, zones might be outside view")
    
    # Test actual camera resolution
    camera_url = "rtsp://admin:password@192.168.29.213:554/ch0_0.264"
    cap = cv2.VideoCapture(camera_url)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            height, width = frame.shape[:2]
            print(f" Actual camera resolution: {width}x{height}")
            
            # Check if zones fit
            for use_case, coords in zones.items():
                max_x = max(coord[0] for coord in coords)
                max_y = max(coord[1] for coord in coords)
                
                if max_x > width or max_y > height:
                    print(f" {use_case} zone extends outside camera view!")
                    print(f"   Zone max: ({max_x}, {max_y}), Camera: ({width}, {height})")
                else:
                    print(f"{use_case} zone fits in camera view")
        cap.release()

if __name__ == "__main__":
    # Run detection test
    detection_ok = test_camera_and_detection()
    
    # Test zone positions
    test_zone_positions()
    
    if not detection_ok:
        print(f"\n RECOMMENDED FIXES:")
        print(f"1. Stand in front of the camera")
        print(f"2. Ensure good lighting")
        print(f"3. Lower confidence threshold in config")
        print(f"4. Adjust zone positions if needed")