import os
import sys
import urllib.request
import cv2

# Add src to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from detector import MediaPipeFaceMeshDetector
from ear import calculate_avg_ear, calculate_ear

def main():
    print("--- Running Visual Detector Test ---")
    
    # 1. Download sample portrait image from Wikimedia Commons
    portrait_url = "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg"
    image_path = "portrait.jpg"
    output_path = "result_portrait.jpg"
    
    if not os.path.exists(image_path):
        print(f"Downloading sample portrait image from {portrait_url}...")
        try:
            req = urllib.request.Request(
                portrait_url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req) as response, open(image_path, 'wb') as out_file:
                out_file.write(response.read())
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download test image: {e}")
            return
            
    # 2. Load image using OpenCV
    frame = cv2.imread(image_path)
    if frame is None:
        print("Failed to load downloaded image.")
        return
        
    print(f"Image loaded successfully. Size: {frame.shape[1]}x{frame.shape[0]}")
    
    # 3. Initialize detector
    detector = MediaPipeFaceMeshDetector()
    
    # 4. Detect eye landmarks
    print("Running face mesh landmark detection...")
    result = detector.detect_eye_landmarks(frame)
    
    if result is None:
        print("No face detected in the image!")
        detector.close()
        return
        
    left_eye, right_eye = result
    print(f"Detected {len(left_eye)} points for Left Eye and {len(right_eye)} points for Right Eye.")
    print("Left eye points:", left_eye)
    print("Right eye points:", right_eye)
    
    # 5. Calculate EAR
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    avg_ear = calculate_avg_ear(left_eye, right_eye)
    
    print(f"Left Eye EAR: {left_ear:.4f}")
    print(f"Right Eye EAR: {right_ear:.4f}")
    print(f"Average EAR: {avg_ear:.4f}")
    
    # 6. Draw landmarks on the image to visually verify
    output_frame = frame.copy()
    
    # Draw Left Eye (Green circles)
    for pt in left_eye:
        cv2.circle(output_frame, pt, 2, (0, 255, 0), -1)
        
    # Draw Right Eye (Blue circles)
    for pt in right_eye:
        cv2.circle(output_frame, pt, 2, (255, 0, 0), -1)
        
    # Add EAR text on the image
    cv2.putText(
        output_frame, 
        f"Avg EAR: {avg_ear:.4f}", 
        (30, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1.0, 
        (0, 0, 255), 
        2
    )
    
    # Save the output image
    cv2.imwrite(output_path, output_frame)
    print(f"Saved visual verification result to: {output_path}")
    
    detector.close()
    print("Test finished successfully!")

if __name__ == "__main__":
    main()
