"""
Eye Aspect Ratio (EAR) Detection Module
This module calculates the Eye Aspect Ratio to determine if eyes are open or closed
"""

import numpy as np

def euclidean(p1, p2):
    """Compute Euclidean distance between two points"""
    return np.linalg.norm(np.array(p1) - np.array(p2))

class EyeAspectRatioDetector:
    def init(self, ear_threshold=0.25, ear_consec_frames=20):
        """
        Initialize the EAR detector
        
        Args:
            ear_threshold (float): EAR threshold below which eyes are considered closed
            ear_consec_frames (int): Number of consecutive frames below threshold to trigger drowsiness
        """
        self.EAR_THRESHOLD = ear_threshold
        self.EAR_CONSEC_FRAMES = ear_consec_frames
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0
        
        # Define the facial landmarks for the left and right eye (dlib 68-point model)
        self.LEFT_EYE_START = 42
        self.LEFT_EYE_END = 48
        self.RIGHT_EYE_START = 36
        self.RIGHT_EYE_END = 42
        
    def calculate_ear(self, eye_landmarks):
        """
        Calculate the Eye Aspect Ratio (EAR) for given eye landmarks
        
        Args:
            eye_landmarks: Array of (x, y) coordinates for eye landmarks
            
        Returns:
            float: Eye Aspect Ratio value
        """
        # Compute the euclidean distances between the two sets of vertical eye landmarks
        A = euclidean(eye_landmarks[1], eye_landmarks[5])
        B = euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Prevent division by zero
        if C == 0:
            return 0.0
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_landmarks(self, landmarks):
        """
        Extract left and right eye landmarks from facial landmarks
        
        Args:
            landmarks: Complete facial landmarks array
            
        Returns:
            tuple: (left_eye_landmarks, right_eye_landmarks)
        """
        left_eye = landmarks[self.LEFT_EYE_START:self.LEFT_EYE_END]
        right_eye = landmarks[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
        return left_eye, right_eye
    
    def detect_drowsiness(self, landmarks):
        """
        Detect drowsiness based on eye aspect ratio
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            tuple: (is_drowsy, ear_left, ear_right, blink_counter)
        """
        left_eye, right_eye = self.extract_eye_landmarks(landmarks)
        
        # Calculate EAR for both eyes
        left_ear = self.calculate_ear(left_eye)
        right_ear = self.calculate_ear(right_eye)
        
        # Average the eye aspect ratio for both eyes
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Check if EAR is below threshold
        if avg_ear < self.EAR_THRESHOLD:
            self.COUNTER += 1
            
            # If eyes have been closed for sufficient number of frames
            if self.COUNTER >= self.EAR_CONSEC_FRAMES:
                return True, left_ear, right_ear, self.COUNTER
        else:
            # If eyes are open, reset counter and increment blink count
            if self.COUNTER >= self.EAR_CONSEC_FRAMES:
                self.TOTAL_BLINKS += 1
            self.COUNTER = 0
            
        return False, left_ear, right_ear, self.COUNTER
    
    def reset_counters(self):
        """Reset all counters"""
        self.COUNTER = 0
        self.TOTAL_BLINKS = 0
    
    def get_blink_count(self):
        """Get total blink count"""
        return self.TOTAL_BLINKS
    
    def update_threshold(self, new_threshold):
        """Update EAR threshold"""
        self.EAR_THRESHOLD = new_threshold
        
    def update_consec_frames(self, new_frames):
        """Update consecutive frames threshold"""
        self.EAR_CONSEC_FRAMES = new_frames


import cv2
import mediapipe as mp
import numpy as np
from code import EyeAspectRatioDetector  # import your EAR class    

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Initialize EAR detector
detector = EyeAspectRatioDetector()

# Start webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Convert landmarks to numpy array
            h, w, _ = frame.shape
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) 
                                   for lm in face_landmarks.landmark])
            
            # Run drowsiness detection
            is_drowsy, left_ear, right_ear, counter = detector.detect_drowsiness(landmarks)
            
            # Display EAR on frame
            avg_ear = (left_ear + right_ear) / 2.0
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if is_drowsy:
                cv2.putText(frame, "⚠️ DROWSINESS ALERT!", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()