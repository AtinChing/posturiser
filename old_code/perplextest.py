import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from webcam
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the RGB image
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
        )

        # Calculate angles between landmarks
        left_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y]
        right_ear = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y]
        left_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y]
        right_shoulder = [results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]

        # Calculate angle between left ear, left shoulder, and right shoulder
        angle_left = calculate_angle(left_ear, left_shoulder, right_shoulder)
        
        # Display the angle on the image
        cv2.putText(image, f"Angle: {angle_left}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('MediaPipe Pose', image)

    # Exit on key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
pose.close()
cap.release()
cv2.destroyAllWindows()