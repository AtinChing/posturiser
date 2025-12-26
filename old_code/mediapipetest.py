import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Tuple
from enum import Enum
import math
cap = cv2.VideoCapture(0)
# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

class Landmark(Enum):
    from enum import Enum

class Landmark(Enum):
    NOSE = 0
    EYE_LEFT_INNER = 1
    EYE_LEFT = 2
    EYE_LEFT_OUTER = 3
    EYE_RIGHT_INNER = 4
    EYE_RIGHT = 5
    EYE_RIGHT_OUTER = 6
    EAR_LEFT = 7
    EAR_RIGHT = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    SHOULDER_LEFT = 11
    SHOULDER_RIGHT = 12
    ELBOW_LEFT = 13
    ELBOW_RIGHT = 14
    WRIST_LEFT = 15
    WRIST_RIGHT = 16
    PINKY_LEFT = 17
    PINKY_RIGHT = 18
    INDEX_LEFT = 19
    INDEX_RIGHT = 20
    THUMB_LEFT = 21
    THUMB_RIGHT = 22
    HIP_LEFT = 23
    HIP_RIGHT = 24
    KNEE_LEFT = 25
    KNEE_RIGHT = 26
    ANKLE_LEFT = 27
    ANKLE_RIGHT = 28
    HEEL_LEFT = 29
    HEEL_RIGHT = 30
    FOOT_INDEX_LEFT = 31
    FOOT_INDEX_RIGHT = 32
    HAND_LEFT = 15  # Same as WRIST_LEFT
    HAND_RIGHT = 16  # Same as WRIST_RIGHT

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.9
)
mp_draw = mp.solutions.drawing_utils
HAND_RIGHT = Landmark.WRIST_RIGHT # hand right is actually going to be the right wrist
HAND_LEFT = Landmark.WRIST_LEFT # hand left is actually going to be the left wrist, can play around and change wrist to one of the fingers later
VISIBILITY_THRESHOLD = 0.6
SHOULDER_DISTANCE_THRESHOLD = 0.15

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Display the frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    landmarks = results.pose_landmarks
            
    if landmarks:
        default_colour = (245, 117, 66)
            
        # drawing jackshit
        connection_drawing_spec = mp_draw.DrawingSpec(
            color=default_colour,
            thickness=2
        )

        landmark_drawing_spec = mp_draw.DrawingSpec(
            color=default_colour,
            thickness=1,
            circle_radius=2
        )
        mp_draw.draw_landmarks(
            frame,
            landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec,
            connection_drawing_spec
        )
        # Extract key landmarks
        nose = landmarks.landmark[Landmark.NOSE.value]
        left_shoulder = landmarks.landmark[Landmark.SHOULDER_LEFT.value]
        right_shoulder = landmarks.landmark[Landmark.SHOULDER_RIGHT.value]
        left_hip = landmarks.landmark[Landmark.HIP_LEFT.value]
        right_hip = landmarks.landmark[Landmark.HIP_RIGHT.value]
        right_hand = landmarks.landmark[Landmark.HAND_RIGHT.value]
        left_hand = landmarks.landmark[Landmark.HAND_LEFT.value]
        right_elbow = landmarks.landmark[Landmark.ELBOW_RIGHT.value]
        left_elbow = landmarks.landmark[Landmark.ELBOW_LEFT.value]
        #shoulder_higher = right_shoulder.y < right_elbow.y
        print("RIGHT SHOULDER", right_shoulder.y)
        print("LEFT SHOULDER", left_shoulder.y)
    
    cv2.imshow('Pose Detection', frame)

    ## Write the frame if output path is specified
    #if output_path:
    #    out.write(frame)
        
    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Convert the frame to RGB

        

## Initialize video writer if output path is specified
#if output_path:
#    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


# Clean up
cap.release()
#if output_path:
#    out.release()
cv2.destroyAllWindows()