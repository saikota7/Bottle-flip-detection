import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist

# --- Step 1: Setup Environment ---
# Initialize Mediapipe for Hand Detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for webcam

# --- Step 2: Define Utility Functions ---
def detect_bottle(frame):
    """
    Detects the bottle based on distinguishable features.
    Returns bounding box (x, y, w, h) of the bottle if found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Assuming the bottle cap has a unique color (e.g., red)
    lower_red = np.array([0, 120, 70])  # Adjust based on bottle cap color
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours of the bottle cap
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if cv2.contourArea(largest) > 500:  # Filter small false detections
            return x, y, w, h
    return None

def analyze_trajectory(positions):
    """
    Checks if the trajectory matches a flip.
    Parameters:
    - positions: List of (x, y) tuples of the bottle over frames.
    """
    if len(positions) < 5:  # Not enough data for trajectory analysis
        return False

    diffs = [positions[i + 1][1] - positions[i][1] for i in range(len(positions) - 1)]
    upward_motion = all(d > 0 for d in diffs[:len(diffs)//2])
    downward_motion = all(d < 0 for d in diffs[len(diffs)//2:])
    return upward_motion and downward_motion

def check_upright(frame, bbox):
    """
    Checks if the bottle is upright (based on cap/base orientation).
    """
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Check symmetry or specific patterns (adjust as needed for the bottle design)
    top_section = edges[0:h//2, :]
    bottom_section = edges[h//2:, :]
    return np.sum(top_section) < np.sum(bottom_section)

# --- Step 3: Main Loop ---
hand_positions = []  # Track positions of hands for cheating detection
bottle_positions = []  # Track positions of the bottle for trajectory analysis

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)
                hand_positions.append((hand_lm.landmark[0].x, hand_lm.landmark[0].y))
        
        # Detect bottle
        bottle_bbox = detect_bottle(frame)
        if bottle_bbox:
            x, y, w, h = bottle_bbox
            bottle_positions.append((x + w//2, y + h//2))  # Center of bottle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Analyze trajectory
        if len(bottle_positions) > 10:  # Check after sufficient data points
            if analyze_trajectory(bottle_positions):
                cv2.putText(frame, "Flip Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            bottle_positions = []  # Reset for the next flip
        
        # Check cheating (e.g., hand proximity)
        if hand_positions:
            last_hand_pos = hand_positions[-1]
            if len(bottle_positions) > 0 and dist.euclidean(last_hand_pos, bottle_positions[-1]) < 50:
                cv2.putText(frame, "Cheating Detected!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show result
        cv2.imshow("Bottle Flip Game", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
