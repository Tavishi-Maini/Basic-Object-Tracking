import cv2
import numpy as np

# --- 1. Define HSV color range for the object ---
# Example: red object
lower_color = np.array([0, 120, 70])
upper_color = np.array([10, 255, 255])

# --- 2. Initialize webcam ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. Preprocess frame ---
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- 4. Create mask for the color ---
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # Optional: clean noise
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # --- 5. Find contours ---
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 10:
            # Draw circle + center point
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # --- 6. Show frame ---
    cv2.imshow("Object Tracker", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 7. Cleanup ---
cap.release()
cv2.destroyAllWindows()
