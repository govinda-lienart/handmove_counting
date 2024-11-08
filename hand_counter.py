import cv2
import mediapipe as mp

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for default camera

cross_count = 0
line_y_position = 200  # Set your virtual line position (in pixels)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(image_rgb)

    # Draw the line
    cv2.line(image, (0, line_y_position), (640, line_y_position), (255, 0, 0), 2)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Check if the wrist landmark crosses the line
            wrist_y = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image.shape[0]
            if wrist_y < line_y_position:  # Hand is above the line
                cross_count += 1
                print("Hand crossed the line! Count:", cross_count)

    # Display the resulting image
    cv2.imshow('Hand Tracking', image)

    if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()