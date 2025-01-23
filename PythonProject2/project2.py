import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load your trained model (replace with the correct path to your model)
model_path = "action_recognition_model.h5"
model = load_model(model_path)

# Actions corresponding to the model's output (adjust based on your model's output classes)
actions = ["walk", "run", "jump", "wave"]  # Adjust based on your model's classes

# Load the video generated from the Weizmann dataset (make sure this file path is correct)
video_path = "weizmann_actions_video.avi"
cap = cv2.VideoCapture(video_path)

# Check if the video was successfully opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to match the model's input size (e.g., 128x128 or 224x224, based on your model's input)
    frame_resized = cv2.resize(frame, (128, 128))

    # Normalize the frame and expand the dimensions to match the input shape for the model
    frame_expanded = np.expand_dims(frame_resized, axis=0) / 255.0

    # Predict the action for the current frame
    prediction = model.predict(frame_expanded)
    predicted_class = np.argmax(prediction)

    # Get the action label corresponding to the predicted class
    predicted_action = actions[predicted_class]

    # Debugging: Print the prediction value to check if it is correct
    print(f"Prediction: {prediction} -> Predicted Action: {predicted_action}")

    # Draw the predicted action label on the frame
    cv2.putText(frame, f"Action: {predicted_action}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Resize the frame for display to 1366×768
    frame_display = cv2.resize(frame, (1366, 768))
    cv2.imshow("Action Recognition Video", frame_display)

    # Exit the video playback when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()