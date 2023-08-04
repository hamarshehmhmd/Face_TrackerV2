import cv2
import numpy as np
import os
import face_recognition

# Load the cascade for face detection
username = ""
desktop_path = os.path.join("/Users", username, "Desktop")
cascade_file_path_face = os.path.join(desktop_path, "haarcascade_frontalface_default.xml")
cascade_file_path_eye = os.path.join(desktop_path, "haarcascade_eye.xml")

face_cascade = cv2.CascadeClassifier(cascade_file_path_face)

# Load the cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cascade_file_path_eye)

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Load the reference image from the desktop for face matching
reference_image_path = os.path.join(desktop_path, "Image name goes here")
reference_image = face_recognition.load_image_file(reference_image_path)
reference_encoding = face_recognition.face_encodings(reference_image)[0]

while True:
    # Read the frame from the video capture object
    ret, frame = cap.read()

    # Check if the frame was successfully read
    if not ret:
        print("Error: Failed to read frame from video capture object")
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Detect eyes in the face region of interest
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Loop through the detected eyes
        for (ex, ey, ew, eh) in eyes:
            # Draw a rectangle around the eyes
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Check if the face matches the reference image
        face_encoding = face_recognition.face_encodings(frame, [(y, x + w, y + h, x)])[0]
        match = face_recognition.compare_faces([reference_encoding], face_encoding)[0]

        if match:
            # Draw a box around the matched face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Write "Identity Match" above the face
            cv2.putText(frame, 'Identity Match', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # Write "Human" above the face
            cv2.putText(frame, 'Human', (x, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face and Eye Detection', frame)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
