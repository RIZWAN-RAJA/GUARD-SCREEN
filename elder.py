import cv2
import numpy as np

def detect_faces(image):
    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)

    return faces

def determine_person_type(face):
    # Based on face size, determine if the person is a child or an adult
    x, y, w, h = face
    face_size = w * h

    if face_size < 20000:  # Adjust this threshold based on your observations
        return "child"
    else:
        return "adult"

def blur_screen(image, faces):
    for face in faces:
        person_type = determine_person_type(face)

        # Apply a blur effect to the entire screen if a child is detected
        if person_type == "child":
            image = cv2.GaussianBlur(image, (99, 99), 30)
           
    return image

def main():
    # Open the webcam (you can replace this with your camera source)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Detect faces in the frame
        faces = detect_faces(frame)

        # Check if faces are detected
        if len(faces) > 0:
            # Blur the screen if a child is detected
            frame = blur_screen(frame, faces)

        # Display the resulting frame
        cv2.imshow('Blurry Screen for Kids', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
