import dlib
import cv2
import matplotlib.pyplot as plt
from google.colab import files
import numpy as np

# Load the pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/content/shape_predictor_68_face_landmarks.dat") # You need to download this file

# Upload image file
uploaded = files.upload()

# Check if image is uploaded
if len(uploaded.keys()) == 0:
    print("Error: No image uploaded.")
else:
    # Get the uploaded image file
    file_name = list(uploaded.keys())[0]
    image = cv2.imdecode(np.frombuffer(uploaded[file_name], np.uint8), -1)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        # Predict facial landmarks
        landmarks = predictor(gray, face)

        # Iterate over the facial landmarks
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y

            # Draw a circle on each facial landmark
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Display the output image with facial landmarks using matplotlib
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Facial Landmarks")
    plt.axis('off')
    plt.show()
