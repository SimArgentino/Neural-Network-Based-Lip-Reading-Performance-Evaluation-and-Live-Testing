import os
import cv2  # OpenCV for video capturing and image processing
import dlib  # Library for facial landmark detection
import math  # For mathematical operations
import json  # For JSON operations (though not used directly here)
import statistics  # For statistical operations (though not used directly here)
from PIL import Image  # Python Imaging Library for image operations
import imageio.v2 as imageio  # Image input/output library
import numpy as np  # For numerical operations and array handling
import csv  # For CSV file operations (though not used directly here)
from collections import deque  # Deque for efficiently adding/removing elements from both ends
import tensorflow as tf  # TensorFlow for deep learning models
import sys  # System-specific parameters and functions

from constants import TOTAL_FRAMES, VALID_WORD_THRESHOLD, NOT_TALKING_THRESHOLD, PAST_BUFFER_SIZE, LIP_WIDTH, LIP_HEIGHT

# Add the data directory to the system path
sys.path.append('../data')


# Dictionary mapping class indices to words
label_dict = {
    0: 'auto',
    1: 'cane',
    2: 'ciao',
    3: 'demo',
    4: 'gatto',
    5: 'moto',
    6: 'ok',
    7: 'pasta',
    8: 'pizza',
    9: 'uova'
}

# Define the input shape for the model
input_shape = (TOTAL_FRAMES, 80, 112, 3)

# Define the deep learning model architecture using Keras Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(8, (3, 3, 3), activation='relu', input_shape=input_shape),  # 3D Convolution layer
    tf.keras.layers.MaxPooling3D((2, 2, 2)),  # Max pooling layer to reduce dimensionality
    tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu'),  # 2nd 3D Convolution layer
    tf.keras.layers.MaxPooling3D((2, 2, 2)),  # 2nd Max pooling layer
    tf.keras.layers.Conv3D(256, (3, 3, 3), activation='relu'),  # 3rd 3D Convolution layer
    tf.keras.layers.Flatten(),  # Flatten the output to feed into dense layers
    tf.keras.layers.Dense(1024, activation='relu'),  # Fully connected (dense) layer with ReLU activation
    tf.keras.layers.Dropout(0.5),  # Dropout layer for regularization
    tf.keras.layers.Dense(256, activation='relu'),  # 2nd Dense layer
    tf.keras.layers.Dropout(0.5),  # Dropout layer for regularization
    tf.keras.layers.Dense(64, activation='relu'),  # 3rd Dense layer
    tf.keras.layers.Dropout(0.5),  # Dropout layer for regularization
    tf.keras.layers.Dense(len(label_dict), activation='softmax')  # Output layer with softmax activation
])

# Load the pre-trained weights into the model
model.load_weights(r'C:\Users\simon\Desktop\Personal_dataset_build_livetest - Copia\PesiRete_1.h5')

# Load the facial detector from dlib
detector = dlib.get_frontal_face_detector()

# Load the shape predictor for facial landmark detection
predictor = dlib.shape_predictor(r"C:\Users\simon\Desktop\Personal_dataset_build_livetest - Copia\face_weights.dat")

# Initialize video capture from the default webcam (index 0)
cap = cv2.VideoCapture(0)

# Variables to track frames and prediction states
curr_word_frames = []  # List to store frames for the current word
not_talking_counter = 0  # Counter to track non-talking frames
first_word = True  # Flag for the first word detection
labels = []  # List to store detected labels (though not used directly here)
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)  # Deque to store past frames
ending_buffer_size = 5  # Buffer size to handle ending frames
predicted_word_label = None  # Variable to store the predicted word label
draw_prediction = False  # Flag to indicate if the prediction should be drawn
spoken_already = []  # List to keep track of already spoken words
count = 0  # Counter for displaying predictions

# Main loop to process video frames
while True:
    _, frame = cap.read()  # Read a frame from the webcam
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

    # Detect faces in the grayscale frame
    faces = detector(gray)
    
    for face in faces:
        # Get the coordinates of the face region
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Get the facial landmarks for the detected face
        landmarks = predictor(image=gray, box=face)

        # Calculate the distance between the upper and lower lip landmarks
        mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
        mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
        lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])

        # Get coordinates for the lip region
        lip_left = landmarks.part(48).x
        lip_right = landmarks.part(54).x
        lip_top = landmarks.part(50).y
        lip_bottom = landmarks.part(58).y

        # Calculate padding to maintain a consistent lip region size
        width_diff = LIP_WIDTH - (lip_right - lip_left)
        height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
        pad_left = width_diff // 2
        pad_right = width_diff - pad_left
        pad_top = height_diff // 2
        pad_bottom = height_diff - pad_top

        # Ensure padding does not exceed frame boundaries
        pad_left = min(pad_left, lip_left)
        pad_right = min(pad_right, frame.shape[1] - lip_right)
        pad_top = min(pad_top, lip_top)
        pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

        # Extract and resize the lip region with padding
        lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
        lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))

        # Convert the lip region to LAB color space for contrast enhancement
        lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3,3))
        l_channel_eq = clahe.apply(l_channel)

        # Merge the equalized L channel back with A and B channels
        lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
        lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
        
        # Apply Gaussian blur and bilateral filter for noise reduction
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
        lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
        
        # Apply sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
        lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
        
        # Update lip frame with the processed version
        lip_frame = lip_frame_eq
        
        # Draw landmarks around the mouth for visualization
        for n in range(48, 61):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img=frame, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # Check if the person is talking based on lip distance
        if lip_distance > 38:  # Threshold to determine if the person is talking
            cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            curr_word_frames += [lip_frame.tolist()]  # Add lip frame to current word frames
            not_talking_counter = 0  # Reset the not-talking counter
            draw_prediction = False  # Reset the draw prediction flag

        else:
            cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            not_talking_counter += 1  # Increment the not-talking counter

            if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES:
                # If enough non-talking frames are detected and we have enough frames for a prediction
                curr_word_frames = list(past_word_frames) + curr_word_frames

                # Prepare the frames for prediction
                curr_data = np.array([curr_word_frames[:input_shape[0]]])

                # Print the shape of the current data
                print("*********", curr_data.shape)
                print(spoken_already)

                # Predict the word
                prediction = model.predict(curr_data)

                # Get the probabilities for each class
                prob_per_class = []
                for i in range(len(prediction[0])):
                    prob_per_class.append((prediction[0][i], label_dict[i]))
                
                # Sort the probabilities
                sorted_probs = sorted(prob_per_class, key=lambda x: x[0], reverse=True)
                
                # Print the probabilities for each label
                for prob, label in sorted_probs:
                    print(f"{label}: {prob:.3f}")

                # Get the predicted class index and label
                predicted_class_index = np.argmax(prediction)
                predicted_word_label = label_dict[predicted_class_index]
                spoken_already.append(predicted_word_label)  # Add to spoken words

                print("FINISHED!", predicted_word_label)
                draw_prediction = True  # Set the flag to draw the prediction
                count = 0  # Reset the count for displaying prediction

                # Clear the current word frames and reset the counters
                curr_word_frames = []
                not_talking_counter = 0

            elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:
                # Continue collecting frames for the current word
                curr_word_frames += [lip_frame.tolist()]
                not_talking_counter = 0

            elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                # If not enough valid frames, clear the current word frames
                curr_word_frames = []

            # Add the current frame to the past frames buffer
            past_word_frames += [lip_frame.tolist()]
            if len(past_word_frames) > PAST_BUFFER_SIZE:
                past_word_frames.pop(0)

    # Display the predicted word if available
    if draw_prediction and count < 20:
        count += 1
        cv2.putText(frame, predicted_word_label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

    # Show the frame with annotations
    cv2.imshow(winname="Mouth", mat=frame)

    # Check for key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        spoken_already = []  # Clear the spoken words list if 'q' is pressed

    # Exit the loop when 'ESC' key is pressed
    if key == 27:
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
