import cv2
import numpy as np
from tensorflow.keras.datasets import mnist
import tensorflow as tf
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for the model
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# Convert labels to categorical
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))



def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (28, 28))  
    normalized = resized / 255.0  
    reshaped = normalized.reshape(1, 28, 28, 1)  # Reshape for model input
    return reshaped

# Open the webcam
cap = cv2.VideoCapture(0)  

roi_start_x, roi_start_y = 1200, 100  
roi_width, roi_height = 500, 500  

print("Press 'q' to quit the application.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    cv2.rectangle(frame, (roi_start_x, roi_start_y), 
                         (roi_start_x + roi_width, roi_start_y + roi_height), 
                         (0, 0, 255), 2) 
    # Preprocess the frame
    roi = frame[roi_start_y:roi_start_y + roi_height, roi_start_x:roi_start_x + roi_width]
    input_data = preprocess_frame(roi)
    # Predict using the model
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability

    # Display the prediction on the live video feed
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"Predicted: {predicted_class}"
    frame_with_text = cv2.putText(frame, text, (10, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the live video feed with predictions
    cv2.imshow('Live Camera Feed', frame_with_text)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
