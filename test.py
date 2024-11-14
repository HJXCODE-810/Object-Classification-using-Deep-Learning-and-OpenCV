import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:/Users/lenovo/object_classifier_mv2.h5')

# Define class names based on the order in which ImageDataGenerator loads them
class_names = ['CAP', 'Earpod', 'Key', 'motor']  # Replace with actual class names as in your folder structure

# Set parameters
image_height, image_width = 128, 128  # Same as used during training
confidence_threshold = 0.7  # Confidence threshold for drawing bounding box
# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for prediction
    input_image = cv2.resize(frame, (image_width, image_height))
    input_image = input_image.astype('float32') / 255.0  # Rescale
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Get class name and confidence
    class_name = class_names[predicted_class]
    confidence = np.max(predictions)

     # If confidence is higher than the threshold, draw a bounding box
    if confidence > confidence_threshold:
        # Display the confidence and class name
        text = f"{class_name} ({confidence:.2f})"
        
        # Draw bounding box
        h, w, _ = frame.shape
        cv2.rectangle(frame, (10, 10), (w - 10, h - 10), (0, 255, 0), 2)
        cv2.putText(frame, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the result
    cv2.imshow('Object Classification BY HJX', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
