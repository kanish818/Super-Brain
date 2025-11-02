import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import serial

# Load the trained model
try:
    model = load_model('eeg_movement_model.h5')
    print("Model loaded successfully.")
except OSError as e:
    print(f"Error loading model: {e}")
    exit(1)

# Define the mapping for output classes
class_mapping = {0: 'Forward', 1: 'Backward', 2: 'Stop'}

# Initialize a buffer to store recent EEG data
sequence_length = 10  # Length of the sequence required by the model
data_buffer = []
scaler = StandardScaler()

# Preprocess data for model input
def preprocess_streamed_data(data_buffer, sequence_length):
    try:
        # Ensure enough data points are collected
        if len(data_buffer) >= sequence_length:
            # Select the latest sequence
            sequence = np.array(data_buffer[-sequence_length:])
            sequence = scaler.fit_transform(sequence.reshape(-1, 1))  # Scale data
            sequence = sequence.reshape(1, sequence_length, 1)  # Reshape for model
            return sequence
    except Exception as e:
        print(f"Error during data preprocessing: {e}")
    return None

# Receive data from serial in real time
def receive_and_predict(serial_port='COM3', baud_rate=230400):
    try:
        ser = serial.Serial(serial_port, baud_rate)
        time.sleep(2)  # Wait for Arduino to reset and start sending data
        print("Serial port initialized. Listening for data...")
        
        while True:
            # Read the first byte, which includes the start of frame marker and MSBs
            msb_byte = ser.read(1)
            if not msb_byte:
                continue  # Skip if no data received
            
            # Extract the MSB
            msb = ord(msb_byte) & 0x7F  # Remove the start of frame marker (MSB bit)
            
            # Read the second byte, which includes the LSBs
            lsb_byte = ser.read(1)
            if not lsb_byte:
                continue  # Skip if no data received
            
            # Extract the LSB
            lsb = ord(lsb_byte) & 0x7F  # Only consider the 7 LSB bits

            # Reconstruct the original analog value
            value = (msb << 7) | lsb  # Combine MSB and LSB

            # Add value to the buffer
            data_buffer.append(value)

            # Maintain buffer size
            if len(data_buffer) > sequence_length:
                data_buffer.pop(0)

            # Preprocess data if enough samples are collected
            input_data = preprocess_streamed_data(data_buffer, sequence_length)
            if input_data is not None:
                # Perform prediction
                prediction = model.predict(input_data, verbose=0)
                predicted_index = np.argmax(prediction)  # Get predicted class index
                predicted_label = class_mapping.get(predicted_index, "Unknown")  # Map index to label
                print(f"Predicted Label: {predicted_label}")
            
            # Simulate real-time delay for prediction
            time.sleep(1)

    except KeyboardInterrupt:
        print("Real-time prediction stopped by user.")
    except serial.SerialException as e:
        print(f"Serial communication error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    receive_and_predict()
