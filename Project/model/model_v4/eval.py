import time
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import pandas as pd
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

# Load the example dataset
try:
    example_dataset = pd.read_csv(r"Latest\Project\Final.csv")  # Replace with the actual file path
    example_dataset['Timestamp'] = pd.to_datetime(example_dataset['Timestamp'], unit='s')  # Convert timestamp
    print("Loaded example dataset.")
except FileNotFoundError as e:
    print(f"Dataset file not found: {e}")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

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

# Simulate real-time predictions using the dataset
try:
    for _, row in example_dataset.iterrows():
        try:
            # Extract data row
            timestamp = row['Timestamp']
            value = row['Value']
            expected_result = row.get('Result', None)  # Optional 'Result' column

            print(f"Timestamp: {timestamp}, Value: {value}, Expected: {expected_result}")

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
                print(f"Predicted Label: {predicted_label}, Expected Label: {expected_result}")
            
            # Simulate real-time delay
            time.sleep(0.1)
        except KeyError as e:
            print(f"Missing data in row: {e}")
        except Exception as e:
            print(f"Error processing row: {e}")
except KeyboardInterrupt:
    print("Simulation stopped by user.")
except Exception as e:
    print(f"Unexpected error during simulation: {e}")