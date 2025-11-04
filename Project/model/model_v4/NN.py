import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Activation
from tensorflow.keras.utils import get_custom_objects


def swish(x):
    return x * tf.keras.backend.sigmoid(x)

get_custom_objects().update({'swish': Activation(swish)})

# Function to load the CSV data
def load_data(file_path):
    # Load data using pandas to handle CSV with headers
    data = pd.read_csv(file_path)
    # print(data.describe)
    # Extract 'Value' as features and 'Result' as labels
    X = data['Value'].values.reshape(-1, 1)  # Reshaping as it is a single feature
    y = data['Result'].values  # Labels are in 'Result'
    # print(y.info)
    return X, y


# Preprocessing data: scaling and encoding labels
def preprocess_data(X, y):
    # Standardizing 'Value' column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshaping to fit the Conv1D layer input requirement
    X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Encoding the labels ('Result')
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)  # Convert to one-hot encoding
    
    return X_reshaped, y_categorical, label_encoder

# Function to create sequences from the dataset
def create_sequences(X, y, sequence_length=10):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])  # Corresponding label for the sequence
    return np.array(X_seq), np.array(y_seq)

# Load and preprocess data
X, y = load_data('../../../Final_clean.csv')
X, y, label_encoder = preprocess_data(X, y)

# Create sequences
sequence_length = 10  # You can adjust this length as needed
X_seq, y_seq = create_sequences(X, y, sequence_length=sequence_length)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the model
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dropout(0.3))
# model.add(LSTM(64, return_sequences=True))
# model.add(LSTM(64))
# model.add(Flatten())
# model.add(Dense(128, activation='swish'))
# model.add(Dense(64, activation='swish'))
# model.add(Dense(32, activation='swish'))
# model.add(Dropout(0.5))
# model.add(Dense(3, activation='softmax'))  
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='swish', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(LSTM(64, return_sequences=True, activation='swish'))
model.add(LSTM(64, activation='swish'))
model.add(Flatten())
model.add(Dense(128, activation='swish'))
model.add(Dense(64, activation='swish'))
model.add(Dense(32, activation='swish'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax')) 

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.2f}")

# Save the model
model.save('eeg_movement_model.h5')
