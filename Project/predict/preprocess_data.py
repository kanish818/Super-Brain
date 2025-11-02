# import serial
# import numpy as np
# from scipy.signal import butter, lfilter

# ser_input = serial.Serial('COM3', 230400)

# def butter_bandpass(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='band')
#     return b, a

# def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256.0, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

# def extract_features(data):
#     features = []
#     for channel_data in data:
#         alpha_band = bandpass_filter(channel_data, 8, 12)
#         alpha_power = np.mean(np.square(alpha_band))
#         features.append(alpha_power)
#     return np.array(features)

# eeg_data = {'C3': [], 'Cz': [], 'C4': []}

# while True:
#     if ser_input.in_waiting > 0:
#         line = ser_input.readline().strip()

#         # Decode the line if it's a bytes object
#         if isinstance(line, bytes):
#             try:
#                 line = line.decode('utf-8', errors='ignore')  # Ignore problematic bytes
#             except UnicodeDecodeError:
#                 print(f"Could not decode line: {line}")
#                 continue

#         try:
#             # Split the line into signals and convert them to integers
#             signalC3, signalCz, signalC4 = map(int, line.split(","))
#             eeg_data['C3'].append(signalC3)
#             eeg_data['Cz'].append(signalCz)
#             eeg_data['C4'].append(signalC4)

#             if len(eeg_data['Cz']) >= 256:
#                 data_array = [np.array(eeg_data['C3']), np.array(eeg_data['Cz']), np.array(eeg_data['C4'])]
#                 features = extract_features(data_array)
#                 print(features)
                
#                 # Trim the data to keep the buffer size constant
#                 eeg_data['C3'] = eeg_data['C3'][1:]
#                 eeg_data['Cz'] = eeg_data['Cz'][1:]
#                 eeg_data['C4'] = eeg_data['C4'][1:]

#         except ValueError:
#             print(f"Error processing line: {line}")




import serial
import numpy as np
from scipy.signal import butter, lfilter
import time

ser_input = serial.Serial('COM3', 230400)
time.sleep(2)  # Wait for Arduino to reset and start sending data

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.5, highcut=50.0, fs=256.0, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def extract_features(data):
    features = []
    for channel_data in data:
        alpha_band = bandpass_filter(channel_data, 8, 12)
        alpha_power = np.mean(np.square(alpha_band))
        features.append(alpha_power)
    return np.array(features)

eeg_data = {'C3': [], 'Cz': [], 'C4': []}

def read_value():
    # Read the first byte (MSB)
    msb_byte = ser_input.read(1)
    if not msb_byte:
        return None  # Skip if no data received
    
    # Read the second byte (LSB)
    lsb_byte = ser_input.read(1)
    if not lsb_byte:
        return None  # Skip if no data received
    
    # Combine MSB and LSB to get the original value
    msb = ord(msb_byte) & 0x7F
    lsb = ord(lsb_byte) & 0x7F
    value = (msb << 7) | lsb
    
    return value

while True:
    # Collect data for each channel
    signalC3 = read_value()
    signalCz = read_value()
    signalC4 = read_value()

    if signalC3 is not None and signalCz is not None and signalC4 is not None:
        eeg_data['C3'].append(signalC3)
        eeg_data['Cz'].append(signalCz)
        eeg_data['C4'].append(signalC4)

        if len(eeg_data['Cz']) >= 256:
            data_array = [np.array(eeg_data['C3']), np.array(eeg_data['Cz']), np.array(eeg_data['C4'])]
            features = extract_features(data_array)
            # print(features)
            
            # Trim the data to keep the buffer size constant
            eeg_data['C3'] = eeg_data['C3'][1:]
            eeg_data['Cz'] = eeg_data['Cz'][1:]
            eeg_data['C4'] = eeg_data['C4'][1:]
