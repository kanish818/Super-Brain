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
    try:
        msb_byte = ser_input.read(1)
        lsb_byte = ser_input.read(1)
        msb = ord(msb_byte) & 0x7F
        lsb = ord(lsb_byte) & 0x7F
        value = (msb << 7) | lsb
        return value
    except Exception as e:
        print(f"Error reading value: {e}")
        return None

while True:
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
            print(features)
            eeg_data['C3'] = eeg_data['C3'][1:]
            eeg_data['Cz'] = eeg_data['Cz'][1:]
            eeg_data['C4'] = eeg_data['C4'][1:]
