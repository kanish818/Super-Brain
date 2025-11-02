import serial
import numpy as np
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

ser = serial.Serial('/dev/ttyACM0', 9600)

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

eeg_data = { 'C3': [], 'Cz': [], 'C4': [] }

while True:
    if ser.in_waiting > 0:
        line = ser.readline().decode().strip()
        signalC3, signalCz, signalC4 = map(int, line.split(","))
        eeg_data['C3'].append(signalC3)
        eeg_data['Cz'].append(signalCz)
        eeg_data['C4'].append(signalC4)

        if len(eeg_data['C3']) >= 256: 
            data_array = [np.array(eeg_data['C3']), np.array(eeg_data['Cz']), np.array(eeg_data['C4'])]
            features = extract_features(data_array)
            print(features)  
            eeg_data['C3'] = eeg_data['C3'][1:]
            eeg_data['Cz'] = eeg_data['Cz'][1:]
            eeg_data['C4'] = eeg_data['C4'][1:]
