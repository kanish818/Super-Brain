import serial
import time
import csv

def receive_data(serial_port='COM3', baud_rate=230400, output_file='eeg_data_realtime.csv'):
    ser = serial.Serial(serial_port, baud_rate)
    time.sleep(2)  # Wait for Arduino to reset and start sending data
    
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Timestamp', 'Value'])  # Write the header
        
        while True:
            try:
                msb_byte = ser.read(1)
                lsb_byte = ser.read(1)
                msb = ord(msb_byte) & 0x7F
                lsb = ord(lsb_byte) & 0x7F
                value = (msb << 7) | lsb
                timestamp = time.time()
                csvwriter.writerow([timestamp, value])
                print(f"Received value: {value} at {timestamp}")
            except Exception as e:
                print(f"Error receiving data: {e}")

if __name__ == "__main__":
    receive_data()
