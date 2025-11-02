# # import serial
# # import time

# # def receive_data(serial_port='COM3', baud_rate=230400, output_file='eeg_data.txt'):
# #     ser = serial.Serial(serial_port, baud_rate)
# #     time.sleep(2)  # Wait for Arduino to reset and start sending data
# #     with open(output_file, 'w') as f:
# #         while True:
# #             # Read the first byte, which includes the start of frame marker and MSBs
# #             msb_byte = ser.read(1)
# #             if not msb_byte:
# #                 continue  # Skip if no data received
            
# #             # Check if it's the start of a new frame (MSB of this byte should be 1)
# #             msb = ord(msb_byte) & 0x7F  # Remove the start of frame marker (MSB bit)
            
# #             # Read the second byte, which includes the LSBs
# #             lsb_byte = ser.read(1)
# #             if not lsb_byte:
# #                 continue  # Skip if no data received
            
# #             lsb = ord(lsb_byte) & 0x7F  # Only consider the 7 LSB bits

# #             # Reconstruct the original analog value from the two bytes
# #             value = (msb << 7) | lsb  # Combine MSB and LSB

# #             # Write and print the value
# #             f.write(f"{value}\n")
# #             print(f"Received value: {value}")

# <<<<<<< HEAD
# # if _name_ == "_main_":
# =======
# # if __name__ == "__main__":

# #     receive_data()


import serial
import time
import csv

def receive_data(serial_port='COM3', baud_rate=230400, output_file='eeg_dataREALTIMEEEEEE.csv'):
    ser = serial.Serial(serial_port, baud_rate)
    time.sleep(2)  # Wait for Arduino to reset and start sending data
    
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Timestamp', 'Value'])  # Write the header
        
        while True:
            # Read the first byte, which includes the start of frame marker and MSBs
            msb_byte = ser.read(1)
            if not msb_byte:
                continue  # Skip if no data received
            
            # Check if it's the start of a new frame (MSB of this byte should be 1)
            msb = ord(msb_byte) & 0x7F  # Remove the start of frame marker (MSB bit)
            
            # Read the second byte, which includes the LSBs
            lsb_byte = ser.read(1)
            if not lsb_byte:
                continue  # Skip if no data received
            
            lsb = ord(lsb_byte) & 0x7F  # Only consider the 7 LSB bits


            # Reconstruct the original analog value from the two bytes
            value = (msb << 7) | lsb  # Combine MSB and LSB
            
            # Record the current timestamp
            timestamp = time.time()

            # Write to CSV file and print the value
            csvwriter.writerow([timestamp, value])
            print(f"Received value: {value} at {timestamp}")

            # Reconstruct the original analog value from the two bytes
            value = (msb << 7) | lsb  # Combine MSB and LSB
            
            # Record the current timestamp
            timestamp = time.time()

            # Write to CSV file and print the value
            csvwriter.writerow([timestamp, value])
            print(f"Received value: {value} at {timestamp}")

if __name__ == "__main__":
    receive_data()