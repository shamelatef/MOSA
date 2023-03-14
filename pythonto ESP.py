import socket

# Set the IP address and port number of the ESP8266
ip = '172.20.10.2'
port = 80

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the ESP8266
s.connect((ip, port))

# Send a signal to turn on the LED
s.send(b'0')

# Close the socket
s.close()
