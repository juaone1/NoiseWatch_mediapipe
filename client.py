# import os
# import socket
# import struct

# def send_image(client_socket, folder_name, file_name, file_path):
#     # Send folder name length
#     client_socket.sendall(struct.pack('!I', len(folder_name)))

#     # Send folder name
#     client_socket.sendall(folder_name.encode())

#     # Send file name length
#     client_socket.sendall(struct.pack('!I', len(file_name)))

#     # Send file name
#     client_socket.sendall(file_name.encode())

#     # Send file size
#     file_size = os.path.getsize(file_path)
#     client_socket.sendall(struct.pack('!Q', file_size))

#     # Send file
#     with open(file_path, 'rb') as f:
#         data = f.read(4096)
#         while data:
#             client_socket.sendall(data)
#             data = f.read(4096)

# root_folder = "dataset"
# server_host = 'localhost'
# server_port = 12345

# for folder_name in os.listdir(root_folder):
#     folder_path = os.path.join(root_folder, folder_name)
#     if os.path.isdir(folder_path):
#         for file_name in os.listdir(folder_path):
#             file_path = os.path.join(folder_path, file_name)

#             if os.path.isfile(file_path):
#                 client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#                 client_socket.connect((server_host, server_port))

#                 send_image(client_socket, folder_name, file_name, file_path)
#                 print(f"Sent {file_name} in {folder_name} folder")
                
#                 client_socket.close()

import socket
import os
import struct
import pickle

# Set up socket and connect to the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = "192.168.1.21"
s.connect((server_ip, 1234))

# Send the file type
file_type = "pickle"
s.send(bytes(file_type, "utf-8"))

# Receive a response from the server to indicate that the file type has been received
response = s.recv(1024).decode()
print(response)

# Send the data length prefix
with open("embeddings.pkl", "rb") as f:
    data = f.read()
    data_length_prefix = struct.pack("!I", len(data))
    s.sendall(data_length_prefix)

# Send the file data in chunks
bytes_sent = 0
while bytes_sent < len(data):
    remaining_bytes = len(data) - bytes_sent
    chunk_size = 4096 if remaining_bytes > 4096 else remaining_bytes
    s.send(data[bytes_sent:bytes_sent+chunk_size])
    bytes_sent += chunk_size

# Receive a response from the server to indicate that the file data has been received
response = s.recv(1024).decode()
print(response)

# Close the connection
s.close()
