import socket
import struct
import os

# Get the IP address of the Raspberry Pi
def get_local_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# Set up socket and bind to a port
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
local_ip = get_local_ip_address()
s.bind((local_ip, 1234))
s.listen(5)


while True:
    # Accept incoming connections
    clientsocket, address = s.accept()
    print(f"Connection from {address} has been established!")

    # Receive the file type
    file_type = clientsocket.recv(1024).decode()

    # Send a response to the client to indicate that the file type has been received
    clientsocket.send(bytes("File type received", "utf-8"))

    # Receive the length prefix indicating the size of the incoming data
    length_prefix = clientsocket.recv(4)
    data_length = struct.unpack("!I", length_prefix)[0]

    # Receive the file data
    received_bytes = 0
    file_data = b""
    while received_bytes < data_length:
        remaining_bytes = data_length - received_bytes
        data = clientsocket.recv(4096 if remaining_bytes > 4096 else remaining_bytes)
        received_bytes += len(data)
        file_data += data

    # If the file type is pickle, deserialize the data
    if file_type == "pickle":
        with open(os.path.join("server", "embeddings.pkl"), "wb") as f:
            f.write(file_data)

    # If the file type is image, write the data to a file
    elif file_type == "image":
        with open(os.path.join("server", "received_image.png"), "wb") as f:
            f.write(file_data)

    # Send a response to the client to indicate that the file data has been received
    clientsocket.send(bytes("File data received", "utf-8"))

    # Close the connection
    clientsocket.close()