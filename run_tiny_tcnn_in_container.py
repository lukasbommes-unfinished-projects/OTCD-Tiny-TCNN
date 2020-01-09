import pickle
import socket
from socket_utils import socket_config, send_msg, recv_msg


# TODO:
# - have a while loop retrieve data from the socket queue
# - feed forward data into Tiny T-CNN
# - enqueue network outputs in another queue anbd send via socket back to other container

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((socket_config["host"], socket_config['port']))
sock.listen(1)

while True:
    # Wait for a connection
    client_socket, client_address = sock.accept()
    print("Waiting for connection...")

    try:
        while True:
            # get data from other container via socket
            socket_data = recv_msg(client_socket)
            if socket_data:

                # feed data forward through Tiny T-CNN
                # move tensors back to GPU


                # Send data to web client via socketIO
                socket_data = pickle.loads(socket_data)

                print("received data from tracker.py:")
                print(type(socket_data["boxes"]), "boxes", socket_data["boxes"], "boxes shape", socket_data["boxes"].shape)
                print(type(socket_data["im_data"]), "im_data", socket_data["im_data"], "im_data shape", socket_data["im_data"].shape)
                #print(socket_data[0], socket_data[1])

                # send model output back to other container
                output = "output"
                socket_data = pickle.dumps(output)
                print("Sending back output...")
                send_msg(client_socket, socket_data)
                print("Output sent.")

    finally:
        client_socket.close()
        print("Socket closed.")
