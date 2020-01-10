import pickle
import socket
from socket_utils import socket_config, send_msg, recv_msg

import torch

from tiny_tcnn.models.pnet_dense import PropagationNetwork
from tiny_tcnn.utils import load_pretrained_weights

# establish socket connection to OTCD container
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((socket_config["host"], socket_config['port']))
sock.listen(1)

# init Tiny T-CNN as tracking net
if torch.cuda.is_available():
    print("Using GPU")
    device = torch.device("cuda:0")
else:
    print("Using CPU")
    device = torch.device("cpu")
tracking_net = PropagationNetwork(vector_type='p')
tracking_net = tracking_net.to(device)

weights_file = "save/tracking_net_tiny_tcnn.pth"
tracking_net = load_pretrained_weights(tracking_net, weights_file)
tracking_net.eval()

# receive data from OTCD container and feed forward into Tiny T-CNN
# send model output back to OTCD container

while True:
    # Wait for a connection
    client_socket, client_address = sock.accept()
    print("Waiting for connection...")

    try:
        while True:
            socket_data = recv_msg(client_socket)
            if socket_data:

                # retrieve boxes_prev and motion vector image via socket
                socket_data = pickle.loads(socket_data)
                #print("received data from tracker.py:")
                #print(type(socket_data["boxes"]), "boxes", socket_data["boxes"], "boxes shape", socket_data["boxes"].shape)
                #print(type(socket_data["im_data"]), "im_data", socket_data["im_data"], "im_data shape", socket_data["im_data"].shape)
                boxes_prev = torch.from_numpy(socket_data["boxes"])
                motion_vectors_p = torch.from_numpy(socket_data["im_data"])

                # change box format from tlbr to tlwh
                boxes_prev = boxes_prev.clone()
                boxes_prev[..., 2:4] = boxes_prev[..., 2:4] - boxes_prev[..., 0:2] + 1

                # add frame_idx into boxes
                num_boxes = (boxes_prev.shape)[1]
                boxes_prev_tmp = torch.zeros(1, num_boxes, 5).float()
                boxes_prev_tmp[..., 1:] = boxes_prev
                boxes_prev = boxes_prev_tmp

                # rescale because motion vector image is factor 16 smaller than original frame
                boxes_prev = boxes_prev / 16.0

                print("boxes_prev", boxes_prev)

                boxes_prev = boxes_prev.to(device)
                motion_vectors_p = motion_vectors_p.to(device)

                # feed data forward through Tiny T-CNN
                with torch.set_grad_enabled(False):
                    start_inference = torch.cuda.Event(enable_timing=True)
                    end_inference = torch.cuda.Event(enable_timing=True)
                    start_inference.record()
                    # model expects boxes in format [frame_idx, xmin, ymin, w, h]
                    # and motion vectors in format [1, C, H, W] where channels are
                    # in RGB order with red as x motion and green as y motion
                    velocities_pred = tracking_net(motion_vectors_p, None, boxes_prev)
                    end_inference.record()
                    torch.cuda.synchronize()
                    last_inference_dt = start_inference.elapsed_time(end_inference) / 1000.0

                #print("velocities_pred shape", velocities_pred.shape)

                # post-process velcoties, e.g. de-standardize
                velocities_pred = velocities_pred.cpu()

                # convert output to numpy because tensors are not compatible between pytorch 0.3.1 and 1.4.0
                velocities_pred = velocities_pred.numpy()

                # send model output back to other container
                socket_data = {
                    "velocities_pred": velocities_pred,
                    "track_time": last_inference_dt
                }
                socket_data = pickle.dumps(socket_data)
                #print("Sending back output...")
                send_msg(client_socket, socket_data)
                #print("Output sent.")

                print("Predicted vecloties: ", velocities_pred)

    finally:
        client_socket.close()
        print("Socket closed.")
