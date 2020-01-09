import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision

from lib.resnet_atrous import resnet18
from lib.utils import load_pretrained_weights_to_modified_resnet, \
    change_box_format


class PropagationNetwork(nn.Module):
    def __init__(self, vector_type="p"):
        super(PropagationNetwork, self).__init__()

        self.POOLING_SIZE = 7  # the ROIs are split into m x m regions

        resnet = resnet18()
        resnet_weights = model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth')
        load_pretrained_weights_to_modified_resnet(resnet, resnet_weights)

        if vector_type == "p":
            conv0_in_channels = 2
        elif vector_type == "p+b":
            conv0_in_channels = 4

        base = [
            nn.Conv2d(conv0_in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.relu,
            resnet.layer2,
            resnet.relu
        ]
        self.base = nn.Sequential(*base)

        # fix pretrained base layers
        for p in self.base[6].parameters(): p.requires_grad = False
        for p in self.base[4].parameters(): p.requires_grad = False

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv1x1 = nn.Conv2d(256, 2*self.POOLING_SIZE*self.POOLING_SIZE, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False)
        self.pooling = nn.AvgPool2d(kernel_size=self.POOLING_SIZE, stride=self.POOLING_SIZE)

        # fix pretrained layers
        for p in self.base[0].parameters(): p.requires_grad = False
        for p in self.base[1].parameters(): p.requires_grad = False
        for p in self.base[3].parameters(): p.requires_grad = False
        for p in self.conv4.parameters(): p.requires_grad = False
        for p in self.bn4.parameters(): p.requires_grad = False


    def forward(self, motion_vectors_p, motion_vectors_b, boxes_prev):
        # motion vector are of shape [1, C, H, W]
        # channels are in RGB order where red is x motion and green is y motion
        motion_vectors_p = motion_vectors_p[:, :2, :, :]   # pick out the red and green channel
        motion_vectors = motion_vectors_p

        # concatenate P and B vectors in channel dimension
        if motion_vectors_b is not None:
            motion_vectors_b = motion_vectors_b[:, :2, :, :]
            motion_vectors = torch.cat((motion_vectors_p, motion_vectors_b), axis=1)

        x = self.base(motion_vectors)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.conv1x1(x)

        boxes_prev = boxes_prev.view(-1, 5)
        boxes_prev_ = change_box_format(boxes_prev)

        # compute ratio of input size to size of base output
        x = torchvision.ops.ps_roi_pool(x, boxes_prev_, output_size=(self.POOLING_SIZE, self.POOLING_SIZE), spatial_scale=1/4)
        x = self.pooling(x)
        x = x.squeeze()
        velocities_pred = x.view(-1, 2)
        return velocities_pred


    # names of layer weights (excludes batch norm layers, etc.), needed for weight logging
    layer_keys = [
        'base.0.weight',
        'base.4.0.conv1.weight',
        'base.4.0.conv2.weight',
        'base.4.2.conv1.weight',
        'base.4.2.conv2.weight',
        'base.6.0.conv1.weight',
        'base.6.0.conv2.weight',
        'base.6.0.downsample.0.weight',
        'base.6.2.conv1.weight',
        'base.6.2.conv2.weight',
        'conv4.weight',
        'conv1x1.weight',
    ]


if __name__ == "__main__":

    model = PropagationNetwork()
    print([p.requires_grad for p in model.parameters()])
    print("Model No. of Params {}".format(count_params(model)))
    print(model)
