"""Data transforms.

All transforms below can be used with `torchvision.transforms.Compose`, e.g.

```
custom_transform = torchvision.transforms.Compose([
    RandomFlip(directions=["x", "y"]),
    StandardizeVelocities(mean=[1.3, 2.4, 0.2], std=[1.1, 1.5, 1.3]),
    ScaleImage(items=["motion_vectors"], scale=600, max_size=1000),
])
```

All the transforms can then be applied to a sample by calling
```
sample = custom_transform(sample)
```

The sample should be a Python dictionary with the following key values pairs:

- motion_vectors (`list` of `torch.Tensor`): Each item is a motion vector image
    of shape (B, H, W, C) or (H, W, C), dtype float32 and channel order BGR.
    Typically, C = 3. Blue channel has no meaning, green channel corresponds to
    y component of motion vector and red channel to x component of motion vector.
    The list can contain up to two items where the first one corresponds to the
    P vectors and the second one to the B vectors. The second item is optional.

- boxes_prev (`torch.Tensor`): Set of bounding boxes in previous frame. Shape
    (B, N, 5) or (N, 5) with batch size B, number of boxes N and format
    [frame_idx, xmin, ymin, w, h] in the last dimension.

- velocities (`torch.Tensor`): Set of box velocities. Shape (B, N, 4) or (N, 4)
    with batch size B, number of boxes N and format [v_xc, v_yc, v_w, v_h] in
    the last dimenion. The last dimenision stand for the velocities of each box
    center point and velocities of box width and height.

- det_boxes_prev (`torch.Tensor`): Same format as `boxes_prev`, but these are
    the detection boxes in the previous frame rather than the tracked boxes.
"""
import random
import pickle
import copy
import numpy as np
import cv2
import torch


class StandardizeMotionVectors:
    """Subtracts mean from motion vectors and divides by standard deviation.

    motion_vectors[channel] = (motion_vectors[channel] - mean[channel]) / std[channel]

    Args:
        mean (`list` of `float`): Mean values for blue, green and red channel to
            be subtracted from the motion vector image.

        std (`list` of `float`): Standard deviations per channel by which to
            divide the mean subtracted motion vector image.
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample_transformed = copy.deepcopy(sample)
        for i in range(len(sample["motion_vectors"])):
            motion_vectors = sample["motion_vectors"][i].clone()
            mean = torch.tensor(self.mean[i])
            std = torch.tensor(self.std[i])
            motion_vectors = (motion_vectors - mean) / std
            sample_transformed["motion_vectors"][i] = motion_vectors
        return sample_transformed

    def __repr__(self):
        repr = "StandardizeMotionVectors (\n    mean_p={},\n    std_p={},\n    mean_b={},\n    std_b={}\n)".format(
            [float(val) for val in self.mean[0]], [float(val) for val in self.std[0]],
            [float(val) for val in self.mean[1]], [float(val) for val in self.std[1]])
        return repr


class StandardizeVelocities:
    """Subtracts mean from velocities and divides by standard deviation.

    velocities[x] = (velocities[x] - mean[x]) / std[x], x = {v_xc, v_yc, v_w, v_h}

    Args:
        mean (`list` of `float`): Mean values for [v_cx, v_cy, v_w, v_h] to
            be subtracted from the velocities.

        std (`list` of `float`): Standard deviations for [v_cx, v_cy, v_w, v_h]
            by which to divide the mean subtracted velocities.

        inverse (`bool`): If True compute the inverse of this transform, that is
            velocities[x] = velocities[x] * std[x] + mean[x], x = {v_xc, v_yc,
            v_w, v_h}
    """
    def __init__(self, mean, std, inverse=False):
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.inverse = inverse

    def __call__(self, sample):
        velocities = sample["velocities"].clone()
        if self.inverse:
            velocities = velocities * self.std + self.mean
        else:
            velocities = (velocities - self.mean) / self.std
        sample_transformed = copy.deepcopy(sample)
        sample_transformed["velocities"] = velocities
        return sample_transformed

    def __repr__(self):
        repr = "StandardizeVelocities (\n    mean={},\n    std={}\n)".format(
            [float(val) for val in self.mean], [float(val) for val in self.std])
        return repr


# def scale_image_(image, scale=600, max_size=1000):
#     # determine the scaling factor
#     image = image.numpy()
#     size_min = np.min(image.shape[-3:-1])
#     size_max = np.max(image.shape[-3:-1])
#     scaling_factor = float(scale) / float(size_min)
#     if np.round(scaling_factor * size_max) > max_size:
#         scaling_factor = float(max_size) / float(size_max)
#      # scale the frame
#     if image.ndim == 3:
#         image_resized = cv2.resize(image, None, None, fx=scaling_factor,
#             fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
#     elif image.ndim == 4:
#         image_resized = []
#         for batch_idx in range(image.shape[0]):
#             image_resized.append(cv2.resize(image[batch_idx, ...], None, None,
#                 fx=scaling_factor, fy=scaling_factor,
#                 interpolation=cv2.INTER_LINEAR))
#         image_resized = np.stack(image_resized, axis=0)
#     else:
#         raise ValueError("Invalid frame dimension")
#     image_resized = torch.from_numpy(image_resized)
#     return image_resized, scaling_factor
#
#
# def scale_boxes_(sample, sample_transformed, scaling_factor, box_types):
#     # helper to scale multiple boxes only if exist in dictionary
#     for box_type in box_types:
#         try:
#             boxes = sample[box_type].clone()
#         except KeyError:
#             pass
#         else:
#             boxes[..., 1:] *= scaling_factor
#             sample_transformed[box_type] = boxes
#
#
# class ScaleImage:
#     """Scale the motion vectors or frame considering a maximum size.
#
#     Frame is scaled so that it's shortest side has the size specified by the
#     `scale` parameter. The aspect ratio is constant. In case, the
#     longer side would exceed the value specified by `max_size` a new
#     scaling factor is computed so that the longer side matches `max_size`.
#     The shorter side is then smaller than specified by `scale`.
#
#     Args:
#         items (`list` of `str`): Can be "motion_vectors" and/or "frame".
#             Determines whether to scale motion vectors and/or frame inside
#             the sample dict provided during call.
#
#         scale (`int` or `float`): The desired size of the shorter
#             side of the frame after scaling.
#
#         max_size (`int` or `float`): The desired maximum size of the
#             longer side of the frame after scaling.
#
#         return_scale (`bool`): If True return the used scaling factor is
#             inserted in the output sample with key `scaling_factor`.
#
#         scale_boxes (`bool`): If True scale `boxes_prev`, `boxes` and
#             `det_boxes_prev` if existing by the scaling factor used to scale the
#             motion vectors/frame.
#     """
#     def __init__(self, items=["motion_vectors"], scale=600, max_size=1000, return_scale=False, scale_boxes=True):
#         self.items = items
#         self.scale = scale
#         self.max_size = max_size
#         self.return_scale = return_scale
#         self.scale_boxes = scale_boxes
#
#     def __call__(self, sample):
#         sample_transformed = copy.deepcopy(sample)
#         for item in self.items:
#             image = sample[item].clone()
#             image_resized, scaling_factor = scale_image_(image, self.scale, self.max_size)
#             sample_transformed[item] = image_resized
#         # scale boxes
#         if self.scale_boxes:
#             scale_boxes_(sample, sample_transformed, scaling_factor,
#                 ["boxes_prev", "boxes", "det_boxes_prev"])
#         if self.return_scale:
#             sample_transformed["scaling_factor"] = scaling_factor
#         return sample_transformed
#
#     def __repr__(self):
#         repr = ("ScaleImage (\n    items={},\n    scale={},\n    max_size={},\n"
#                 "    return_scale={},\n    scale_boxes={}\n)").format(self.items,
#                 self.scale, self.max_size, self.return_scale, self.scale_boxes)
#         return repr
#
#
# class RandomScaleImage:
#     """Scale motion vector or frame by a randomly chosen scale.
#
#     Args:
#         items (`list` of `str`): Can be "motion_vectors" and/or "frame".
#             Determines whether to scale motion vectors and/or frame inside
#             the sample dict provided during call.
#
#         scales (`list` of `int` or `list` of `float`): A set of scales one of
#             which is uniformly randomly chosen during each call of this
#             transform.
#
#         max_size (`int` or `float`): The desired maximum size of the
#             longer side of the frame after scaling.
#
#         return_scale (`bool`): If True return the used scaling factor is
#             inserted in the output sample with key `scaling_factor`.
#
#         scale_boxes (`bool`): If True scale `boxes_prev`, `boxes` and
#             `det_boxes_prev` if existing by the scaling factor used to scale the
#             motion vectors/frame.
#     """
#     def __init__(self, items=["motion_vectors"], scales=[300, 400, 500, 600], max_size=1000, return_scale=False, scale_boxes=True):
#         self.items = items
#         self.scales = scales
#         self.max_size = max_size
#         self.return_scale = return_scale
#         self.scale_boxes = scale_boxes
#
#     def scale_boxes_(sample, sample_transformed, scaling_factor, box_type):
#         try:
#             boxes = sample[box_type].clone()
#         except KeyError:
#             pass
#         else:
#             boxes[..., 1:] *= scaling_factor
#             sample_transformed[box_type] = boxes
#
#     def __call__(self, sample):
#         sample_transformed = copy.deepcopy(sample)
#         scale = random.choice(self.scales)
#         for item in self.items:
#             image = sample[item].clone()
#             image_resized, scaling_factor = scale_image_(image, scale, self.max_size)
#             sample_transformed[item] = image_resized
#         # scale boxes
#         if self.scale_boxes:
#             scale_boxes_(sample, sample_transformed, scaling_factor,
#                 ["boxes_prev", "boxes", "det_boxes_prev"])
#         if self.return_scale:
#             sample_transformed["scaling_factor"] = scaling_factor
#         return sample_transformed
#
#     def __repr__(self):
#         repr = ("RandomScaleImage (\n    items={},\n    scales={},\n    max_size={},\n"
#                 "    return_scale={},\n    scale_boxes={}\n)").format(self.items,
#                 self.scales, self.max_size, self.return_scale, self.scale_boxes)
#         return repr


def flip_motion_vectors_(motion_vectors, direction):
    # motion_vectors must have shape (H, W, C) with BGR order (x: red, y: green)
    if direction == "y":
        motion_vectors = cv2.flip(motion_vectors, 1)
        motion_vectors[:, :, 2] = -1 * motion_vectors[:, :, 2]  # change sign of x-motion
    elif direction == "x":
        motion_vectors = cv2.flip(motion_vectors, 0)
        motion_vectors[:, :, 1] = -1 * motion_vectors[:, :, 1]  # change sign of y-motion
    return motion_vectors


def flip_boxes_(boxes, direction, image_width, image_height):
    # boxes must have shape (N, 4) with [xmin, ymin, w, h] in the last dim
    if direction == "y":
        boxes[:, 0] = image_width - boxes[:, 2] - boxes[:, 0]
    elif direction == "x":
        boxes[:, 1] = image_height - boxes[:, 3] - boxes[:, 1]
    return boxes


def flip_velocities_(velocities, direction):
    # velocities must have shape (N, 4) with [v_xc, v_yc, v_w, v_h] in the last dim
    if direction == "y":
        velocities[:, 0] = -1 * velocities[:, 0]
    elif direction == "x":
        velocities[:, 1] = -1 * velocities[:, 1]
    return velocities


def flip_(sample, sample_transformed, direction):
    velocities = sample["velocities"].clone()
    try:
        boxes_prev = sample["boxes_prev"].clone()
    except KeyError:
        boxes_prev = None
    try:
        boxes = sample["boxes"].clone()
    except KeyError:
        boxes = None
    try:
        det_boxes_prev = sample["det_boxes_prev"].clone()
    except KeyError:
        det_boxes_prev = None
    # mvs have shape (H, W, C) or (B, H, W, C)
    image_width = sample["motion_vectors"][0].shape[-2]
    image_height = sample["motion_vectors"][0].shape[-3]
    # flip motion vectors
    for i in range(len(sample["motion_vectors"])):
        motion_vectors = sample["motion_vectors"][i].clone().numpy()
        if motion_vectors.ndim == 3:
            motion_vectors = flip_motion_vectors_(motion_vectors, direction)
        elif motion_vectors.ndim == 4:
            motion_vectors_flipped = []
            for batch_idx in range(motion_vectors.shape[0]):
                motion_vectors_flipped.append(flip_motion_vectors_(
                    motion_vectors[batch_idx, ...], direction))
            motion_vectors = np.stack(motion_vectors_flipped, axis=0)
        motion_vectors = torch.from_numpy(motion_vectors)
        sample_transformed["motion_vectors"][i] = motion_vectors
    # flip other items in the sample
    if sample["motion_vectors"][0].ndim == 3:
        if boxes_prev is not None:
            boxes_prev[:, 1:] = flip_boxes_(boxes_prev[:, 1:], direction,
                image_width, image_height)
        if boxes is not None:
            boxes[:, 1:] = flip_boxes_(boxes[:, 1:], direction,
                image_width, image_height)
        if det_boxes_prev is not None:
            det_boxes_prev[:, 1:] = flip_boxes_(det_boxes_prev[:, 1:],
                direction, image_width, image_height)
        velocities = flip_velocities_(velocities, direction)
    elif sample["motion_vectors"][0].ndim == 4:
        for batch_idx in range(sample["motion_vectors"][0].shape[0]):
            if boxes_prev is not None:
                boxes_prev[batch_idx, :, 1:] = flip_boxes_(
                    boxes_prev[batch_idx, :, 1:], direction, image_width,
                    image_height)
            if boxes is not None:
                boxes[batch_idx, :, 1:] = flip_boxes_(
                    boxes[batch_idx, :, 1:], direction, image_width,
                    image_height)
            if det_boxes_prev is not None:
                det_boxes_prev[batch_idx, :, 1:] = flip_boxes_(
                    det_boxes_prev[batch_idx, :, 1:], direction, image_width,
                    image_height)
            velocities[batch_idx, ...] = flip_velocities_(
                velocities[batch_idx, ...], direction)
    else:
        raise ValueError("Invalid dimension of motion vectors")
    sample_transformed["velocities"] = velocities
    if boxes_prev is not None:
        sample_transformed["boxes_prev"] = boxes_prev
    if boxes is not None:
        sample_transformed["boxes"] = boxes
    if det_boxes_prev is not None:
        sample_transformed["det_boxes_prev"] = det_boxes_prev
    return sample_transformed


class Flip:
    """Flips motion vectors, boxes and velocities along a given axis.

    This method flips the motion vector image either along the x or y axis.
    Bounding boxes and velocities are flipped accordingly. Also, the directions
    of the motion vectors inside the motion vector image are inverted
    accordingly.

    Args:
        direction (`str`): Either "x" or "y" indicating along which axis to
        flip.
    """
    def __init__(self, direction="x"):
        self.direction = direction

    def __call__(self, sample):
        sample_transformed = copy.deepcopy(sample)
        flip_(sample, sample_transformed, direction=self.direction)
        return sample_transformed

    def __repr__(self):
        return "Flip (direction={})".format(self.direction)


class RandomFlip:
    """Randomly flips motion vectors, boxes and velocities.

    Whether or not to flip is chosen randomly with equal probabilities. If a
    flip along both y and x axis shall be performed, it is first
    dtermined whether to flip along y axis (p = 0.5) and afterwards
    whether to flip along x axis (p = 0.5). The probability of an image
    being flipped both along x and y axis is thus p = 0.25.

    Args:
        directions (`list` of `str`): List can contain "y" and
            "x" to determine along which axis a flip can occur.
    """
    def __init__(self, directions=["x", "y"]):
        self.directions = directions

    def __call__(self, sample):
        sample_transformed = copy.deepcopy(sample)
        for direction in self.directions:
            if random.choice([True, False]):  # 50 percent chance that flip happens
                flip_(sample, sample_transformed, direction=direction)
        return sample_transformed

    def __repr__(self):
        return "RandomFlip (directions={})".format(self.directions)


class RandomMotionChange:
    """Randomly modifies the color channels of the motion vector image.

    A random value is added to the x and y channel of the motion vector image.
    The value for each channel is drawn from a normal distribution which has the
    same mean as the corresponding motion vector channel and a standard
    deviation which is the standard deviation of the motion vector channel times
    the `scale` factor.

    Mean and standard deviation of both the x and y channel of the motion vector
    image are computed and a random value for each channel is sampled

    Args:
        scale (`float`): The scaling factor for the standard deviation of the
            distribution from which color modifications are drawn. Must lie in
            range (0, 1].
    """
    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, sample):
        sample_transformed = copy.deepcopy(sample)
        # sample statistics from P vectors
        mean_x = torch.mean(sample_transformed["motion_vectors"][0][..., 2])
        std_x = torch.std(sample_transformed["motion_vectors"][0][..., 2])
        mean_y = torch.mean(sample_transformed["motion_vectors"][0][..., 1])
        std_y = torch.std(sample_transformed["motion_vectors"][0][..., 1])
        # choose a random color change from a scaled version of that distribution
        x_rand = float(np.random.normal(loc=mean_x, scale=std_x*self.scale))
        y_rand = float(np.random.normal(loc=mean_y, scale=std_y*self.scale))
        # apply color change to cx and y channel
        for i in range(len(sample_transformed["motion_vectors"])):
            sample_transformed["motion_vectors"][i][..., 2] += x_rand
            sample_transformed["motion_vectors"][i][..., 1] += y_rand
        return sample_transformed

    def __repr__(self):
        return "RandomMotionChange (scale={})".format(self.scale)
