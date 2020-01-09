import math
import torch
import numpy as np
import cv2
from scipy.interpolate import griddata


def get_vectors_by_source(motion_vectors, source):
    """Returns subset of motion vectors with a specified source frame.

    The source parameter of a motion vector specifies the temporal position of
    the reference (source) frame relative to the current frame. Each vector
    starts at the point (src_x, sry_y) in the source frame and points to the
    point (dst_x, dst_y) in the current frame. If the source value is for
    example -1, then the reference frame is the previous frame.

    For B frames there are motion vectors which refer macroblocks both to past
    frames and future frames. By setting the source parameter to "past" this
    method filters out motion vectors referring to future frames and returns the
    set of motion vectors which refer to past frames (e.g. the equivalent to the
    motion vectors in P frames). Similarly, by setting the value to "future"
    only vectors referring to future frames are returned.

    Args:
        motion_vectors (`numpy.ndarray`): Array of shape (N, 11) containing all
            N motion vectors inside a frame. N = 0 is allowed meaning no vectors
            are present in the frame.

        source (`ìnt` or `string`): Motion vectors with this value for their
            source parameter (the location of the reference frame) are selected.
            If "future", all motion vectors with a positive source value are
            returned (only for B-frames). If "past" all motion vectors with
            a negative source value are returned.

    Returns:
        motion_vectors (`numpy.ndarray`): Array of shape (M, 11) containing all
        M motion vectors with the specified source value. If N = 0 => M = 0
        that is an empty numpy array of shape (0, 11) is returned.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        if source == "past":
            idx = np.where(motion_vectors[:, 0] < 0)[0]
        elif source == "future":
            idx = np.where(motion_vectors[:, 0] > 0)[0]
        else:
            idx = np.where(motion_vectors[:, 0] == source)[0]
        return motion_vectors[idx, :]


def get_nonzero_vectors(motion_vectors):
    """Returns subset of motion vectors which have non-zero magnitude."""
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        idx = np.where(np.logical_or(motion_vectors[:, 7] != 0, motion_vectors[:, 8] != 0))[0]
        return motion_vectors[idx, :]


def normalize_vectors(motion_vectors):
    """Normalizes motion vectors to the past frame as reference frame.

    The source value in the first column is set to -1 for all p-vectors and
    set to 1 for all b-vectors. The x and y motion values are scaled
    accordingly. Vector source position and destination position are unchanged.

    Args:
        motion_vectors (`numpy.ndarray`): Array of shape (N, 11) containing all
            N motion vectors inside a frame. N = 0 is allowed meaning no vectors
            are present in the frame.

    Returns:
        motion_vectors (`numpy.ndarray`): Array of shape (M, 11) containing the
        normalized motion vectors. If N = 0 => M = 0 that is an empty numpy
        array of shape (0, 11) is returned.
    """
    if np.shape(motion_vectors)[0] == 0:
        return motion_vectors
    else:
        motion_vectors[:, 7] = motion_vectors[:, 7] / motion_vectors[:, 0]  # motion_x
        motion_vectors[:, 8] = motion_vectors[:, 8] / motion_vectors[:, 0]  # motion_y
        motion_vectors[:, 0] = np.sign(motion_vectors[:, 0])
        return motion_vectors


def motion_vectors_to_hsv_image(motion_vectors, frame_shape=(1920, 1080)):
    """Convert a set of motion vectors into a HSV image and return it as BGR image.

    Args:
        motion_vectors (`numpy.ndarray`): Motion vector array with shape [N, 10]
            as returned by VideoCap. The motion vector array should only contain P-vectors
            which can be filtered out by using get_vectors_by_source(motion_vectors, "past").
            Also, the reference frame should be normalized by using normalize_vectors.

        frame_shape (`tuple` of `int`): Desired (width, height) in pixels of the returned image.
            Should correspond to the size of the source footage of which the motion vectors
            where extracted.

    Returns:
        `numpy.ndarray` The motion vectors encoded as uint8 image. Image shape
        is (height, widht, 3) and channel order is BGR. The BGR image is
        generated from a HSV representation of the motion vectors where the
        vector magnitude is stored in the saturation channel and the vector
        angle is stored in the hue channel.
    """
    # compute necessary frame shape
    need_width = math.ceil(frame_shape[0] / 16) * 16
    need_height = math.ceil(frame_shape[1] / 16) * 16

    image = np.zeros((need_height, need_width, 3), dtype=np.uint8)

    if np.shape(motion_vectors)[0] != 0:

        # get minimum and maximum values
        mvs_dst_x = motion_vectors[:, 5]
        mvs_dst_y = motion_vectors[:, 6]
        mb_w = motion_vectors[:, 1]
        mb_h = motion_vectors[:, 2]
        mvs_tl_x = (mvs_dst_x - 0.5 * mb_w).astype(np.int64)
        mvs_tl_y = (mvs_dst_y - 0.5 * mb_h).astype(np.int64)

        # scale motion value with the motion_scale
        mvs_motion_x = (motion_vectors[:, 7] / motion_vectors[:, 9]).reshape(-1, 1)
        mvs_motion_y = (motion_vectors[:, 8] / motion_vectors[:, 9]).reshape(-1, 1)

        # transform x and y motion to angle in range 0...180 and magnitude in range 0...255
        mvs_motion_magnitude, mvs_motion_angle = cv2.cartToPolar(mvs_motion_x, -1.0*mvs_motion_y)
        mvs_motion_angle = mvs_motion_angle * 180 / (2 * np.pi)  # hue channel is [0, 180]
        mvs_motion_magnitude = cv2.normalize(mvs_motion_magnitude, None, 0, 255, cv2.NORM_MINMAX)

        for i, motion_vector in enumerate(motion_vectors):
            # repeat value in theshape of the underlying macroblock, e.g. 16 x 16 or 16 x 8
            mvs_motion_angle_repeated = np.repeat(np.repeat(mvs_motion_angle[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)
            mvs_motion_magnitude_repeated = np.repeat(np.repeat(mvs_motion_magnitude[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)

            # insert repeated block into image, angle is hue (channel 0), magitude is saturation (channel 1)
            image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 0] = mvs_motion_angle_repeated
            image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 1] = mvs_motion_magnitude_repeated

    # crop the image back to frame_shape
    image = image[0:frame_shape[1], 0:frame_shape[0], :]
    image[:, :, 2] = 255
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image



def motion_vectors_to_image(motion_vectors, frame_shape=(1920, 1080), scale=False):
    """Converts a set of motion vectors into a BGR image.

    Args:
        motion_vectors (`numpy.ndarray`): Motion vector array with shape [N, 10]
            as returned by VideoCap. The motion vector array should only contain P-vectors
            which can be filtered out by using get_vectors_by_source(motion_vectors, "past").
            Also, the reference frame should be normalized by using normalize_vectors.

        frame_shape (`tuple` of `int`): Desired (width, height) in pixels of the returned image.
            Should correspond to the size of the source footage of which the motion vectors
            where extracted.

        scale (`bool`): If True, scale motion vector components in the output image to
            range [0, 1]. If False, do not scale.

    Returns:
        `numpy.ndarray` The motion vectors encoded as float32 image. Image shape
        is (height, widht, 3) and channel order is BGR. The red channel contains
        the x motion components of the motion vectors and the green channel the
        y motion components.
    """
    # compute necessary frame shape
    need_width = math.ceil(frame_shape[0] / 16) * 16
    need_height = math.ceil(frame_shape[1] / 16) * 16

    image = np.zeros((need_height, need_width, 3), dtype=np.float32)

    if np.shape(motion_vectors)[0] != 0:

        # get minimum and maximum values
        mvs_dst_x = motion_vectors[:, 5]
        mvs_dst_y = motion_vectors[:, 6]
        mb_w = motion_vectors[:, 1]
        mb_h = motion_vectors[:, 2]
        mvs_tl_x = (mvs_dst_x - 0.5 * mb_w).astype(np.int64)
        mvs_tl_y = (mvs_dst_y - 0.5 * mb_h).astype(np.int64)

        # compute value
        mvs_motion_x = (motion_vectors[:, 7] / motion_vectors[:, 9]).reshape(-1, 1)
        mvs_motion_y = (motion_vectors[:, 8] / motion_vectors[:, 9]).reshape(-1, 1)

        if scale:
            mvs_min_x = np.min(mvs_motion_x)
            mvs_max_x = np.max(mvs_motion_x)
            mvs_min_y = np.min(mvs_motion_y)
            mvs_max_y = np.max(mvs_motion_y)
            mvs_motion_x = (mvs_motion_x - mvs_min_x) / (mvs_max_x - mvs_min_x)
            mvs_motion_y = (mvs_motion_y - mvs_min_y) / (mvs_max_y - mvs_min_y)

        for i, motion_vector in enumerate(motion_vectors):
            # repeat value
            mvs_motion_x_repeated = np.repeat(np.repeat(mvs_motion_x[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)
            mvs_motion_y_repeated = np.repeat(np.repeat(mvs_motion_y[i, :].reshape(1, 1), mb_h[i], axis=0), mb_w[i], axis=1)

            # insert repeated block into image
            image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 2] = mvs_motion_x_repeated
            image[mvs_tl_y[i]:mvs_tl_y[i]+mb_h[i], mvs_tl_x[i]:mvs_tl_x[i]+mb_w[i], 1] = mvs_motion_y_repeated

    # crop the image back to frame_shape
    image = image[0:frame_shape[1], 0:frame_shape[0], :]

    return image


def motion_vectors_to_grid(motion_vectors, frame_shape=(1920, 1080)):
    """Converts motion vectors into an image."""
    motion_vectors_grid = np.zeros((math.ceil(frame_shape[1]/16), math.ceil(frame_shape[0]/16), 3), dtype=np.float32)
    if motion_vectors.shape[0] > 0:
        motion_vectors = motion_vectors.astype(np.float32)
        mvs_x = motion_vectors[:, 5].astype(np.int64)
        mvs_y = motion_vectors[:, 6].astype(np.int64)
        x = ((mvs_x - 8) // 16).astype(np.int64)
        y = ((mvs_y - 8) // 16).astype(np.int64)
        motion_vectors_grid[y, x, 2] = motion_vectors[:, 7] / motion_vectors[:, 9]  # x component
        motion_vectors_grid[y, x, 1] = motion_vectors[:, 8] / motion_vectors[:, 9] # y component
    return motion_vectors_grid


def motion_vectors_to_hsv_grid(motion_vectors, frame_shape=(1920, 1080)):
    """Converts motion vectors into an image."""
    motion_vectors_grid = np.zeros((math.ceil(frame_shape[1]/16), math.ceil(frame_shape[0]/16), 3), dtype=np.uint8)
    if motion_vectors.shape[0] > 0:
        motion_vectors = motion_vectors.astype(np.float32)
        mvs_x = motion_vectors[:, 5].astype(np.int64)
        mvs_y = motion_vectors[:, 6].astype(np.int64)
        x = ((mvs_x - 8) // 16).astype(np.int64)
        y = ((mvs_y - 8) // 16).astype(np.int64)
        mvs_motion_x = motion_vectors[:, 7] / motion_vectors[:, 9]  # x component
        mvs_motion_y = motion_vectors[:, 8] / motion_vectors[:, 9]  # x component
        mvs_motion_magnitude, mvs_motion_angle = cv2.cartToPolar(mvs_motion_x, -1.0*mvs_motion_y)
        mvs_motion_angle = mvs_motion_angle * 180 / (2 * np.pi)  # hue channel is [0, 180]
        mvs_motion_magnitude = cv2.normalize(mvs_motion_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        motion_vectors_grid[y, x, 0] = mvs_motion_angle.reshape(-1)
        motion_vectors_grid[y, x, 1] = mvs_motion_magnitude.reshape(-1)
        motion_vectors_grid[:, :, 2] = 255
        motion_vectors_grid = cv2.cvtColor(motion_vectors_grid, cv2.COLOR_HSV2BGR)
    return motion_vectors_grid


def motion_vectors_to_grid_interpolated(motion_vectors, frame_shape=(1920, 1080), method='nearest'):
    """Interpolates motion vectors on a 16 x 16 grid and converts them into an image."""
    motion_vectors_grid = np.zeros((math.ceil(frame_shape[1]/16), math.ceil(frame_shape[0]/16), 3), dtype=np.float32)
    if motion_vectors.shape[0] > 0:
        mvs_x = motion_vectors[:, 5]
        mvs_y = motion_vectors[:, 6]
        mvs_x_motion = motion_vectors[:, 7] / motion_vectors[:, 9]
        mvs_y_motion = motion_vectors[:, 8] / motion_vectors[:, 9]
        xi = np.arange(8, math.ceil(frame_shape[0] / 16) * 16, 16)
        yi = np.arange(8, math.ceil(frame_shape[1] / 16) * 16, 16)
        mvs_x_motion_interp = griddata((mvs_x, mvs_y), mvs_x_motion, (xi[None, :], yi[:, None]), method=method)
        mvs_y_motion_interp = griddata((mvs_x, mvs_y), mvs_y_motion, (xi[None, :], yi[:, None]), method=method)
        motion_vectors_grid[:, :, 2] = mvs_x_motion_interp
        motion_vectors_grid[:, :, 1] = mvs_y_motion_interp
    return motion_vectors_grid


def motion_vectors_to_hsv_grid_interpolated(motion_vectors, frame_shape=(1920, 1080), method='nearest'):
    """Interpolates motion vectors on a 16 x 16 grid and converts them into an image."""
    motion_vectors_grid = np.zeros((math.ceil(frame_shape[1]/16), math.ceil(frame_shape[0]/16), 3), dtype=np.uint8)
    if motion_vectors.shape[0] > 0:
        mvs_x = motion_vectors[:, 5]
        mvs_y = motion_vectors[:, 6]
        mvs_x_motion = motion_vectors[:, 7] / motion_vectors[:, 9]
        mvs_y_motion = motion_vectors[:, 8] / motion_vectors[:, 9]
        xi = np.arange(8, math.ceil(frame_shape[0] / 16) * 16, 16)
        yi = np.arange(8, math.ceil(frame_shape[1] / 16) * 16, 16)
        mvs_motion_magnitude, mvs_motion_angle = cv2.cartToPolar(mvs_x_motion, -1.0*mvs_y_motion)
        mvs_motion_angle = mvs_motion_angle * 180 / (2 * np.pi)  # hue channel is [0, 180]
        mvs_motion_magnitude = cv2.normalize(mvs_motion_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mvs_motion_angle_interp = griddata((mvs_x, mvs_y), mvs_motion_angle, (xi[None, :], yi[:, None]), method=method)
        mvs_motion_magnitude_interp = griddata((mvs_x, mvs_y), mvs_motion_magnitude, (xi[None, :], yi[:, None]), method=method)
        motion_vectors_grid[:, :, 0] = np.squeeze(mvs_motion_angle_interp)
        motion_vectors_grid[:, :, 1] = np.squeeze(mvs_motion_magnitude_interp)
        motion_vectors_grid[:, :, 2] = 255
        motion_vectors_grid = cv2.cvtColor(motion_vectors_grid, cv2.COLOR_HSV2BGR)
    return motion_vectors_grid
