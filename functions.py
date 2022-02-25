import cv2
import math
import numpy as np

# rotate function open source from internet
def rotate(image, theta):
    """
    Rotates image
    :param image: image to rotate
    :param theta: degrees to rotate
    :return: rotated image
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, theta, 1)
    rotated = cv2.warpAffine(image, M, (int(w), int(h)), cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated


def crop_transform(image, pattern, center_xy):
    """
    Crops pattern such that it fits in image
    :param image: image to paint on
    :param pattern: pattern to paint
    :param center_xy: the xy coords on image where the center of pattern should be placed
    :return: cropped pattern, left, right, top, bot in img viewport
    """
    img_dim_x, img_dim_y, _ = image.shape
    pat_dim_x, pat_dim_y, _ = pattern.shape
    (x, y) = center_xy
    pat_left_imgviewport = int(math.floor(x - pat_dim_x/2))
    pat_right_imgviewport = int(math.ceil(x + pat_dim_x/2))
    pat_top_imgviewport = int(math.floor(y - pat_dim_y/2))
    pat_bot_imgviewport = int(math.ceil(y + pat_dim_y/2))

    cropleft = -pat_left_imgviewport if pat_left_imgviewport < 0 else 0
    if pat_right_imgviewport > img_dim_x:
        cropright = pat_dim_x - (pat_right_imgviewport - img_dim_x)
    else:
        cropright = pat_right_imgviewport
    croptop = -pat_top_imgviewport if pat_top_imgviewport < 0 else 0
    if pat_bot_imgviewport > img_dim_y:
        cropbot = pat_dim_y - (pat_bot_imgviewport - img_dim_y)
    else:
        cropbot = pat_bot_imgviewport

    modified_pattern = pattern[cropleft:cropright, croptop:cropbot]

    pat_left_imgviewport = max(pat_left_imgviewport, 0)
    pat_top_imgviewport = max(pat_top_imgviewport, 0)
    pat_bot_imgviewport = min(pat_bot_imgviewport, img_dim_y)
    pat_right_imgviewport = min(pat_right_imgviewport, img_dim_x)

    return modified_pattern, pat_left_imgviewport, pat_right_imgviewport, pat_top_imgviewport, pat_bot_imgviewport

def init_img(x, y):
    return np.zeros((x, y, 3))

def paint_pattern(image, pattern, center_xy, alpha):
    """
    paints a pattern on an image, crops pattern if out of bounds
    :param image: image to paint on
    :param pattern: pattern to paint
    :param center_xy: the xy coords on image where the center of pattern should be placed
    :param alpha: Strength of pattern that is painted (in transparency)
    :return: image with pattern painted on it
    """
    pat_dim_x, pat_dim_y, _ = pattern.shape
    (x, y) = center_xy
    if pat_dim_x % 2 == 0:
        pat_left_imgviewport = int(math.floor(x - pat_dim_x / 2))
        pat_right_imgviewport = int(math.ceil(x + pat_dim_x / 2))
    else:
        pat_left_imgviewport = int(math.floor(x - pat_dim_x / 2))
        pat_right_imgviewport = int(math.floor(x + pat_dim_x / 2))
    if pat_dim_x % 2 == 0:
        pat_top_imgviewport = int(math.floor(y - pat_dim_y / 2))
        pat_bot_imgviewport = int(math.ceil(y + pat_dim_y / 2))    
    else:
        pat_top_imgviewport = int(math.floor(y - pat_dim_y / 2))
        pat_bot_imgviewport = int(math.floor(y + pat_dim_y / 2))

    image[pat_left_imgviewport:pat_right_imgviewport, pat_top_imgviewport:pat_bot_imgviewport] += pattern * alpha
    return image, (pat_left_imgviewport, pat_top_imgviewport), (pat_right_imgviewport, pat_bot_imgviewport)

def linalg_solve_img(target_image, pattern_positions, patterns, pad, x_max, y_max):
    target_image = target_image.flatten()
    #target_image = target_image.reshape((target_image.shape[0], 1))
    def _p(x):
        return (x[0]+pad, x[1]+pad)
    _patterns = []
    for pos in pattern_positions:
        pos = _p(pos)
        for pat in patterns:
            _patterns.append(paint_pattern(init_img(x_max+pad*2, y_max+pad*2), pat, pos, 1.0)[0]
                             [pad:x_max+pad,pad:y_max+pad].flatten())
    patterns = np.asarray(_patterns).transpose()
    activations = np.linalg.lstsq(patterns, target_image)[0]
    produced_image = patterns @ activations
    produced_image = produced_image.reshape((x_max, y_max, 3)).astype(int)
    return produced_image
