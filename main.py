import cv2
import numpy
import os
import numpy as np
from tqdm import tqdm
from functions import *
from makegif import mkgif
import time
import shutil
import random

dirpath = r"C:\Users\jonod\Documents\patterner_input\patterns3"
# ATM pattern sizes should be even numbers, otherwise code will not work
pad = 31
x_stride = 15
y_stride = 15
pat_x, pat_y = (30, 30)
start_x, start_y = (15, 15)
pattern_positions = []
target_image_path = r"C:\Users\jonod\Documents\patterner_input\lenna.png"
gradient_mode = "full"  # 'full' or 'stochastic'
gradient_solve = True
if any([x.endswith(".png") for x in os.listdir(os.path.dirname(__file__))]):
    pass
    mkgif()
    with open("oldartifacts/count.txt", 'r') as f:
        count = int(f.readlines()[0].strip())
    foldpath = os.path.join("oldartifacts", "set" + str(count))
    os.mkdir(foldpath)
    shutil.copy("result.gif", foldpath)
    img_files = [x for x in os.listdir(os.path.dirname(__file__)) if x.endswith('.png')]
    img_files = [(x, x.lstrip("img_").zfill(12)) for x in img_files]
    img_files.sort(key=lambda x: x[1])
    img = img_files[-1][0]
    shutil.copy(os.path.join(os.path.dirname(__file__), img), foldpath)
    for ele in [x[0] for x in img_files] + ["result.gif"]:
        os.remove(ele)
    count += 1
    os.remove("oldartifacts/count.txt")
    with open("oldartifacts/count.txt", 'w') as f:
        f.write(str(count))

target_image = cv2.imread(target_image_path)
x_max, y_max, _ = target_image.shape
# 0.0000001 / (amount of overlapping patterns) seems to lie around a max for decent convergence, higher seems to be better
learning_rate = 0.00000001
#learning_rate = 0.000001
epochs = 1000
save_each_epoch = True


# Load patterns
patterns = []
for path in os.listdir(dirpath):
    loaded_pattern = cv2.imread(os.path.join(dirpath, path))
    patterns.append(loaded_pattern)
    for degrees in [22.5 * x for x in range(0, 15)]:
        rotated_pattern = rotate(loaded_pattern, degrees)
        patterns.append(rotated_pattern)

# Calculate pattern positions
x0 = start_x
y0 = start_y
while y0 < y_max:
    while x0 < x_max:
        pattern_positions.append((x0, y0))
        x0 += x_stride
    x0 = start_x
    y0 += y_stride
if gradient_solve:
    # Initialize activation values
    activation_values = [[0 for x in patterns] for y in pattern_positions]


    # gradient descent to replicate target image using patterns
    for epoch in tqdm(range(0, epochs)):
        img = np.zeros((x_max + pad*2, y_max + pad*2, 3))
        for i0 in range(len(activation_values)):
            for i1 in range(len(activation_values[0])):
                x, y = pattern_positions[i0]
                x, y = (x + pad, y + pad)
                pat = patterns[i1]
                act = activation_values[i0][i1]
                img, _, _ = paint_pattern(img, pat, (x, y), act)
        img = img[pad:pad + x_max, pad:pad + y_max]
        if save_each_epoch:
            write_img = np.copy(img)
            write_img[write_img > 255.0] = 255.0
            write_img[write_img < 0.0] = 0.0
            cv2.imwrite("img_" + str(epoch) + ".png", write_img)

        # Pixelwise image gradient
        l1_error_gradient = target_image - img
        print(np.sum(np.abs(l1_error_gradient)))
        # clip gradient for stability
        l1_error_gradient[l1_error_gradient > 1.0] = 1.0
        l1_error_gradient[l1_error_gradient < -1.0] = -1.0

        _temp_new_gradient = np.zeros((x_max+pad*2, y_max+pad*2, 3))
        _temp_new_gradient[pad:x_max+pad, pad:y_max+pad] = l1_error_gradient
        l1_error_gradient = _temp_new_gradient
        # Update to every activation value
        # Se på dette: https://math.berkeley.edu/~peyam/Math54Fa11/Homeworks/Homework%207%20-%20Solutions.pdf
        # Essentially we are trying to find the point closest to the image point (which is in the vector space of color images)
        # in the subspace spanned by the patterns.
        if gradient_mode == "full":
            for i0 in range(len(activation_values)):
                x, y = pattern_positions[i0]
                x, y = (x + pad, y + pad)
                for i1 in range(len(activation_values[0])):
                    pat = patterns[i1]
                    mask = np.zeros((x_max + pad*2, y_max + pad*2, 3))
                    mask, (xt, yt), (xb, yb) = paint_pattern(mask, pat, (x, y), 1.0)
                    mask = mask[xt:xb, yt:yb]
                    local_error_gradient = l1_error_gradient[xt:xb, yt:yb]
                    error = np.sum(np.multiply(local_error_gradient, mask))*learning_rate
                    activation_values[i0][i1] += error
        elif gradient_mode == "stochastic":
            i0 = random.randint(0, len(activation_values) - 1)
            i1 = random.randint(0, len(activation_values[0]) - 1)
            x, y = pattern_positions[i0]
            x, y = (x + pad, y + pad)
            pat = patterns[i1]
            mask = np.zeros((x_max + pad*2, y_max + pad*2, 3))
            mask, (xt, yt), (xb, yb) = paint_pattern(mask, pat, (x, y), 1.0)
            mask = mask[xt:xb, yt:yb]
            local_error_gradient = l1_error_gradient[xt:xb, yt:yb]
            error = np.sum(np.multiply(local_error_gradient, mask))*learning_rate
            activation_values[i0][i1] += error
else:
    def _p(x):
        return (x[0]+pad, x[1]+pad)
    # We want to map the image onto the subspace of all images spanned by the patterns
    # This can be solved using least square linear equations solvers
    # However this is storage intensive, as each pattern for each location requires storing the entire image
    # size
    if pat_x <= x_stride and pat_y <= y_stride:
        print("Begin linalg non-overlap solve")
        # If the patterns at different locations are strictly non-overlapping the problem decomposes into optimizing
        # for a linear equation at each position, which is an easier problem
        img = np.zeros((x_max+pad, y_max+pad, 3))
        for point in pattern_positions:
            x, y = point
            pat_left_imgviewport = int(math.floor(x - pat_x / 2))
            pat_right_imgviewport = int(math.ceil(x + pat_x / 2))
            pat_top_imgviewport = int(math.floor(y - pat_y / 2))
            pat_bot_imgviewport = int(math.ceil(y + pat_y / 2))
            pat_left_imgviewport = max(0, pat_left_imgviewport)
            pat_right_imgviewport = min(x_max, pat_right_imgviewport)
            pat_top_imgviewport = max(0, pat_top_imgviewport)
            pat_bot_imgviewport = min(y_max, pat_bot_imgviewport)
            _patterns = []
            for pat in patterns:
                to_paint_in = np.zeros((pat_x, pat_y, 3))
                pattern, _, _ = paint_pattern(to_paint_in, pat, (pat_x/2, pat_y/2), 1.0)
                _patterns.append(pattern[:pat_right_imgviewport-pat_left_imgviewport,:pat_bot_imgviewport-pat_top_imgviewport].flatten())
            patterns2 = np.asarray(_patterns).transpose()

            local_target = target_image[pat_left_imgviewport:pat_right_imgviewport, pat_top_imgviewport:pat_bot_imgviewport].flatten()
            activations = np.linalg.lstsq(patterns2, local_target)[0]

            produced_image = patterns2 @ activations
            produced_image = produced_image.reshape((pat_right_imgviewport-pat_left_imgviewport, pat_bot_imgviewport-pat_top_imgviewport, 3)).astype(int)
            img[pat_left_imgviewport:pat_right_imgviewport, pat_top_imgviewport:pat_bot_imgviewport] = produced_image
        cv2.imwrite("result_img.png", img)
    else:
        print("Begin linalg overlap solve")
        target_image = target_image.flatten()
        #target_image = target_image.reshape((target_image.shape[0], 1))
        _patterns = []
        for pos in pattern_positions:
            pos = _p(pos)
            for pat in patterns:
                _patterns.append(paint_pattern(init_img(x_max+pad*2, y_max+pad*2), pat, pos, 1.0)[0]
                                 [pad:x_max+pad,pad:y_max+pad].flatten())
        print("positions loaded")
        patterns = np.asarray(_patterns).transpose()
        print("Calculate activations")
        activations = np.linalg.lstsq(patterns, target_image)[0]
        print("Reproduce output")
        produced_image = patterns @ activations
        produced_image = produced_image.reshape((x_max, y_max, 3)).astype(int)
        cv2.imwrite("result_img.png", produced_image)
