import os
import imageio

def mkgif():
    img_files = [x for x in os.listdir(os.path.dirname(__file__)) if x.endswith('.png')]
    img_files = [(x, imageio.imread(x)) for x in img_files]
    img_files = [(x.lstrip("img_").zfill(12), y) for (x, y) in img_files]
    img_files.sort(key=lambda x: x[0])
    img_files = [y for (x, y) in img_files]
    imageio.mimsave(os.path.join(os.path.dirname(__file__), 'result.gif'), img_files)