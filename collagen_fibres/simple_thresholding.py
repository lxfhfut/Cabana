import cv2
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import separate_stains, rgb_from_hed, rgb2hed, hed2rgb, rgb2gray
from skimage.filters import median
from skimage.morphology import disk

if __name__ == "__main__":
    img = np.array(Image.open(r"C:\Users\lxfhf\OneDrive\Documents\TWOMBLI\Input\K4145.vsi - 40x_BF_01Annotation (Polygon) (Malignant area)_4.tif"))
    ihc_hed = rgb2hed(img)
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    gray = (rgb2gray(ihc_e)*255).astype(np.uint8)

    print(np.mean(gray[gray<=180]))
    pos_ind = np.where(gray > 180)
    ihc_e[pos_ind[0], pos_ind[1], :] = [0,0,0]
    plt.imshow(ihc_e)
    plt.show()