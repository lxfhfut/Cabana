import os
import cv2
import numpy as np
import imageio.v3 as iio
from skimage import exposure
from utils import join_path
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import pandas as pd


def quantify_black_space(image_path, save_dir, ext=".png", max_hdm=220, sat_ratio=0.1, dark_line=False):
    ext_list = [ext.lower()] if isinstance(ext, str) else [e.lower() for e in ext]

    # Check if the path is a directory or a single file
    img_paths = []
    if os.path.isdir(image_path):
        for f in os.listdir(image_path):
            if any(f.lower().endswith(e) for e in ext_list):
                img_paths.append(os.path.join(image_path, f))
    elif os.path.isfile(image_path) and any(image_path.lower().endswith(e) for e in ext_list):
        img_paths.append(image_path)

    header = ["Image Name", "% Black"]
    df = pd.DataFrame(columns=header)
    for img_path in img_paths:
        enhanced_image = enhance_contrast(img_path, max_hdm=max_hdm, sat_ratio=sat_ratio, dark_line=dark_line)
        cv2.imwrite(join_path(save_dir, os.path.basename(img_path)[:-4]+"_roi.png"),
                    (enhanced_image > 0).astype(np.uint8)*255)
        df.loc[len(df)] = [os.path.basename(img_path),
                           np.count_nonzero(enhanced_image > 0)/np.prod(enhanced_image.shape[:2])]
    result_csv = join_path(save_dir, "_ResultsHDM.csv")
    df.to_csv(result_csv, index=False)
        # print(f"Image name {os.path.basename(img_path)}: "
        #       f"% HDM = {np.count_nonzero(enhanced_image>0)/np.prod(enhanced_image.shape[:2])}")


def enhance_contrast(image_path, max_hdm=220, sat_ratio=0.1, dark_line=False):
    # Load the image
    raw_image = np.asarray(iio.imread(image_path))

    if len(raw_image.shape) == 3:
       image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2GRAY)
    else:
        image = raw_image

    if image.dtype != np.uint8:
        # Normalize the image to 0-255 and convert to uint8
        image = ((image - image.min()) / (image.max() - image.min() + np.finfo(float).eps) * 255).astype(np.uint8)

    if not dark_line:
        image = 255 - image

    image = np.clip(image, 0, max_hdm).astype(float)
    image = 255 - ((image - image.min()) / (image.max() - image.min() + np.finfo(float).eps) * 255).astype(np.uint8)
    percent_saturation = sat_ratio * 100
    pl, pu = np.percentile(image, (percent_saturation/2.0, 100-percent_saturation/2.0))
    enhanced_image = exposure.rescale_intensity(image, in_range=(pl, pu))

    return enhanced_image.astype(np.uint8)


if __name__ == "__main__":
    directory = "/Users/lxfhfut/Downloads/input/540.vsi - 20x_BF multi-band_01Annotation (Ellipse) (Tumor)_0.tif"
    quantify_black_space(directory, ext=['.png', '.tif'], max_hdm=220, sat_ratio=0.0, dark_line=True)

