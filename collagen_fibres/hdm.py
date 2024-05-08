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

    img_names = []
    hdm = []
    for img_path in img_paths:
        enhanced_image = enhance_contrast(img_path, max_hdm=max_hdm, sat_ratio=sat_ratio, dark_line=dark_line)
        cv2.imwrite(join_path(save_dir, os.path.basename(img_path)[:-4]+"_roi.png"), enhanced_image)
        img_names.append(os.path.basename(img_path)[:-4]+"_roi.png")
        hdm.append(np.count_nonzero(enhanced_image > 0)/np.prod(enhanced_image.shape[:2]))
    result_csv = join_path(save_dir, "ResultsHDM.csv")
    data = {'Image': img_names, '% HDM Area': hdm}
    df_hdm = pd.DataFrame(data)
    df_hdm.to_csv(result_csv, index=False)
    return df_hdm


def enhance_contrast(image_path, max_hdm=220, sat_ratio=0.1, dark_line=False):
    raw_image = np.asarray(iio.imread(image_path))
    image = raw_image if raw_image.dtype == np.uint8 else cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

    if dark_line:
        image = 255 - image

    image = np.clip(image, 0, max_hdm).astype(float)
    image = ((image - image.min()) / (image.max() - image.min() + np.finfo(float).eps) * 255).astype(np.uint8)
    percent_saturation = sat_ratio * 100
    pl, pu = np.percentile(image, (percent_saturation/2.0, 100-percent_saturation/2.0))
    enhanced_image = exposure.rescale_intensity(image, in_range=(pl, pu))

    return enhanced_image.astype(np.uint8)


if __name__ == "__main__":
    directory = "/Users/lxfhfut/Downloads/input/540.vsi - 20x_BF multi-band_01Annotation (Ellipse) (Tumor)_0.tif"
    quantify_black_space(directory, ext=['.png', '.tif'], max_hdm=220, sat_ratio=0.0, dark_line=True)

