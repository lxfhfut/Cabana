import os
import cv2
import numpy as np
import imageio.v3 as iio
from skimage import exposure
from utils import join_path
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import pandas as pd


class HDM:
    def __init__(self, max_hdm=220, sat_ratio=0.1, dark_line=False):
        self.max_hdm = max_hdm
        self.sat_ratio = sat_ratio
        self.dark_line = dark_line
        self.df_hdm = None

    def quantify_black_space(self, image_path, save_dir, ext=".png"):
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
            enhanced_image = self.enhance_contrast(img_path)
            cv2.imwrite(join_path(save_dir, os.path.basename(img_path)[:-4]+"_roi.png"), enhanced_image)
            img_names.append(os.path.basename(img_path)[:-4]+"_roi.png")
            hdm.append(np.count_nonzero(enhanced_image > 0)/np.prod(enhanced_image.shape[:2]))
        result_csv = join_path(save_dir, "ResultsHDM.csv")
        data = {'Image': img_names, '% HDM Area': hdm}
        self.df_hdm = pd.DataFrame(data)
        self.df_hdm.to_csv(result_csv, index=False)

    def enhance_contrast(self, image_path):
        raw_image = np.asarray(iio.imread(image_path))
        image = raw_image if raw_image.dtype == np.uint8 else cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image

        if self.dark_line:
            image = 255 - image

        image = np.clip(image, 0, self.max_hdm).astype(float)
        image = ((image - image.min()) / (image.max() - image.min() + np.finfo(float).eps) * 255).astype(np.uint8)
        percent_saturation = self.sat_ratio * 100
        pl, pu = np.percentile(image, (percent_saturation/2.0, 100-percent_saturation/2.0))
        enhanced_image = exposure.rescale_intensity(image, in_range=(pl, pu))

        return enhanced_image.astype(np.uint8)


