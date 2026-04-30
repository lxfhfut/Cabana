"""Shared per-image (and per-batch) pipeline stages.

Each function here is a pure-ish unit invoked by both ``Cabana`` (single image)
and ``BatchCabana`` (folder of images). Keeping the science in one place
prevents the two pipelines from drifting apart.
"""

import os
from pathlib import Path

import cv2
import numpy as np
import imageio.v3 as iio
from skimage.color import rgb2hed, hed2rgb, rgb2gray

from .hdm import HDM
from .detector import FibreDetector
from .utils import join_path


FIBRE_AREA_METRICS = (
    'Area (ROI)', '% ROI Area', 'Area (WIDTH)', '% WIDTH Area',
    'Mean Fibre Intensity (ROI)', 'Mean Fibre Intensity (WIDTH)',
    'Mean Fibre Intensity (HDM)',
)


def run_hdm(args, source_path, hdm_dir, ext='.png'):
    """Run HDM quantification on a single image path or a directory of images.

    Parameters
    ----------
    args : dict
        Loaded parameters YAML — needs Quantification + Detection sections.
    source_path : str
        Either a single image path or a directory containing images.
    hdm_dir : str
        Output directory for HDM artifacts and ``ResultsHDM.csv``.
    ext : str
        Image extension(s) to process. Default ``.png``.

    Returns
    -------
    pandas.DataFrame
        Per-image HDM quantification results (the same object stored on
        ``HDM.df_hdm`` after the call).
    """
    hdm = HDM(
        max_hdm=args["Quantification"]["Maximum Display HDM"],
        sat_ratio=args["Quantification"]["Contrast Enhancement"],
        dark_line=args["Detection"]["Dark Line"],
    )
    hdm.quantify_black_space(source_path, hdm_dir, ext=ext)
    return hdm.df_hdm


def compute_fibre_areas(img_mask_path, ori_img_path, width_mask_path, hdm_mask_path):
    """Compute the seven fibre-area / mean-intensity metrics for one image.

    Returns a dict keyed by ``FIBRE_AREA_METRICS``. If the ROI mask is empty,
    all values are 0.
    """
    img_mask = cv2.imread(img_mask_path, 0)
    area_roi = float(np.sum(img_mask > 128))
    if area_roi == 0:
        return dict.fromkeys(FIBRE_AREA_METRICS, 0)

    percent_roi = area_roi / img_mask.shape[0] / img_mask.shape[1]
    ori_img = iio.imread(ori_img_path)

    hed = rgb2hed(ori_img)
    null = np.zeros_like(hed[:, :, 0])
    ihc_e = hed2rgb(np.stack((null, hed[:, :, 1], null), axis=-1))
    red_img = (rgb2gray(ihc_e) * 255).astype(np.uint8)

    width_mask = cv2.imread(width_mask_path, 0)
    hdm_mask = cv2.imread(hdm_mask_path, 0)
    area_width = float(np.sum(width_mask < 128))
    percent_width = area_width / np.prod(width_mask.shape[:2])

    grayscale = (rgb2gray(ori_img) * 255).astype(np.uint8)
    if np.count_nonzero(red_img < 180):
        mean_intensity_roi = np.mean(red_img[(img_mask > 128) & (red_img < 180)])
        mean_intensity_width = np.mean(red_img[(width_mask < 128) & (red_img < 180)])
        mean_intensity_hdm = np.mean(red_img[(hdm_mask > 0) & (red_img < 180)])
    else:
        mean_intensity_roi = np.mean(grayscale[img_mask > 128])
        mean_intensity_width = np.mean(grayscale[width_mask < 128])
        mean_intensity_hdm = np.mean(grayscale[hdm_mask > 0])

    if np.isnan(mean_intensity_roi):
        mean_intensity_roi = np.mean(grayscale[img_mask > 128]) if np.any(img_mask > 128) else 0
    if np.isnan(mean_intensity_width):
        mean_intensity_width = np.mean(grayscale[width_mask < 128]) if np.any(width_mask < 128) else 0
    if np.isnan(mean_intensity_hdm):
        mean_intensity_hdm = np.mean(grayscale[hdm_mask > 0]) if np.any(hdm_mask > 0) else 0

    return {
        'Area (ROI)': area_roi,
        '% ROI Area': percent_roi,
        'Area (WIDTH)': area_width,
        '% WIDTH Area': percent_width,
        'Mean Fibre Intensity (ROI)': mean_intensity_roi,
        'Mean Fibre Intensity (WIDTH)': mean_intensity_width,
        'Mean Fibre Intensity (HDM)': mean_intensity_hdm,
    }


def build_fibre_detector(args):
    """Construct a FibreDetector from a parsed parameters dict."""
    d = args["Detection"]
    line_widths = np.arange(
        d["Min Line Width"],
        d["Max Line Width"] + d["Line Width Step"],
        d["Line Width Step"],
    )
    return FibreDetector(
        line_widths=line_widths,
        low_contrast=d["Low Contrast"],
        high_contrast=d["High Contrast"],
        dark_line=d["Dark Line"],
        extend_line=d["Extend Line"],
        correct_pos=False,
        min_len=d["Minimum Line Length"],
        max_len=d["Maximum Line Length"],
    )


def detect_one_image(det, roi_img_path, mask_dir, export_subdir, color_subdir,
                     mask_filename=None, artifact_stem=None):
    """Run detection on one ROI image and write the standard artifacts.

    Parameters
    ----------
    det : FibreDetector
    roi_img_path : str
        Path to the input ROI image.
    mask_dir : str
        Top-level Masks directory; receives the binary contour image.
    export_subdir : str
        Per-image directory under Exports/ for Mask/Width PNGs.
    color_subdir : str
        Per-image directory under Colors/ for color and gray Width PNGs.
    mask_filename : str, optional
        Filename for the mask file written to ``mask_dir``. Defaults to the
        basename of ``roi_img_path``.
    artifact_stem : str, optional
        Filename stem used for files written into ``export_subdir`` and
        ``color_subdir``. Defaults to ``mask_filename`` without extension.
        Cabana passes the original image stem (no ``_roi`` suffix);
        BatchCabana passes the roi-image stem.

    Returns
    -------
    dict
        ``{'contour_img', 'width_img', 'binary_contours', 'binary_widths',
        'int_width_img'}``.
    """
    det.detect_lines(roi_img_path)
    contour_img, width_img, binary_contours, binary_widths, int_width_img = det.get_results()

    base = mask_filename or os.path.basename(roi_img_path)
    stem = artifact_stem if artifact_stem is not None else os.path.splitext(base)[0]

    Path(export_subdir).mkdir(parents=True, exist_ok=True)
    Path(color_subdir).mkdir(parents=True, exist_ok=True)

    iio.imwrite(join_path(mask_dir, base), binary_contours)
    iio.imwrite(join_path(export_subdir, stem + "_Mask.png"), binary_contours)
    iio.imwrite(join_path(export_subdir, stem + "_Width.png"), binary_widths)
    iio.imwrite(join_path(color_subdir, stem + "_color_mask.png"), contour_img)
    iio.imwrite(join_path(color_subdir, stem + "_color_width.png"), width_img)
    iio.imwrite(join_path(color_subdir, stem + "_gray_width.png"), int_width_img)

    return {
        'contour_img': contour_img,
        'width_img': width_img,
        'binary_contours': binary_contours,
        'binary_widths': binary_widths,
        'int_width_img': int_width_img,
    }
