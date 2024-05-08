import os
import time

import yaml
import csv
import cv2
import shutil
import numpy as np
import pandas as pd
from glob import glob
from log import Log
from PIL import Image
import tifffile as tiff
import imageio.v3 as iio
from detector import FibreDetector
from analyzer import SkeletonAnalyzer
from hdm import HDM
from orientation import OrientationAnalyzer

from pathlib import Path
from utils import split2batches, mask_color_map, orient_vf, info_color_map, sanitize_filename
from utils import create_folder, join_path, sbs_color_map, sbs_color_survey, width_color_map, get_img_paths
from skimage.feature import peak_local_max
from sklearn.metrics.pairwise import euclidean_distances
from segmenter import parse_args, segment_single_image, visualize_fibres
from skimage.color import rgb2hed, hed2rgb, rgb2gray

from tqdm import tqdm
from utils import read_bar_format, array_divide


class Cabana:
    def __init__(self, program_folder, input_folder, out_folder,
                 batch_size=5, batch_idx=0, ignore_large=True):
        self.param_file = "Parameters.yml"

        self.args = None  # args for Cabana program
        self.seg_args = parse_args()  # args for segmentation
        self.ims_res = 1.0  # µm/pixel
        self.df_stats = pd.DataFrame()

        # self.ij = ij
        self.program_folder = program_folder
        self.input_folder = input_folder
        self.output_folder = out_folder
        self.batch_idx = batch_idx
        self.batch_size = batch_size
        self.ignore_large = ignore_large

        # Create sub-folders in output directory
        self.roi_dir = join_path(self.output_folder, 'ROIs', "")
        self.bin_dir = join_path(self.output_folder, 'Bins', "")
        self.mask_dir = join_path(self.output_folder, 'Masks', "")
        self.hdm_dir = join_path(self.output_folder, 'HDM', "")
        self.export_dir = join_path(self.output_folder, 'Exports', "")
        self.color_dir = join_path(self.output_folder, 'Colors', "")
        self.fibre_dir = join_path(self.output_folder, 'Fibres', "")
        self.eligible_dir = join_path(self.output_folder, 'Eligible')

        create_folder(self.eligible_dir)
        create_folder(self.fibre_dir)
        create_folder(self.mask_dir)
        create_folder(self.hdm_dir)
        create_folder(self.export_dir)
        create_folder(self.color_dir)
        create_folder(self.roi_dir)
        create_folder(self.bin_dir)
        setattr(self.seg_args, 'roi_dir', self.roi_dir)
        setattr(self.seg_args, 'bin_dir', self.bin_dir)

    def initialize_params(self):
        param_path = join_path(self.program_folder, self.param_file)
        with open(param_path) as pf:
            try:
                self.args = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                print(exc)

        # overwrite specific fields of seg_args with those in the parameter file
        setattr(self.seg_args, 'num_channels', int(self.args['Segmentation']["Number of Labels"]))
        setattr(self.seg_args, 'max_iter', int(self.args['Segmentation']["Max Iterations"]))
        setattr(self.seg_args, 'hue_value', float(self.args['Segmentation']["Normalized Hue Value"]))
        setattr(self.seg_args, 'rt', float(self.args['Segmentation']["Color Threshold"]))
        setattr(self.seg_args, 'max_size', int(self.args['Segmentation']["Max Size"]))
        setattr(self.seg_args, 'min_size', int(self.args['Segmentation']["Min Size"]))

    def remove_large_images(self):
        img_paths = get_img_paths(self.input_folder)
        path_batches, res_batches = split2batches(img_paths, self.batch_size)

        max_line_width = self.args['Detection']["Max Line Width"]
        max_size = self.args["Segmentation"]["Max Size"]
        with open(join_path(self.eligible_dir, "IgnoredImages.txt"), 'w') as f:
            count_black = 0
            count_oversized = 0
            for img_path in path_batches[self.batch_idx]:
                img_name = os.path.basename(img_path)
                img_name = sanitize_filename(img_name)  # replace special characters in image names with "_"
                img = cv2.imread(img_path)

                # if gray image convert to rgb
                img = np.repeat(img[:, :, np.newaxis], 3, axis=2) if len(img.shape) < 3 else img

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                height, width = img.shape[:2]
                bright_percent = np.sum(gray > 5) / width / height
                if np.min([height, width]) < 2 * max_line_width or bright_percent <= 0.01:
                    count_black += 1
                    f.write(img_path + "\n")
                elif height*width <= max_size**2 and bright_percent > 0.01:
                    cv2.imwrite(join_path(self.eligible_dir, os.path.splitext(img_name)[0] + ".png"), img)
                elif height*width > max_size and bright_percent > 0.01:
                    if self.ignore_large:
                        count_oversized += 1
                        f.write(img_path+"\n")
                    else:
                        row_blk_sz = int(np.ceil(height / int(np.ceil(height / max_size))))
                        col_blk_sz = int(np.ceil(width / int(np.ceil(width / max_size))))
                        count_oversized += 1
                        for i in range(0, height, row_blk_sz):
                            for j in range(0, width, col_blk_sz):
                                box = (j, i, j + col_blk_sz, i + row_blk_sz)
                                if j + col_blk_sz > width:
                                    box = (width - col_blk_sz, i, width, i + row_blk_sz)
                                if i + row_blk_sz > height:
                                    box = (j, height - row_blk_sz, j + col_blk_sz, height)

                                img_crop = img[box[1]:box[3], box[0]:box[2]]
                                cv2.imwrite(join_path(
                                    self.eligible_dir, img_name[:-4] + "_blk_{}_{}.png".format(
                                        i//row_blk_sz, j//col_blk_sz)), img_crop)

        if count_black > 0:
            Log.logger.warning('{} black or small-sized images have been ignored.'.format(count_black))
        if count_oversized > 0:
            msg = "ignored" if self.ignore_large else "split into smaller blocks"
            Log.logger.warning('{} over-sized images have been {}.'.format(count_oversized, msg))

        self.input_folder = self.eligible_dir
        self.ims_res = res_batches[self.batch_idx]

        # Log.logger.info(f"Image resolution of Batch {self.batch_idx+1} "
        #                 f"is {str(res_batches[self.batch_idx])}µm/pixel")

    def generate_rois(self):
        img_paths = get_img_paths(self.input_folder)
        img_paths.sort()
        if self.args["Initialization"]["Segmentation"]:
            Log.logger.info('Segmenting {} images in {}'.format(len(img_paths), self.input_folder))
            for img_path in img_paths:
                setattr(self.seg_args, 'input', img_path)
                segment_single_image(self.seg_args)
        else:
            Log.logger.info("No segmentation is applied prior to image analysis.")
            for img_path in img_paths:
                setattr(self.seg_args, 'input', img_path)
                img = cv2.imread(self.seg_args.input)
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                img_name = os.path.splitext(os.path.basename(self.seg_args.input))[0]
                cv2.imwrite(join_path(self.seg_args.roi_dir, img_name + '_roi.png'), img)
                cv2.imwrite(join_path(self.seg_args.bin_dir, img_name + '_mask.png'), mask)

        Log.logger.info('ROIs have been saved in {}'.format(self.seg_args.roi_dir))
        Log.logger.info('Masks have been saved in {}'.format(self.seg_args.bin_dir))

    def detect_fibres(self):
        dark_line = self.args["Detection"]["Dark Line"]
        extend_line = self.args["Detection"]["Extend Line"]
        correct_pos = self.args["Detection"]["Correct Position"]
        min_line_width = self.args["Detection"]["Min Line Width"]
        max_line_width = self.args["Detection"]["Max Line Width"]
        line_width_step = self.args["Detection"]["Line Width Step"]
        line_widths = np.arange(min_line_width, max_line_width + line_width_step, line_width_step)
        low_contrast = self.args["Detection"]["Low Contrast"]
        high_contrast = self.args["Detection"]["High Contrast"]
        min_len = self.args["Detection"]["Minimum Branch Length"]
        Log.logger.info(f"Detecting fibres with line widths: {line_widths} pixels "
                        f"for {len(glob(join_path(self.roi_dir, '*.png')))} images.")
        det = FibreDetector(line_widths=line_widths,
                            low_contrast=low_contrast,
                            high_contrast=high_contrast,
                            dark_line=dark_line,
                            extend_line=extend_line,
                            correct_pos=correct_pos,
                            min_len=min_len)
        for img_path in tqdm(glob(join_path(self.roi_dir, '*.png')), bar_format=read_bar_format):
            det.detect_lines(img_path)
            contour_img, width_img, binary_contours, binary_widths = det.get_results()

            ori_img_name = os.path.basename(img_path)
            name_wo_ext = ori_img_name[:ori_img_name.rindex('.')]

            # export binary contours to 'Mask' folder
            iio.imwrite(join_path(self.mask_dir, ori_img_name), binary_contours)

            # export binary contours and widths to 'Export' folder
            Path(join_path(self.export_dir, name_wo_ext)).mkdir(parents=True, exist_ok=True)
            iio.imwrite(
                join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Mask.png"), binary_contours)
            iio.imwrite(
                join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Width.png"), binary_widths)

            # export colorized contours and widths to 'Color' folder
            Path(join_path(self.color_dir, name_wo_ext)).mkdir(parents=True, exist_ok=True)
            iio.imwrite(
                join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_mask.png"), contour_img)
            iio.imwrite(
                join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_width.png"), width_img)

    def analyze_orientations(self):
        orient_analyzer = OrientationAnalyzer(2.0)
        alignments = []
        variances = []
        img_names = []
        Log.logger.info(f"Analyzing orientations for {len(glob(join_path(self.roi_dir, '*.png')))} images.")
        for img_path in tqdm(glob(join_path(self.roi_dir, '*.png')), bar_format=read_bar_format):
            ori_img_name = os.path.basename(img_path)
            img_names.append(ori_img_name)
            name_wo_ext = ori_img_name[:ori_img_name.rindex('.')]
            mask = iio.imread(join_path(self.bin_dir, name_wo_ext[:-4]+"_mask.png"))
            orient_analyzer.compute_orient(img_path)
            alignments.append(orient_analyzer.mean_coherency(mask=mask))
            variances.append(orient_analyzer.circular_variance(mask=mask))

            # export visualizations to 'Export' folder
            iio.imwrite(join_path(
                self.export_dir, name_wo_ext, name_wo_ext + "_Energy.tif"),
                orient_analyzer.get_energy_image(mask))
            iio.imwrite(join_path(
                self.export_dir, name_wo_ext, name_wo_ext + "_Coherency.tif"),
                orient_analyzer.get_orientation_image(mask))
            iio.imwrite(join_path(
                self.export_dir, name_wo_ext, name_wo_ext + "_Orientation.tif"),
                orient_analyzer.get_orientation_image(mask))
            iio.imwrite(join_path(
                self.export_dir, name_wo_ext, name_wo_ext + "_Color_Survey.tif"),
                orient_analyzer.draw_color_survey(mask))

            # export vector fields and angular hists to 'Color' folder
            iio.imwrite(join_path(
                self.color_dir, name_wo_ext, name_wo_ext + "_orient_vf.png"),
                orient_analyzer.draw_vector_field(mask/255.0))
            iio.imwrite(join_path(
                self.color_dir, name_wo_ext, name_wo_ext + "_angular_hist.png"),
                orient_analyzer.draw_angular_hist(mask=mask))

        data = {'Image': img_names,
                'Orient. Alignment': alignments,
                'Orient. Variance': variances}
        df_orient = pd.DataFrame(data)
        self.df_stats = self.df_stats.merge(df_orient, on="Image")

    def quantify_skeletons(self):
        min_skel_size = round(self.args["Quantification"]["Minimum Skeleton Size (µm)"] / self.ims_res)
        min_branch_len = round(self.args["Quantification"]["Minimum Branch Length (µm)"] / self.ims_res)
        min_hole_area = round(self.args["Quantification"]["Minimum Hole Area (µm²)"] / self.ims_res ** 2)
        min_curve_win = self.args["Quantification"]["Minimum Curvature Window (µm)"]
        max_curve_win = self.args["Quantification"]["Maximum Curvature Window (µm)"]
        curve_win_step = self.args["Quantification"]["Curvature Window Step (µm)"]

        Log.logger.info(f"min skeleton size = {min_skel_size} px, "
                        f"min branch length = {min_branch_len} px, "
                        f"min hole area = {min_hole_area} px²")
        # fibre detection returns fibres in black color, so dark_line is set to "True" for skeleton analysis
        skel_analyzer = SkeletonAnalyzer(skel_thresh=min_skel_size,
                                         branch_thresh=min_branch_len,
                                         hole_threshold=min_hole_area,
                                         dark_line=True)
        img_names, end_points, branch_points, growth_units = [], [], [], []
        proj_areas, lacunarities, total_lengths, frac_dims, total_areas = [], [], [], [], []
        curvatures = {}
        for win_sz in np.arange(min_curve_win, max_curve_win + curve_win_step, curve_win_step):
            curvatures[f"Curvature (win_sz={win_sz:.0f}µm)"] = []

        Log.logger.info(f"Quantifying skeletons for {len(glob(join_path(self.mask_dir, '*.png')))} images.")
        for img_path in tqdm(glob(join_path(self.mask_dir, '*.png')), bar_format=read_bar_format):
            skel_analyzer.reset()
            skel_analyzer.analyze_image(img_path)
            ori_img_name = os.path.basename(img_path)
            img_names.append(ori_img_name)
            end_points.append(skel_analyzer.num_tips)
            branch_points.append(skel_analyzer.num_branches)
            growth_units.append(skel_analyzer.growth_unit * self.ims_res)
            proj_areas.append(skel_analyzer.proj_area * self.ims_res**2)
            lacunarities.append(skel_analyzer.lacunarity)
            total_lengths.append(skel_analyzer.total_length * self.ims_res)
            frac_dims.append(skel_analyzer.frac_dim)
            total_areas.append(np.prod(skel_analyzer.raw_image.shape[:2]) * self.ims_res**2)

            # export images to 'Export' folder
            name_wo_ext = ori_img_name[:ori_img_name.rindex('.')]
            iio.imwrite(
                join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Skeleton.png"),
                skel_analyzer.key_pts_image)

            iio.imwrite(
                join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Length_Map.tif"),
                skel_analyzer.length_map_all)

            # calculate curvatures for various window sizes
            for win_sz in np.arange(min_curve_win, max_curve_win + curve_win_step, curve_win_step):
                skel_analyzer.calc_curve_all(round(win_sz / self.ims_res))
                curvatures[f"Curvature (win_sz={win_sz:.0f}µm)"].append(skel_analyzer.avg_curve_all)
                iio.imwrite(join_path(
                    self.export_dir, name_wo_ext, f"{name_wo_ext}_Curve_Map_{win_sz:.0f}.tif"),
                    skel_analyzer.curve_map_all)

        data = {'Image': img_names,
                'Area of Fibre Spines (µm²)': proj_areas,
                'Lacunarity': lacunarities,
                'Total Length (µm)': total_lengths,
                'Endpoints': end_points,
                'HGU (µm)': growth_units,
                'Branchpoints': branch_points,
                'Box-Counting Fractal Dimension': frac_dims,
                'Total Image Area (µm²)': total_areas
                }
        data.update(curvatures)
        df_skel = pd.DataFrame(data)
        self.df_stats = self.df_stats.merge(df_skel, on="Image")

    def quantify_hdm(self):
        Log.logger.info(f"Quantifying High Density Matrix (HDM) areas for "
                        f"{len(glob(join_path(self.eligible_dir, '*.png')))} images.")
        max_hdm = self.args["Quantification"]["Maximum Display HDM"],
        dark_line = self.args["Detection"]["Dark Line"]
        hdm = HDM(max_hdm=max_hdm, sat_ratio=0.1, dark_line=dark_line)
        hdm.quantify_black_space(self.eligible_dir, self.hdm_dir, ext=".png")
        self.df_stats = hdm.df_hdm

    def quantify_images(self):
        if len(glob(join_path(self.roi_dir, '*.png'))) > 0:
            self.quantify_hdm()  # HDM
            self.detect_fibres()  # Ridge detection
            self.quantify_skeletons()  # Skeleton analysis
            self.analyze_orientations()  # Orientation analysis
            Log.logger.info('Image analysis finished.')

    def visualize_fibres(self, thickness=3):
        Log.logger.info('Generating fibre visualization results.')
        img_paths = get_img_paths(self.input_folder)
        img_paths.sort()
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            mask_path = join_path(self.mask_dir, os.path.splitext(img_name)[0] + '_roi.png')
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            visualize_fibres(img, mask,
                             join_path(self.fibre_dir, os.path.splitext(img_name)[0] + '_fibres.png'), thickness)

    def calc_fibre_areas(self):
        img_paths = glob(join_path(self.bin_dir, '*.png'))
        img_paths.sort()
        with open(join_path(self.bin_dir, 'ResultsROI.csv'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Image', 'Area (ROI)', '% ROI Area', 'Area (WIDTH)', '% WIDTH Area',
                 'Mean Fibre Intensity (ROI)',
                 'Mean Fibre Intensity (WIDTH)',
                 'Mean Fibre Intensity (HDM)'])
            Log.logger.info('Calculating fibre areas for {} images.'.format(len(img_paths)))
            for img_path in img_paths:
                img_mask = cv2.imread(img_path, 0)
                area_roi = np.sum(img_mask > 128).astype(float)  # ROI area
                percent_roi = area_roi / img_mask.shape[0] / img_mask.shape[1]  # % ROI area
                name = os.path.basename(img_path)
                ori_img = iio.imread(join_path(self.eligible_dir, name[:-9] + ".png"))

                hed = rgb2hed(ori_img)
                null = np.zeros_like(hed[:, :, 0])
                ihc_e = hed2rgb(np.stack((null, hed[:, :, 1], null), axis=-1))
                red_img = (rgb2gray(ihc_e) * 255).astype(np.uint8)

                name = name[:-9] + '_roi.png'
                name_wo_ext = name[:-4]
                width_mask = cv2.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext+"_Width.png"), 0)
                hdm_mask = cv2.imread(join_path(self.hdm_dir, name), 0)
                area_width = np.sum(width_mask < 128).astype(float)  # WIDTH area
                percent_width = area_width / np.product(width_mask.shape[:2])  # % WIDTH area
                if np.count_nonzero(red_img < 180):
                    mean_intensity_roi = np.mean(red_img[(img_mask > 128) & (red_img < 180)])
                    mean_intensity_width = np.mean(red_img[(width_mask < 128) & (red_img < 180)])
                    mean_intensity_hdm = np.mean(red_img[(hdm_mask > 0) & (red_img < 180)])
                else:
                    grayscale = (rgb2gray(ori_img)*255).astype(np.uint8)
                    mean_intensity_roi = np.mean(grayscale[img_mask > 128])
                    mean_intensity_width = np.mean(grayscale[width_mask < 128])
                    mean_intensity_hdm = np.mean(grayscale[hdm_mask > 0])

                data = [name, area_roi, percent_roi, area_width, percent_width,
                        mean_intensity_roi, mean_intensity_width, mean_intensity_hdm]
                writer.writerow(data)

            Log.logger.info('Areas have been saved in {}'.format(join_path(self.bin_dir, 'ResultsROI.csv')))

    def analyze_all_gaps(self):
        min_gap_diameter = -1
        if "Minimum Gap Diameter (pixels)" in self.args["Quantification"].keys():
            min_gap_diameter = self.args["Quantification"]["Minimum Gap Diameter (pixels)"]
        if min_gap_diameter == -1:
            Log.logger.warning("'Minimum Gap Diameter can not be found in {}.".format(self.param_file))
        elif min_gap_diameter == 0:
            Log.logger.warning("Skipping gap analysis.")
            return
        else:
            Log.logger.info(
                f"Performing gap analysis with "
                f"minimum gap diameter = {min_gap_diameter*self.ims_res:.1f}µm "
                f"({min_gap_diameter} pixels).")

            gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
            Path(gap_result_dir).mkdir(parents=True, exist_ok=True)
            # gap_analysis_file = join_path(gap_result_dir, "GapAnalysisSummary.txt")
            # with open(gap_analysis_file, 'w+') as summary_file:
            img_paths = glob(join_path(self.mask_dir, '*.png'))
            names, means, stds, percentile5, median, percentile95, counts = [], [], [], [], [], [], []
            for img_path in img_paths:
                min_gap_radius = min_gap_diameter / 2
                min_dist = int(np.max([1, min_gap_radius]))
                img = cv2.imread(img_path, 0)
                mask = img.copy()

                # set border pixels to zero to avoid partial circles
                mask[0, :] = mask[-1, :] = mask[:, :1] = mask[:, -1:] = 0

                final_circles = []
                downsample_factor = 2
                while True:
                    dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

                    # downsample distance map and upscale detected centers to original image size
                    dist_map_downscaled = cv2.resize(dist_map, None, fx=1/downsample_factor, fy=1/downsample_factor)
                    centers_downscaled = peak_local_max(dist_map_downscaled, min_distance=min_dist, exclude_border=False)
                    centers = centers_downscaled * downsample_factor

                    # centers = peak_local_max(dist_map, min_distance=min_dist, exclude_border=False)
                    radius = dist_map[centers[:, 0], centers[:, 1]]

                    eligible_centers = centers[radius > min_gap_radius, :]
                    eligible_radius = radius[radius > min_gap_radius]
                    eligible_circles = np.hstack([eligible_centers, eligible_radius[:, None]])

                    if len(eligible_circles) == 0:
                        break

                    result = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                    while len(eligible_circles) > 0:
                        if eligible_circles[1:, :].size > 0:
                            pw_euclidean_dist = \
                                euclidean_distances(eligible_circles[[0], :2], eligible_circles[1:, :2])[0]
                            pw_radius_sum = eligible_circles[0, 2] + eligible_circles[1:, 2]
                            neighbor_idx = np.nonzero(pw_euclidean_dist < pw_radius_sum)[0] + 1
                            eligible_circles = np.delete(eligible_circles, neighbor_idx, axis=0)

                        circle = eligible_circles[0, :]
                        result = cv2.circle(result, (int(circle[1]), int(circle[0])), int(circle[2]), (0, 0, 0), -1)
                        final_circles.append(eligible_circles[0, :])
                        eligible_circles = np.delete(eligible_circles, 0, axis=0)

                    mask = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

                final_result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                for circle in final_circles:
                    final_result = cv2.circle(final_result, (int(circle[1]), int(circle[0])),
                                              int(circle[2]), (0, 0, 255), 1)
                img_name = os.path.basename(img_path)
                cv2.imwrite(join_path(gap_result_dir, os.path.splitext(img_name)[0] + "_GapImage.png"), final_result)
                areas = np.pi * (np.array(final_circles)[:, 2] ** 2) * self.ims_res**2
                names.append(img_name)
                means.append(np.mean(areas))
                stds.append(np.std(areas))
                percentile5.append(np.percentile(areas, 5))
                median.append(np.median(areas))
                percentile95.append(np.percentile(areas, 95))
                counts.append(areas.size)
                final_circles = np.array(final_circles)
                data = {'Area (µm²)': areas,
                        'X': final_circles[:, 1],
                        'Y': final_circles[:, 0]
                        }
                df = pd.DataFrame(data)
                df.to_csv(join_path(gap_result_dir, "IndividualGaps_" + os.path.splitext(img_name)[0] + ".csv"), index=False)
            # summary_file.write("filename mean std percentile5 median percentile95 counts\n")
            if len(names) > 0:
                data = {'Image': names,
                        'Mean (All gaps area in µm²)': means,
                        'Std (All gaps area in µm²)': stds,
                        'Percentile5 (All gaps area in µm²)': percentile5,
                        'Median (All gaps area in µm²)': median,
                        'Percentile95 (All gaps area in µm²)': percentile95,
                        'Gap Circles Count (All)': counts
                        }
                df = pd.DataFrame(data)
                df.to_csv(join_path(gap_result_dir, "GapAnalysisSummary.csv"), index=False)

    def analyze_intra_gaps(self):
        gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
        img_paths = glob(join_path(self.bin_dir, '*.png'))
        img_paths.sort()

        names, means, stds, means_radius, stds_radius, counts = [], [], [], [], [], []
        Log.logger.info('Performing intra gap analysis for {} images'.format(len(img_paths)))
        for img_path in img_paths:
            base_name = os.path.basename(img_path)[:-9]
            binary_mask = cv2.imread(img_path, 0)
            img_fibre = cv2.imread(join_path(self.mask_dir, base_name + '_roi.png'), 0)
            color_img_fibre = cv2.cvtColor(img_fibre, cv2.COLOR_GRAY2BGR)
            csv_file_path = join_path(gap_result_dir, 'IndividualGaps_' + base_name + '_roi.csv')
            if not os.path.exists(csv_file_path):
                continue

            df_circles = pd.read_csv(csv_file_path)
            areas = []
            circle_cnt = 0
            for index, row in df_circles.iterrows():
                area, x, y = row['Area (µm²)'], int(row['X']), int(row['Y'])
                radius = int(np.sqrt(area / np.pi) / self.ims_res)   # convert back to measurements in pixels
                if binary_mask[y, x] > 0:
                    color_img_fibre = cv2.circle(color_img_fibre, (x, y), radius, (0, 255, 0), 1)
                    areas.append(area)
                    circle_cnt += 1

            areas = np.array(areas)
            radius = np.sqrt(areas / np.pi)
            cv2.imwrite(join_path(gap_result_dir,
                                  base_name + "_roi_GapImage_intra_gaps.png"), color_img_fibre)
            names.append(base_name + "_roi.png")
            if len(areas) > 0:
                means.append(np.mean(areas))
                stds.append(np.std(areas))
                means_radius.append(np.mean(radius))
                stds_radius.append(np.std(radius))
                counts.append(circle_cnt)
            else:
                means.append(0)
                stds.append(0)
                means_radius.append(0)
                stds_radius.append(0)
                counts.append(0)

        if names:
            data = {'Image': names,
                    'Mean (ROI gaps area in µm²)': means,
                    'Std (ROI gaps area in µm²)': stds,
                    'Mean (ROI gaps radius in µm)': means_radius,
                    'Std (ROI gaps radius in µm)': stds_radius,
                    'Gap Circles Count (ROI)': counts}
            df = pd.DataFrame(data)
            df.to_csv(join_path(gap_result_dir, 'IntraGapAnalysisSummary.csv'), index=False)
        else:
            Log.logger.warning('No gap analysis results. Skipping intra gap analysis.')

        # Moving images to 'Export' folder
        img_paths = glob(join_path(gap_result_dir, '*.png'))
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            img_name = img_name[:img_name.index("_GapImage")]
            shutil.copy(img_path, join_path(self.output_folder, "Exports", img_name))

    def combine_statistics(self):
        Log.logger.info('Combining statistics.')

        segmenter_csv = join_path(self.bin_dir, 'ResultsROI.csv')
        df_segmenter = pd.read_csv(segmenter_csv)

        # Check if image names are unique
        segmenter_img_names = df_segmenter['Image'].values

        if len(segmenter_img_names) != len(set(segmenter_img_names)):
            Log.logger.critical('Images names are not unique in ' + segmenter_csv)
            return

        area_roi = []
        percent_roi = []
        area_width = []
        percent_width = []
        mean_red_intensity_roi = []
        mean_red_intensity_width = []
        mean_red_intensity_hdm = []
        for i in range(len(self.df_stats)):
            img_name = self.df_stats.loc[i, 'Image']
            if img_name[:-4] + ".png" in df_segmenter['Image'].values:
                max_idx = (df_segmenter['Image'] == (img_name[:-4] + ".png")).idxmax()
                area_roi.append(df_segmenter.loc[max_idx, 'Area (ROI)'])
                percent_roi.append(df_segmenter.loc[max_idx, '% ROI Area'])
                area_width.append(df_segmenter.loc[max_idx, 'Area (WIDTH)'])
                percent_width.append(df_segmenter.loc[max_idx, '% WIDTH Area'])
                mean_red_intensity_roi.append(df_segmenter.loc[max_idx, 'Mean Fibre Intensity (ROI)'])
                mean_red_intensity_width.append(df_segmenter.loc[max_idx, 'Mean Fibre Intensity (WIDTH)'])
                mean_red_intensity_hdm.append(df_segmenter.loc[max_idx, 'Mean Fibre Intensity (HDM)'])
            else:
                Log.logger.info(img_name + ' cannot be found in ' + segmenter_csv)
                area_roi.append(0)
                percent_roi.append(0)
                area_width.append(0)
                percent_width.append(0)
                mean_red_intensity_roi.append(0)
                mean_red_intensity_width.append(0)
                mean_red_intensity_hdm.append(0)

        area_roi = [v*self.ims_res**2 for v in area_roi]
        area_width = [v*self.ims_res**2 for v in area_width]
        self.df_stats.insert(
            self.df_stats.columns.get_loc("Area of Fibre Spines (µm²)") + 1,
            "Fibre Area (ROI, µm²)", area_roi)
        self.df_stats.insert(
            self.df_stats.columns.get_loc("Fibre Area (ROI, µm²)") + 1,
            "Fibre Area (WIDTH, µm²)", area_width)
        self.df_stats.insert(
            self.df_stats.columns.get_loc("% HDM Area") + 1,
            "% ROI Area", percent_roi)
        self.df_stats.insert(
            self.df_stats.columns.get_loc("% ROI Area") + 1, "% WIDTH Area", percent_width)
        self.df_stats.insert(
            self.df_stats.columns.get_loc("% WIDTH Area") + 1,
            "Mean Fibre Intensity (HDM)", mean_red_intensity_hdm)
        self.df_stats.insert(self.df_stats.columns.get_loc("Mean Fibre Intensity (HDM)") + 1,
                             "Mean Fibre Intensity (ROI)", mean_red_intensity_roi)
        self.df_stats.insert(self.df_stats.columns.get_loc("Mean Fibre Intensity (ROI)") + 1,
                             "Mean Fibre Intensity (WIDTH)", mean_red_intensity_width)

        # Calculate other metrics
        def move_column_inplace(df, col, pos):
            col = df.pop(col)
            df.insert(pos, col.name, col)

        percent = self.df_stats.loc[:, '% HDM Area'].tolist()
        self.df_stats.insert(
            self.df_stats.columns.get_loc("Area of Fibre Spines (µm²)")+1,
            'Fibre Area (HDM, µm²)',
            self.df_stats['% HDM Area'] * self.df_stats['Total Image Area (µm²)'])
        total_area = self.df_stats.loc[:, 'Total Image Area (µm²)'].tolist()
        total_length = self.df_stats.loc[:, 'Total Length (µm)'].tolist()
        # num_endpoints = self.df_stats.loc[:, 'Endpoints'].tolist()
        # num_branchpoints = self.df_stats.loc[:, 'Branchpoints'].tolist()

        # avg_length = []
        # for l, e, b in zip(total_length, num_endpoints, num_branchpoints):
        #     avg_length.append((l * 2) / (e + b) if (e + b) != 0 else 0)

        avg_thickness_hdm = []
        for p, a, l in zip(percent, total_area, total_length):
            avg_thickness_hdm.append(a * p / l if l != 0 else 0)

        avg_thickness_roi = []
        for a, l in zip(area_roi, total_length):
            avg_thickness_roi.append(a / l if l != 0 else 0)

        avg_thickness_width = []
        for a, l in zip(area_width, total_length):
            avg_thickness_width.append(a / l if l != 0 else 0)

        # Insert more metrics
        # Avg Length is the same as HGU (hyphal growth unit)
        # self.df_stats.insert(
        #     self.df_stats.columns.get_loc("Total Length (µm)") + 1,
        #     "Avg Length (µm)", avg_length)
        self.df_stats.insert(
            self.df_stats.columns.get_loc("Area of Fibre Spines (µm²)") + 1,
            "Avg Thickness (HDM, µm)", avg_thickness_hdm)
        self.df_stats.insert(
            self.df_stats.columns.get_loc("Fibre Area (ROI, µm²)") + 1,
            "Avg Thickness (ROI, µm)", avg_thickness_roi)
        self.df_stats.insert(
            self.df_stats.columns.get_loc("Fibre Area (WIDTH, µm²)") + 1,
            "Avg Thickness (WIDTH, µm)", avg_thickness_width)

        # re-order HDM columns
        move_column_inplace(self.df_stats, 'Total Image Area (µm²)', 1)  # move to the first column
        move_column_inplace(self.df_stats, 'Fibre Area (HDM, µm²)', 3)
        move_column_inplace(self.df_stats, '% HDM Area',
                            self.df_stats.columns.get_loc("Fibre Area (HDM, µm²)") + 1)
        move_column_inplace(self.df_stats, "Avg Thickness (HDM, µm)",
                            self.df_stats.columns.get_loc("% HDM Area") + 1)
        move_column_inplace(self.df_stats, "Mean Fibre Intensity (HDM)",
                            self.df_stats.columns.get_loc("Avg Thickness (HDM, µm)") + 1)

        # re-order ROI columns
        move_column_inplace(self.df_stats, "Fibre Area (ROI, µm²)",
                            self.df_stats.columns.get_loc("Mean Fibre Intensity (HDM)") + 1)
        move_column_inplace(self.df_stats, '% ROI Area',
                            self.df_stats.columns.get_loc("Fibre Area (ROI, µm²)") + 1)
        move_column_inplace(self.df_stats, "Avg Thickness (ROI, µm)",
                            self.df_stats.columns.get_loc("% ROI Area") + 1)
        move_column_inplace(self.df_stats, "Mean Fibre Intensity (ROI)",
                            self.df_stats.columns.get_loc("Avg Thickness (ROI, µm)") + 1)

        # re-order WIDTH columns
        move_column_inplace(self.df_stats, "Fibre Area (WIDTH, µm²)",
                            self.df_stats.columns.get_loc("Mean Fibre Intensity (ROI)") + 1)
        move_column_inplace(self.df_stats, '% WIDTH Area',
                            self.df_stats.columns.get_loc("Fibre Area (WIDTH, µm²)") + 1)
        move_column_inplace(self.df_stats, "Avg Thickness (WIDTH, µm)",
                            self.df_stats.columns.get_loc("% WIDTH Area") + 1)
        move_column_inplace(self.df_stats, "Mean Fibre Intensity (WIDTH)",
                            self.df_stats.columns.get_loc("Avg Thickness (WIDTH, µm)") + 1)

        means_intra, stds_intra, means_radius_intra, stds_radius_intra, counts_intra = [], [], [], [], []
        intra_gaps_csv = join_path(self.output_folder, 'Masks', 'GapAnalysis', 'IntraGapAnalysisSummary.csv')

        means_total, stds_total, means_radius_total, stds_radius_total, counts_total = [], [], [], [], []
        gap_summary_file = os.path.join(self.output_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummary.csv')

        if os.path.exists(gap_summary_file) and os.path.exists(intra_gaps_csv):
            df_gaps_total = pd.read_csv(gap_summary_file)
            df_gaps_intra = pd.read_csv(intra_gaps_csv)
            gaps_img_names = df_gaps_intra['Image'].values
            if len(gaps_img_names) != len(set(gaps_img_names)):
                Log.logger.critical('Images names are not unique in ' + intra_gaps_csv)
                return

            for img_name in self.df_stats['Image']:
                img_name = img_name[:-4] + ".png"
                if img_name in gaps_img_names:
                    gaps_intra_row = df_gaps_intra[df_gaps_intra['Image'] == img_name].iloc[0]
                    means_intra.append(gaps_intra_row['Mean (ROI gaps area in µm²)'])
                    stds_intra.append(gaps_intra_row['Std (ROI gaps area in µm²)'])
                    means_radius_intra.append(gaps_intra_row['Mean (ROI gaps radius in µm)'])
                    stds_radius_intra.append(gaps_intra_row['Std (ROI gaps radius in µm)'])
                    counts_intra.append(gaps_intra_row['Gap Circles Count (ROI)'])

                    gaps_total_row = df_gaps_total[df_gaps_total['Image'] == img_name].iloc[0]
                    means_total.append(gaps_total_row['Mean (All gaps area in µm²)'])
                    stds_total.append(gaps_total_row['Std (All gaps area in µm²)'])
                    means_radius_total.append(np.sqrt(gaps_total_row['Mean (All gaps area in µm²)'] / np.pi))
                    stds_radius_total.append(np.sqrt(gaps_total_row['Std (All gaps area in µm²)'] / np.pi))
                    counts_total.append(gaps_total_row['Gap Circles Count (All)'])

                else:
                    Log.logger.warning(img_name + ' cannot be found in ' + segmenter_csv)
                    means_intra.append(0)
                    stds_intra.append(0)
                    means_radius_intra.append(0)
                    stds_radius_intra.append(0)
                    counts_intra.append(0)

                    means_total.append(0)
                    stds_total.append(0)
                    means_radius_total.append(0)
                    stds_radius_total.append(0)
                    counts_total.append(0)

            self.df_stats.insert(len(self.df_stats.columns), "Mean (All gaps area in µm²)", means_total)
            self.df_stats.insert(len(self.df_stats.columns), "Std (All gaps area in µm²)", stds_total)
            self.df_stats.insert(len(self.df_stats.columns), "Mean (All gaps radius in µm)", means_radius_total)
            self.df_stats.insert(len(self.df_stats.columns), "Std (All gaps radius in µm)", stds_radius_total)
            self.df_stats.insert(len(self.df_stats.columns), "Gap Circles Count (All)", counts_total)

            self.df_stats.insert(len(self.df_stats.columns), "Mean (ROI gaps area in µm²)", means_intra)
            self.df_stats.insert(len(self.df_stats.columns), "Std (ROI gaps area in µm²)", stds_intra)
            self.df_stats.insert(len(self.df_stats.columns), "Mean (ROI gaps radius in µm)", means_radius_intra)
            self.df_stats.insert(len(self.df_stats.columns), "Std (ROI gaps radius in µm)", stds_radius_intra)
            self.df_stats.insert(len(self.df_stats.columns), "Gap Circles Count (ROI)", counts_intra)

        else:
            Log.logger.warning("No gap analysis results. Skipping normalization for gap stats.")

        # Add extra info: image resolution
        image_res = [self.ims_res] * len(self.df_stats)
        self.df_stats.insert(len(self.df_stats.columns), "Image Res. (µm/pixel)", image_res)

        # Save to new csv file
        final_csv = join_path(self.output_folder, 'QuantificationResults.csv')
        self.df_stats.to_csv(final_csv, index=False, float_format="%.3f")
        Log.logger.info('Statistics have been saved to ' + final_csv)

    def normalize_statistics(self):
        results_csv = join_path(self.output_folder, 'QuantificationResults.csv')
        if os.path.exists(results_csv):
            df_results = pd.read_csv(results_csv)
        else:
            Log.logger.warning(results_csv + " does not exist. Please check if image quantification was run properly.")
            return

        # Normalize fibre area
        fibre_area_width = df_results['Fibre Area (WIDTH, µm²)'].values
        fibre_area_roi = df_results['Fibre Area (ROI, µm²)'].values
        fibre_area_ratio_woroi = array_divide(fibre_area_width, fibre_area_roi)
        df_results.insert(df_results.columns.get_loc('% WIDTH Area') + 1,
                          'Fibre Coverage (WIDTH/ROI)', fibre_area_ratio_woroi)

        # Normalize branch and end points
        total_lengths = df_results['Total Length (µm)'].values
        branchpoints = df_results['Branchpoints'].values
        endpoints = df_results['Endpoints'].values

        branchpts_density = array_divide(branchpoints, total_lengths)
        df_results.insert(df_results.columns.get_loc('Branchpoints') + 1,
                          'Branchpoints Density (µm⁻¹)', branchpts_density)

        endpoints_density = array_divide(endpoints, total_lengths)
        df_results.insert(df_results.columns.get_loc('Endpoints') + 1,
                          'Endpoints Density (µm⁻¹)', endpoints_density)

        # Normalize gap area
        if self.args["Quantification"]["Minimum Gap Diameter (pixels)"] > 0:
            mean_total_area = df_results['Mean (All gaps area in µm²)'].values
            total_image_area = df_results['Total Image Area (µm²)'].values
            mean_gap_area_total_norm = array_divide(mean_total_area, total_image_area)

            std_total_area = df_results['Std (All gaps area in µm²)'].values
            std_gap_area_total_norm = array_divide(std_total_area, total_image_area)

            mean_roi_area = df_results['Mean (ROI gaps area in µm²)'].values
            fibre_roi_area = df_results['Fibre Area (ROI, µm²)'].values
            mean_gap_area_roi_norm = array_divide(mean_roi_area, fibre_roi_area)

            std_roi_area = df_results['Std (ROI gaps area in µm²)'].values
            std_gap_area_roi_norm = array_divide(std_roi_area, fibre_roi_area)

            df_results.insert(df_results.columns.get_loc('Mean (All gaps area in µm²)') + 1,
                              'Normalized Mean (All gaps area)', mean_gap_area_total_norm)
            df_results.insert(df_results.columns.get_loc('Std (All gaps area in µm²)') + 1,
                              'Normalized Std (All gaps area)', std_gap_area_total_norm)
            df_results.insert(df_results.columns.get_loc('Mean (ROI gaps area in µm²)') + 1,
                              'Normalized Mean (ROI gaps area)', mean_gap_area_roi_norm)
            df_results.insert(df_results.columns.get_loc('Std (ROI gaps area in µm²)') + 1,
                              'Normalized Std (ROI gaps area)', std_gap_area_roi_norm)

            # Normalize gap radius
            mean_total_radius = df_results['Mean (All gaps radius in µm)'].values
            mean_gap_radius_total_norm = array_divide(mean_total_radius, np.sqrt(total_image_area))
            std_total_radius = df_results['Std (All gaps radius in µm)'].values
            std_gap_radius_total_norm = array_divide(std_total_radius, np.sqrt(total_image_area))

            mean_roi_radius = df_results['Mean (ROI gaps radius in µm)'].values
            mean_gap_radius_roi_norm = array_divide(mean_roi_radius, np.sqrt(fibre_roi_area))

            std_roi_radius = df_results['Std (ROI gaps radius in µm)'].values
            std_gap_radius_roi_norm = array_divide(std_roi_radius, np.sqrt(fibre_roi_area))

            df_results.insert(df_results.columns.get_loc('Mean (All gaps radius in µm)') + 1,
                              'Normalized Mean (All gaps radius)', mean_gap_radius_total_norm)
            df_results.insert(df_results.columns.get_loc('Std (All gaps radius in µm)') + 1,
                              'Normalized Std (All gaps radius)', std_gap_radius_total_norm)
            df_results.insert(df_results.columns.get_loc('Mean (ROI gaps radius in µm)') + 1,
                              'Normalized Mean (ROI gaps radius)', mean_gap_radius_roi_norm)
            df_results.insert(df_results.columns.get_loc('Std (ROI gaps radius in µm)') + 1,
                              'Normalized Std (ROI gaps radius)', std_gap_radius_roi_norm)

            # Normalize gap number
            circle_cnt_total = df_results['Gap Circles Count (All)'].values
            gap_num_total_norm = array_divide(circle_cnt_total, total_image_area)

            circle_cnt_roi = df_results['Gap Circles Count (ROI)'].values
            gap_num_roi_norm = array_divide(circle_cnt_roi, fibre_roi_area)

            df_results.insert(df_results.columns.get_loc('Gap Circles Count (All)') + 1,
                              'Gap density (All, µm⁻²)', gap_num_total_norm)
            df_results.insert(df_results.columns.get_loc('Gap Circles Count (ROI)') + 1,
                              'Gap density (ROI, µm⁻²)', gap_num_roi_norm)

        # Normalize lacunarity, see https://sci-hub.mksa.top/10.1016/j.jsg.2010.08.010
        ratio = df_results['Total Length (µm)'].values * df_results['Image Res. (µm/pixel)'].values / df_results[
            'Total Image Area (µm²)'].values
        normalized_lacunarity = array_divide(
            df_results['Lacunarity'].values - 1, array_divide(1.0-ratio, ratio))
        df_results.insert(df_results.columns.get_loc('Lacunarity') + 1,
                          'Normalized Lacunarity', normalized_lacunarity)

        # if no segmentation is performed, no need to re-calculate statistics
        if not self.args["Initialization"]["Segmentation"]:
            Log.logger.info(
                "No segmentation results available. Dropping columns with 'ROI'.")
            df_results.drop(list(df_results.filter(regex="ROI")), axis=1, inplace=True)

        # save to new csv file
        final_csv = join_path(self.output_folder, 'QuantificationResults.csv')
        df_results.to_csv(final_csv, index=False, float_format='%.3f')
        Log.logger.info('Statistics have been normalized and saved to ' + final_csv)

    def generate_color_maps(self):
        ori_img_paths = get_img_paths(join_path(self.output_folder, "Eligible"))
        Log.logger.info(f'Generating color maps for {len(ori_img_paths)} images.')
        for ori_img_path in tqdm(ori_img_paths, bar_format=read_bar_format):
            ori_img_name = os.path.basename(ori_img_path)
            name_wo_ext = ori_img_name[:ori_img_name.rindex('.')] + "_roi"

            # Create a sub-folder for better readability
            Path(join_path(self.color_dir, name_wo_ext)).mkdir(parents=True, exist_ok=True)

            rgb_img = iio.imread(ori_img_path)
            roi_img = iio.imread(join_path(self.bin_dir, name_wo_ext[:-4] + "_mask.png"))

            mask_img = 255 - iio.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Mask.png"))
            mask_img = np.repeat(mask_img[:, :, np.newaxis], 3, axis=2)
            mask_glow = mask_color_map(rgb_img, mask_img)

            Image.fromarray(mask_glow).save(join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_mask.png"))

            skeleton = iio.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Skeleton.png"))
            red_pos = np.where((skeleton[..., 0] == 255) & (skeleton[..., 1] == 0) & (skeleton[..., 2] == 0))
            skeleton[red_pos[0], red_pos[1], :] = [0, 255, 0]
            index_pos = np.where(cv2.cvtColor(skeleton, cv2.COLOR_BGR2GRAY) == 0)
            skeleton[index_pos[0], index_pos[1], :] = rgb_img[index_pos[0], index_pos[1], :]
            Image.fromarray(skeleton).save(join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_skeleton.png"))

            energy_map = iio.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Energy.tif"))
            cohere_map = iio.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Coherency.tif"))
            orient_map = iio.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Orientation.tif"))
            length_map = iio.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Length_Map.tif"))

            bg_index_pos = np.where(roi_img < 128)
            energy_map[bg_index_pos[0], bg_index_pos[1]] = 0
            cohere_map[bg_index_pos[0], bg_index_pos[1]] = 0
            orient_map[bg_index_pos[0], bg_index_pos[1]] = 0
            sbs_color_map(
                rgb_img, orient_map,
                join_path(self.color_dir, name_wo_ext,
                          name_wo_ext + "_color_orientation.png"), cbar_label="Orientation")
            sbs_color_map(
                rgb_img, cohere_map,
                join_path(self.color_dir, name_wo_ext,
                          name_wo_ext + "_color_coherency.png"), cbar_label="Coherency")

            color_length = info_color_map(rgb_img, length_map, cbar_label="Length (µm)", cmap="plasma", radius=1)
            Image.fromarray(color_length).save(join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_length.png"))

            curve_paths = glob(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Curve_Map_*"))
            for curve_path in curve_paths:
                curve_name_wo_ext = os.path.basename(curve_path)[:-4]
                suffix = curve_name_wo_ext[len(name_wo_ext + "_Curve_Map"):]
                curve_map = tiff.imread(curve_path)
                color_curve = info_color_map(rgb_img, curve_map/180, cbar_label="Curliness", cmap="plasma", radius=1)
                Image.fromarray(color_curve).save(
                    join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_curve" + suffix + ".png"))

            width_img = 255 - iio.imread(
                join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Width.png"))
            width_img = np.repeat(width_img[:, :, np.newaxis], 3, axis=2)
            width_map = width_color_map(rgb_img, width_img, mask_img)
            Image.fromarray(width_map).save(join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_width.png"))

            color_survey = iio.imread(
                join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Color_Survey.tif"))
            color_survey[bg_index_pos[0], bg_index_pos[1], :] = [228, 228, 228]
            sbs_color_survey(rgb_img, color_survey, join_path(self.color_dir,
                                                              name_wo_ext, name_wo_ext + "_orient_color_survey.png"))

            if os.path.exists(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_GapImage.png")):
                gap_img = iio.imread(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_GapImage.png"))
                sbs_color_survey(rgb_img, gap_img,
                                 join_path(self.color_dir, name_wo_ext, name_wo_ext + "_all_gaps.png"))

            if os.path.exists(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_GapImage_intra_gaps.png")):
                intra_gap_img = iio.imread(join_path(
                    self.export_dir, name_wo_ext, name_wo_ext + "_GapImage_intra_gaps.png"))
                sbs_color_survey(rgb_img, intra_gap_img,
                                 join_path(self.color_dir, name_wo_ext, name_wo_ext + "_intra_gaps.png"))

    def run(self):
        self.initialize_params()
        self.remove_large_images()
        self.generate_rois()

        if self.args["Initialization"]["Quantification"]:
            self.quantify_images()
            self.visualize_fibres()
            self.calc_fibre_areas()
            self.analyze_all_gaps()
            self.analyze_intra_gaps()
            self.combine_statistics()
            self.normalize_statistics()
            self.generate_color_maps()
        else:
            Log.logger.info('Segmentation is done. No further analysis will be conducted.')
