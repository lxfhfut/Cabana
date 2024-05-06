import os
from PIL import Image
import csv
import cv2
import shutil
import numpy as np
import pandas as pd
from glob import glob
from log import Log
import tifffile as tiff
import yaml
import imageio.v3 as iio
from detector import MSRidgeDetector
from analyzer import SkeletonAnalyzer
from hdm import quantify_black_space
from orientation import OrientationAnalyzer
import xml.etree.ElementTree as ET

from pathlib import Path
from utils import split2batches, mask_color_map, orient_vf, info_color_map, sanitize_filename
from utils import create_folder, join_path, sbs_color_map, sbs_color_survey, width_color_map, get_img_paths
from skimage.feature import peak_local_max
from sklearn.metrics.pairwise import euclidean_distances
from segmenter import parse_args, segment_single_image, visualize_ridges
from skimage.color import rgb2hed, hed2rgb, rgb2gray


class Cabana:
    def __init__(self, program_folder, input_folder, out_folder,
                 batch_size=5, batch_idx=0, ignore_oversized=True):
        self.param_file = "Parameters.yml"

        self.args = None  # args for Cabana program
        self.seg_args = parse_args()  # args for segmentation
        self.ims_res = 1.0  # µm/pixel
        self.df_statistics = pd.DataFrame()

        # self.ij = ij
        self.program_folder = program_folder
        self.input_folder = input_folder
        self.output_folder = out_folder
        self.batch_idx = batch_idx
        self.batch_size = batch_size
        self.ignore_oversized = ignore_oversized

        # Create sub-folders in output directory
        self.roi_dir = join_path(self.output_folder, 'ROIs', "")
        self.bin_dir = join_path(self.output_folder, 'Bins', "")
        self.mask_dir = join_path(self.output_folder, 'Masks', "")
        self.hdm_dir = join_path(self.output_folder, 'HDM', "")
        self.export_dir = join_path(self.output_folder, 'Exports', "")
        self.color_dir = join_path(self.output_folder, 'Colors', "")
        self.ridge_dir = join_path(self.output_folder, 'Ridges', "")
        self.eligible_dir = join_path(self.output_folder, 'Eligible')

        create_folder(self.eligible_dir)
        create_folder(self.ridge_dir)
        create_folder(self.mask_dir)
        create_folder(self.hdm_dir)
        create_folder(self.export_dir)
        create_folder(self.color_dir)
        create_folder(self.roi_dir)
        create_folder(self.bin_dir)
        setattr(self.seg_args, 'roi_dir', self.roi_dir)
        setattr(self.seg_args, 'bin_dir', self.bin_dir)

    def initialize_params(self):
        Log.init_log_path(join_path(self.program_folder, 'logs'))
        Log.logger.info('Logs folder: {}'.format(join_path(self.program_folder, 'logs')))
        param_path = join_path(self.program_folder, self.param_file)
        with open(param_path) as pf:
            try:
                self.args = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                print(exc)
        # print(self.args['Initialization']['Max Size'])
        # overwrite some fields of seg_args
        setattr(self.seg_args, 'num_channels', int(self.args['Segmentation']["Number of Labels"]))
        setattr(self.seg_args, 'max_iter', int(self.args['Segmentation']["Max Iterations"]))
        setattr(self.seg_args, 'hue_value', float(self.args['Segmentation']["Normalized Hue Value"]))
        setattr(self.seg_args, 'rt', float(self.args['Segmentation']["Color Threshold"]))
        setattr(self.seg_args, 'max_size', int(self.args['Segmentation']["Max Size"]))
        setattr(self.seg_args, 'min_size', int(self.args['Segmentation']["Min Size"]))
        Log.log_parameters(param_path)

    def remove_oversized_imgs(self):
        img_paths = get_img_paths(self.input_folder)
        path_batches, res_batches = split2batches(img_paths, self.batch_size)

        max_line_width = self.args['Detection']["Max Line Width"]
        max_size = self.args["Segmentation"]["Max Size"]
        with open(join_path(self.eligible_dir, "Ignored_images.txt"), 'w') as f:
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
                    if self.ignore_oversized:
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
            msg = "ignored" if self.ignore_oversized else "split into smaller blocks"
            Log.logger.warning('{} over-sized images have been {}.'.format(count_oversized, msg))

        self.input_folder = self.eligible_dir
        self.ims_res = res_batches[self.batch_idx]

        Log.logger.info("Image resolution of Batch {} has been updated to {} microns/pixel".format(
            self.batch_idx+1, str(res_batches[self.batch_idx])))

    def generate_rois(self):
        img_paths = get_img_paths(self.input_folder)
        img_paths.sort()
        Log.logger.info('Segmenting {} images in {}'.format(len(img_paths), self.input_folder))
        for img_path in img_paths:
            setattr(self.seg_args, 'input', img_path)
            if self.args["Initialization"]["Segmentation"]:
                segment_single_image(self.seg_args)
            else:
                Log.logger.info("No segmentation is applied prior to image analysis.")
                img = cv2.imread(self.seg_args.input)
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                img_name = os.path.splitext(os.path.basename(self.seg_args.input))[0]
                cv2.imwrite(join_path(self.seg_args.roi_dir, img_name + '_roi.png'), img)
                cv2.imwrite(join_path(self.seg_args.bin_dir, img_name + '_mask.png'), mask)

        Log.logger.info('ROIs have been saved in {}'.format(self.seg_args.roi_dir))
        Log.logger.info('Masks have been saved in {}'.format(self.seg_args.bin_dir))

    def detect_ridges(self):
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
        det = MSRidgeDetector(line_widths=line_widths,
                              low_contrast=low_contrast,
                              high_contrast=high_contrast,
                              dark_line=dark_line,
                              extend_line=extend_line,
                              correct_pos=correct_pos,
                              min_len=min_len)
        for img_path in glob(join_path(self.roi_dir, '*.png')):
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
        img_names = []
        for img_path in glob(join_path(self.roi_dir, '*.png')):
            ori_img_name = os.path.basename(img_path)
            img_names.append(ori_img_name)
            name_wo_ext = ori_img_name[:ori_img_name.rindex('.')]
            mask = iio.imread(join_path(self.bin_dir, name_wo_ext[:-4]+"_mask.png"))
            orient_analyzer.compute_orient(img_path)
            alignments.append(orient_analyzer.mean_coherency(mask))

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
                self.color_dir, name_wo_ext, name_wo_ext + "angular_hist.png"),
                orient_analyzer.draw_angular_hist(mask=mask))

        data = {'Image': img_names,
                'Alignment': alignments}
        df_orient = pd.DataFrame(data)
        self.df_statistics = self.df_statistics.merge(df_orient, on="Image")

    def quantify_skeletons(self):
        min_skel_size = self.args["Quantification"]["Minimum Skeleton Size (um)"] / self.ims_res
        min_branch_len = self.args["Quantification"]["Minimum Branch Length (um)"] / self.ims_res
        min_hole_area = self.args["Quantification"]["Minimum Hole Area (um^2)"] / self.ims_res ** 2
        min_curve_win = self.args["Quantification"]["Minimum Curvature Window (um)"]
        max_curve_win = self.args["Quantification"]["Maximum Curvature Window (um)"]
        curve_win_step = self.args["Quantification"]["Curvature Window Step (um)"]

        # ridge detection returns ridges in black color, so dark_line is set to "True" for skeleton analysis
        skel_analyzer = SkeletonAnalyzer(skel_thresh=min_skel_size,
                                         branch_thresh=min_branch_len,
                                         hole_threshold=min_hole_area,
                                         dark_line=True)
        img_names, end_points, branch_points, growth_units = [], [], [], []
        proj_areas, lacunarities, total_lengths, frac_dims, total_areas = [], [], [], [], []
        curvatures = {}
        for win_sz in np.arange(min_curve_win, max_curve_win + curve_win_step, curve_win_step):
            curvatures[f"Curvature_{win_sz:.0f}"] = []

        for img_path in glob(join_path(self.mask_dir, '*.png')):
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
                curvatures[f"Curvature_{win_sz:.0f}"].append(skel_analyzer.avg_curve_all)
                iio.imwrite(join_path(
                    self.export_dir, name_wo_ext, f"{name_wo_ext}_Curve_Map_{win_sz:.0f}.tif"),
                    skel_analyzer.curve_map_all)

        data = {'Image': img_names,
                'Area (microns^2)': proj_areas,
                'Lacunarity': lacunarities,
                'Total Length (microns)': total_lengths,
                'Endpoints': end_points,
                'HGU (microns)': growth_units,
                'Branchpoints': branch_points,
                'Box-Counting Fractal Dimension': frac_dims,
                'TotalImageArea': total_areas
                }
        data.update(curvatures)
        df_skel = pd.DataFrame(data)
        self.df_statistics = self.df_statistics.merge(df_skel, on="Image")

    def quantify_images(self):

        if len(glob(join_path(self.roi_dir, '*.png'))) > 0:
            # HDM
            df_hdm = quantify_black_space(self.eligible_dir, self.hdm_dir, ext=".png",
                                          max_hdm=self.args["Quantification"]["Maximum Display HDM"],
                                          dark_line=self.args["Detection"]["Dark Line"])
            self.df_statistics = df_hdm

            # Ridge detection
            self.detect_ridges()

            # Skeleton analysis
            self.quantify_skeletons()

            # Orientation analysis
            self.analyze_orientations()

            # # Save statistics
            # self.df_statistics.to_csv(join_path(
            #     self.output_folder, "Twombli_Results.csv"), index=False, float_format="%.3f")

            Log.logger.info('Image analyses finished.')
            Log.logger.info('Appending FIJI log...')
            # return self.append_FIJI_log()
        else:
            return {}

    def append_FIJI_log(self):
        with open(join_path(self.output_folder, 'FIJI_log.txt')) as f:
            lines = f.readlines()
            cnt = 0
            for line in lines:
                if line.startswith("Found object at"):
                    cnt = 10
                elif cnt >= 1:
                    cnt -= 1
                    continue
                elif not line.rstrip():
                    continue
                else:
                    Log.logger.info(line.rstrip())

    def generate_visualizations(self, thickness=3):
        Log.logger.info('Generating ridges visualization results...')
        img_paths = get_img_paths(self.input_folder)
        img_paths.sort()
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            mask_path = join_path(self.mask_dir, os.path.splitext(img_name)[0] + '_roi.png')
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            visualize_ridges(img, mask, join_path(self.ridge_dir, os.path.splitext(img_name)[0] + '_ridges.png'), thickness)

    def calc_fibre_areas(self):
        img_paths = glob(join_path(self.bin_dir, '*.png'))
        img_paths.sort()
        with open(join_path(self.bin_dir, 'Results_ROI.csv'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(
                ['Image', 'Area (ROI)', '% ROI Area', 'Area (WIDTH)', '% WIDTH Area',
                 'Mean Fibre Intensity (ROI)',
                 'Mean Fibre Intensity (WIDTH)',
                 'Mean Fibre Intensity (HDM)'])
            Log.logger.info('Calculating fibre areas for {} images'.format(len(img_paths)))
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

            Log.logger.info('Areas have been saved in {}'.format(join_path(self.bin_dir, 'Results_ROI.csv')))

    def gap_analysis(self):
        min_gap_diameter = -1
        if "Minimum Gap Diameter (um)" in self.args["Quantification"].keys():
            min_gap_diameter = self.args["Quantification"]["Minimum Gap Diameter (um)"]
        if min_gap_diameter == -1:
            Log.logger.warning("'Minimum Gap Diameter can not be found in {}.".format(self.param_file))
        elif min_gap_diameter == 0:
            Log.logger.warning("Skipping gap analysis.")
            return
        else:
            Log.logger.info(
                f"Performing gap analysis with "
                f"Minimum Gap Diameter={min_gap_diameter:.1f} microns "
                f"({round(min_gap_diameter/self.ims_res)} pixels).")
            min_gap_diameter = round(min_gap_diameter/self.ims_res)

            gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
            Path(gap_result_dir).mkdir(parents=True, exist_ok=True)
            gap_analysis_file = join_path(gap_result_dir, "GapAnalysisSummary.txt")
            with open(gap_analysis_file, 'w+') as summary_file:
                summary_file.write("filename mean std percentile5 median percentile95 counts\n")
                img_paths = glob(join_path(self.mask_dir, '*.png'))
                Log.logger.info(
                    "Performing gap analysis on {} images with a minimum gap diameter of {}.".format(len(img_paths),
                                                                                                     min_gap_diameter))
                for img_path in img_paths:
                    min_gap_radius = min_gap_diameter / 2
                    min_dist = int(np.max([1, min_gap_radius // 2]))
                    img = cv2.imread(img_path, 0)
                    mask = img.copy()

                    # set border pixels to zero to avoid partial circles
                    mask[0, :] = mask[-1, :] = mask[:, :1] = mask[:, -1:] = 0

                    final_circles = []
                    while True:
                        dist_map = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
                        centers = peak_local_max(dist_map, min_distance=min_dist, exclude_border=False)
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
                    summary_file.write("{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.0f}\n".format(img_name,
                                                                                        np.mean(areas),
                                                                                        np.std(areas),
                                                                                        np.percentile(areas, 5),
                                                                                        np.median(areas),
                                                                                        np.percentile(areas, 95),
                                                                                        areas.size))
                    final_circles = np.array(final_circles)
                    data = {'Area (microns^2)': areas,
                            'X': final_circles[:, 1],
                            'Y': final_circles[:, 0]
                            }
                    df = pd.DataFrame(data)
                    df.to_csv(join_path(gap_result_dir, "IndividualGaps_" + os.path.splitext(img_name)[0] + ".csv"), index=False)

    def correct_gap_analysis(self):
        gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
        img_paths = glob(join_path(self.bin_dir, '*.png'))
        img_paths.sort()

        names, means, stds, means_radius, stds_radius, counts = [], [], [], [], [], []
        Log.logger.info('Correcting gap analysis results for {} images'.format(len(img_paths)))
        for img_path in img_paths:
            base_name = os.path.basename(img_path)[:-9]
            binary_mask = cv2.imread(img_path, 0)
            img_ridge = cv2.imread(join_path(self.mask_dir, base_name + '_roi.png'), 0)
            color_img_ridge = cv2.cvtColor(img_ridge, cv2.COLOR_GRAY2BGR)
            csv_file_path = join_path(gap_result_dir, 'IndividualGaps_' + base_name + '_roi.csv')
            if not os.path.exists(csv_file_path):
                continue

            df_circles = pd.read_csv(csv_file_path)
            areas = []
            circle_cnt = 0
            for index, row in df_circles.iterrows():
                area, x, y = row['Area (microns^2)'], int(row['X']), int(row['Y'])
                radius = int(np.sqrt(area / 3.1416) / self.ims_res)   # convert back to measurements in pixels
                if binary_mask[y, x] > 0:
                    color_img_ridge = cv2.circle(color_img_ridge, (x, y), radius, (0, 255, 0), 1)
                    areas.append(area)
                    circle_cnt += 1

            areas = np.array(areas)
            radius = np.sqrt(areas / 3.1416)
            cv2.imwrite(join_path(gap_result_dir,
                                  base_name + "_roi_GapImage_intra_gaps.png"), color_img_ridge)
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
                    'Mean (ROI gap area in microns^2)': means,
                    'Std (ROI gap area in microns^2)': stds,
                    'Mean (ROI gap radius in microns)': means_radius,
                    'Std (ROI gap radius in microns)': stds_radius,
                    'Gaps number (ROI)': counts}
            df = pd.DataFrame(data)
            df.to_csv(join_path(gap_result_dir, 'GapAnalysisSummaryCorrected.csv'), index=False)
        else:
            Log.logger.warning('There seems no gap analysis to correct.')

    def combine_statistics(self):

        Log.logger.info('Generating statistics...')

        # twombli_csv = join_path(self.output_folder, 'Twombli_Results.csv')
        # if os.path.exists(twombli_csv):
        #     df_twombli = pd.read_csv(twombli_csv)
        # else:
        #     Log.logger.warning(twombli_csv + " does not exist. Please check if TWOMBLI was run properly.")
        #     return

        segmenter_csv = join_path(self.bin_dir, 'Results_ROI.csv')
        df_segmenter = pd.read_csv(segmenter_csv)

        # Check if image names are unique
        # stats_img_names = self.df_statistics['Image'].values
        segmenter_img_names = df_segmenter['Image'].values

        # if len(stats_img_names) != len(set(stats_img_names)):
        #     Log.logger.critical('Images names are not unique in ' + twombli_csv)
        #     return

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
        for i in range(len(self.df_statistics)):
            img_name = self.df_statistics.loc[i, 'Image']
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
        self.df_statistics = (
            self.df_statistics.rename(columns={'Area (microns^2)': 'Projected Area of Ridge Spines (microns^2)'}))
        self.df_statistics = self.df_statistics.rename(columns={'% High Density Matrix': '% HDM Area'})
        self.df_statistics.insert(
            self.df_statistics.columns.get_loc("Projected Area of Ridge Spines (microns^2)") + 1,
            "Fibre Area (ROI, microns^2)", area_roi)
        self.df_statistics.insert(
            self.df_statistics.columns.get_loc("Fibre Area (ROI, microns^2)") + 1,
            "Fibre Area (WIDTH, microns^2)", area_width)
        self.df_statistics.insert(
            self.df_statistics.columns.get_loc("% HDM Area") + 1,
            "% ROI Area", percent_roi)
        self.df_statistics.insert(
            self.df_statistics.columns.get_loc("% ROI Area") + 1, "% WIDTH Area", percent_width)
        self.df_statistics.insert(
            self.df_statistics.columns.get_loc("% WIDTH Area") + 1,
            "Mean Fibre Intensity (HDM)", mean_red_intensity_hdm)
        self.df_statistics.insert(self.df_statistics.columns.get_loc("Mean Fibre Intensity (HDM)") + 1,
                                  "Mean Fibre Intensity (ROI)", mean_red_intensity_roi)
        self.df_statistics.insert(self.df_statistics.columns.get_loc("Mean Fibre Intensity (ROI)") + 1,
                                  "Mean Fibre Intensity (WIDTH)", mean_red_intensity_width)

        # Calculate other metrics
        def move_column_inplace(df, col, pos):
            col = df.pop(col)
            df.insert(pos, col.name, col)

        percent = self.df_statistics.loc[:, '% HDM Area'].tolist()
        self.df_statistics = self.df_statistics.rename(columns={'TotalImageArea': 'Total Image Area (microns^2)'})
        self.df_statistics['Total Image Area (microns^2)'] = self.df_statistics['Total Image Area (microns^2)'].mul(self.ims_res**2)

        self.df_statistics.insert(self.df_statistics.columns.get_loc("Projected Area of Ridge Spines (microns^2)")+1, 'Fibre Area (HDM, microns^2)',
                          self.df_statistics['% HDM Area'] * self.df_statistics['Total Image Area (microns^2)'])
        total_area = self.df_statistics.loc[:, 'Total Image Area (microns^2)'].tolist()
        total_length = self.df_statistics.loc[:, 'Total Length (microns)'].tolist()
        num_endpoints = self.df_statistics.loc[:, 'Endpoints'].tolist()
        num_branchpoints = self.df_statistics.loc[:, 'Branchpoints'].tolist()

        avg_length = []
        for l, e, b in zip(total_length, num_endpoints, num_branchpoints):
            avg_length.append((l * 2) / (e + b) if (e + b) != 0 else 0)

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
        self.df_statistics.insert(self.df_statistics.columns.get_loc("Total Length (microns)") + 1,
                          "Avg Length (microns)", avg_length)
        self.df_statistics.insert(self.df_statistics.columns.get_loc("Projected Area of Ridge Spines (microns^2)") + 1,
                          "Avg Thickness (HDM, microns)", avg_thickness_hdm)
        self.df_statistics.insert(self.df_statistics.columns.get_loc("Fibre Area (ROI, microns^2)") + 1,
                          "Avg Thickness (ROI, microns)", avg_thickness_roi)
        self.df_statistics.insert(self.df_statistics.columns.get_loc("Fibre Area (WIDTH, microns^2)") + 1,
                          "Avg Thickness (WIDTH, microns)", avg_thickness_width)

        # re-order HDM columns
        move_column_inplace(self.df_statistics, 'Total Image Area (microns^2)', 1)  # move to the first column
        move_column_inplace(self.df_statistics, 'Fibre Area (HDM, microns^2)', 3)
        move_column_inplace(self.df_statistics, '% HDM Area',
                            self.df_statistics.columns.get_loc("Fibre Area (HDM, microns^2)") + 1)
        move_column_inplace(self.df_statistics, "Avg Thickness (HDM, microns)",
                            self.df_statistics.columns.get_loc("% HDM Area") + 1)
        move_column_inplace(self.df_statistics, "Mean Fibre Intensity (HDM)",
                            self.df_statistics.columns.get_loc("Avg Thickness (HDM, microns)") + 1)

        # re-order ROI columns
        move_column_inplace(self.df_statistics, "Fibre Area (ROI, microns^2)",
                            self.df_statistics.columns.get_loc("Mean Fibre Intensity (HDM)") + 1)
        move_column_inplace(self.df_statistics, '% ROI Area',
                            self.df_statistics.columns.get_loc("Fibre Area (ROI, microns^2)") + 1)
        move_column_inplace(self.df_statistics, "Avg Thickness (ROI, microns)",
                            self.df_statistics.columns.get_loc("% ROI Area") + 1)
        move_column_inplace(self.df_statistics, "Mean Fibre Intensity (ROI)",
                            self.df_statistics.columns.get_loc("Avg Thickness (ROI, microns)") + 1)

        # re-order WIDTH columns
        move_column_inplace(self.df_statistics, "Fibre Area (WIDTH, microns^2)",
                            self.df_statistics.columns.get_loc("Mean Fibre Intensity (ROI)") + 1)
        move_column_inplace(self.df_statistics, '% WIDTH Area',
                            self.df_statistics.columns.get_loc("Fibre Area (WIDTH, microns^2)") + 1)
        move_column_inplace(self.df_statistics, "Avg Thickness (WIDTH, microns)",
                            self.df_statistics.columns.get_loc("% WIDTH Area") + 1)
        move_column_inplace(self.df_statistics, "Mean Fibre Intensity (WIDTH)",
                            self.df_statistics.columns.get_loc("Avg Thickness (WIDTH, microns)") + 1)

        means, stds, means_radius, stds_radius, counts = [], [], [], [], []
        gaps_csv = join_path(self.output_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummaryCorrected.csv')

        means_total, stds_total, means_radius_total, stds_radius_total, counts_total = [], [], [], [], []
        gap_summary_file = os.path.join(self.output_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummary.txt')
        with open(gap_summary_file, 'r') as file:
            # read the first line and split it into column names
            column_names = file.readline().strip().split(' ')
            num_columns = len(column_names)

            # read the remaining lines and create a list of data
            data = [[' '.join(line.strip().split(' ')[:-num_columns + 1]),
                     *[float(ff) for ff in line.strip().split(' ')[-num_columns + 1:]]] for line in file]

        # create a DataFrame using the column names and data
        df_gaps_total = pd.DataFrame(data, columns=column_names)

        if os.path.exists(gaps_csv):
            df_gaps = pd.read_csv(gaps_csv)
            gaps_img_names = df_gaps['Image'].values
            if len(gaps_img_names) != len(set(gaps_img_names)):
                Log.logger.critical('Images names are not unique in ' + gaps_csv)
                return

            for img_name in self.df_statistics['Image']:
                img_name = img_name[:-4] + ".png"
                if img_name in gaps_img_names:
                    gaps_row = df_gaps[df_gaps['Image'] == img_name].iloc[0]
                    means.append(gaps_row['Mean (ROI gap area in microns^2)'])
                    stds.append(gaps_row['Std (ROI gap area in microns^2)'])
                    means_radius.append(gaps_row['Mean (ROI gap radius in microns)'])
                    stds_radius.append(gaps_row['Std (ROI gap radius in microns)'])
                    counts.append(gaps_row['Gaps number (ROI)'])

                    gaps_total_row = df_gaps_total[df_gaps_total['filename'] == img_name].iloc[0]
                    means_total.append(gaps_total_row['mean'])
                    stds_total.append(gaps_total_row['std'])
                    means_radius_total.append(np.sqrt(gaps_total_row['mean'] / 3.1416))
                    stds_radius_total.append(np.sqrt(gaps_total_row['std'] / 3.1416))
                    counts_total.append(gaps_total_row['counts'])

                else:
                    Log.logger.warning(img_name + ' cannot be found in ' + segmenter_csv)
                    means.append(0)
                    stds.append(0)
                    means_radius.append(0)
                    stds_radius.append(0)
                    counts.append(0)

                    means_total.append(0)
                    stds_total.append(0)
                    means_radius_total.append(0)
                    stds_radius_total.append(0)
                    counts_total.append(0)

            self.df_statistics.insert(len(self.df_statistics.columns), "Mean (Total gap area in microns^2)", means_total)
            self.df_statistics.insert(len(self.df_statistics.columns), "Std (Total gap area in microns^2)", stds_total)
            self.df_statistics.insert(len(self.df_statistics.columns), "Mean (Total gap radius in microns)", means_radius_total)
            self.df_statistics.insert(len(self.df_statistics.columns), "Std (Total gap radius in microns)", stds_radius_total)
            self.df_statistics.insert(len(self.df_statistics.columns), "Gaps number (Total)", counts_total)

            self.df_statistics.insert(len(self.df_statistics.columns), "Mean (ROI gap area in microns^2)", means)
            self.df_statistics.insert(len(self.df_statistics.columns), "Std (ROI gap area in microns^2)", stds)
            self.df_statistics.insert(len(self.df_statistics.columns), "Mean (ROI gap radius in microns)", means_radius)
            self.df_statistics.insert(len(self.df_statistics.columns), "Std (ROI gap radius in microns)", stds_radius)
            self.df_statistics.insert(len(self.df_statistics.columns), "Gaps number (ROI)", counts)

        else:
            Log.logger.warning("There seems no gap analysis result available.")

        # Add extra info: image resolution
        image_res = [self.ims_res] * len(self.df_statistics)
        self.df_statistics.insert(len(self.df_statistics.columns), "Image Res. (microns/pp)", image_res)

        # Save to new csv file
        final_csv = join_path(self.output_folder, 'Quantification_Results.csv')
        self.df_statistics.to_csv(final_csv, index=False, float_format="%.3f")
        Log.logger.info('Statistics have been saved to ' + final_csv)

    def normalize_statistics(self):
        results_csv = join_path(self.output_folder, 'Quantification_Results.csv')
        if os.path.exists(results_csv):
            df_results = pd.read_csv(results_csv)
        else:
            Log.logger.warning(results_csv + " does not exist. Please check if TWOMBLI was run properly.")
            return

        # Normalize fibre area
        # fibre_area_ratio_wot = df_results['Fibre Area (WIDTH, microns^2)'].values / \
        #                        df_results['Total Image Area (microns^2)'].values
        fibre_area_ratio_woroi = df_results['Fibre Area (WIDTH, microns^2)'].values / \
                                 df_results['Fibre Area (ROI, microns^2)'].values
        # df_results.insert(df_results.columns.get_loc('Fibre Area (WIDTH, microns^2)') + 1,
        #                   'Fibre Coverage (WIDTH/Total)', fibre_area_ratio_wot)
        df_results.insert(df_results.columns.get_loc('% WIDTH Area') + 1,
                          'Fibre Coverage (WIDTH/ROI)', fibre_area_ratio_woroi)

        # Normalize branch and end points
        normalised_branchpts = df_results['Branchpoints'].values / df_results['Total Length (microns)'].values
        df_results.insert(df_results.columns.get_loc('Branchpoints') + 1, 'Normalised Branchpoints',
                          normalised_branchpts)
        normalised_endpts = df_results['Endpoints'].values / df_results['Total Length (microns)'].values
        df_results.insert(df_results.columns.get_loc('Endpoints') + 1, 'Normalised Endpoints', normalised_endpts)

        # Normalize gap area
        mean_gap_area_total_norm = df_results['Mean (Total gap area in microns^2)'].values / df_results[
            'Total Image Area (microns^2)'].values
        std_gap_area_total_norm = df_results['Std (Total gap area in microns^2)'].values / df_results[
            'Total Image Area (microns^2)'].values
        mean_gap_area_roi_norm = df_results['Mean (ROI gap area in microns^2)'].values / df_results[
            'Fibre Area (ROI, microns^2)'].values
        std_gap_area_roi_norm = df_results['Std (ROI gap area in microns^2)'].values / df_results[
            'Fibre Area (ROI, microns^2)'].values
        df_results.insert(df_results.columns.get_loc('Mean (Total gap area in microns^2)') + 1,
                          'Normalised Mean (Total gap area)', mean_gap_area_total_norm)
        df_results.insert(df_results.columns.get_loc('Std (Total gap area in microns^2)') + 1,
                          'Normalised Std (Total gap area)', std_gap_area_total_norm)
        df_results.insert(df_results.columns.get_loc('Mean (ROI gap area in microns^2)') + 1,
                          'Normalised Mean (ROI gap area)', mean_gap_area_roi_norm)
        df_results.insert(df_results.columns.get_loc('Std (ROI gap area in microns^2)') + 1,
                          'Normalised Std (ROI gap area)', std_gap_area_roi_norm)

        # Normalize gap radius
        mean_gap_radius_total_norm = df_results['Mean (Total gap radius in microns)'].values / np.sqrt(
            df_results['Total Image Area (microns^2)'].values)
        std_gap_radius_total_norm = df_results['Std (Total gap radius in microns)'].values / np.sqrt(
            df_results['Total Image Area (microns^2)'].values)
        mean_gap_radius_roi_norm = df_results['Mean (ROI gap radius in microns)'].values / np.sqrt(
            df_results['Fibre Area (ROI, microns^2)'].values)
        std_gap_radius_roi_norm = df_results['Std (ROI gap radius in microns)'].values / np.sqrt(
            df_results['Fibre Area (ROI, microns^2)'].values)
        df_results.insert(df_results.columns.get_loc('Mean (Total gap radius in microns)') + 1,
                          'Normalised Mean (Total gap radius)', mean_gap_radius_total_norm)
        df_results.insert(df_results.columns.get_loc('Std (Total gap radius in microns)') + 1,
                          'Normalised Std (Total gap radius)', std_gap_radius_total_norm)
        df_results.insert(df_results.columns.get_loc('Mean (ROI gap radius in microns)') + 1,
                          'Normalised Mean (ROI gap radius)', mean_gap_radius_roi_norm)
        df_results.insert(df_results.columns.get_loc('Std (ROI gap radius in microns)') + 1,
                          'Normalised Std (ROI gap radius)', std_gap_radius_roi_norm)

        # Normalize gap number
        gap_num_total_norm = df_results['Gaps number (Total)'].values / df_results[
            'Total Image Area (microns^2)'].values
        gap_num_roi_norm = df_results['Gaps number (ROI)'].values / df_results['Fibre Area (ROI, microns^2)'].values
        df_results.insert(df_results.columns.get_loc('Gaps number (Total)') + 1,
                          'Gap density (Total, number/microns^2)', gap_num_total_norm)
        df_results.insert(df_results.columns.get_loc('Gaps number (ROI)') + 1,
                          'Gap density (ROI, number/microns^2)', gap_num_roi_norm)

        # Normalize lacunarity, see https://sci-hub.mksa.top/10.1016/j.jsg.2010.08.010
        ratio = df_results['Total Length (microns)'].values * df_results['Image Res. (microns/pp)'].values / df_results[
            'Total Image Area (microns^2)'].values
        normalized_lacunarity = (df_results['Lacunarity'].values - 1) / (1.0 / ratio - 1)
        df_results.insert(df_results.columns.get_loc('Lacunarity') + 1,
                          'Normalised Lacunarity', normalized_lacunarity)

        # if no segmentation is performed, no need to re-calculate statistics
        if not self.args["Initialization"]["Segmentation"]:
            Log.logger.info(
                "No segmentation results available. Dropping columns with 'ROI'.")
            df_results.drop(list(df_results.filter(regex="ROI")), axis=1, inplace=True)

        # Save to new csv file
        final_csv = join_path(self.output_folder, 'Quantification_Results.csv')
        df_results.to_csv(final_csv, index=False, float_format='%.3f')
        Log.logger.info('Statistics have been normalised and saved to ' + final_csv)

    def collect_info_maps(self):
        # copy skeleton, curvature and length map to Exports folder
        msk_sub_folders = [f.name for f in os.scandir(join_path(self.output_folder, 'Masks')) if f.is_dir()]
        for msk_sub_folder in msk_sub_folders:
            if msk_sub_folder.startswith("AnaMorf v"):
                xml_file = join_path(self.output_folder, 'Masks', msk_sub_folder, "properties.xml")
                tree = ET.parse(xml_file)
                cur_wnd_sz = tree.getroot().find(".//entry[@key='Curvature Window']").text
                img_paths = glob(join_path(self.output_folder, 'Masks', msk_sub_folder, '*.png')) + \
                            glob(join_path(self.output_folder, 'Masks', msk_sub_folder, '*.tif'))
                for img_path in img_paths:
                    basename = os.path.basename(img_path)
                    if img_path.endswith("Curve_Map.tif"):
                        shutil.copy(img_path, join_path(self.output_folder,
                                                        "Exports", basename[:-4] +
                                                        "_{:.0f}".format(float(cur_wnd_sz)) + ".tif"))
                    else:
                        shutil.copy(img_path, join_path(self.output_folder, "Exports"))
            elif msk_sub_folder.startswith("GapAnalysis"):
                img_paths = glob(join_path(self.output_folder, 'Masks', msk_sub_folder, '*.png'))
                for img_path in img_paths:
                    shutil.copy(img_path, join_path(self.output_folder, "Exports"))

        # organize images into individual folders
        export_img_paths = glob(join_path(self.output_folder, "Exports", "*.png")) + \
                           glob(join_path(self.output_folder, "Exports", "*.tif*"))
        ori_img_paths = glob(join_path(self.output_folder, "ROIs", "*.png"))

        for ori_img_path in ori_img_paths:
            ori_img_name = os.path.basename(ori_img_path)
            name_wo_ext = ori_img_name[:ori_img_name.rindex('.')]

            Path(join_path(self.export_dir, name_wo_ext)).mkdir(parents=True, exist_ok=True)
            for export_img_path in export_img_paths:
                if name_wo_ext in os.path.basename(export_img_path):
                    Path.rename(Path(export_img_path),
                                Path(join_path(self.export_dir, name_wo_ext, os.path.basename(export_img_path))))

    def generate_color_maps(self):
        ori_img_paths = get_img_paths(join_path(self.output_folder, "Eligible"))

        for ori_img_path in ori_img_paths:
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
            sbs_color_map(rgb_img, orient_map, join_path(self.color_dir, name_wo_ext,
                                                         name_wo_ext + "_color_orientation.png"), cbar_label="Orientation")
            sbs_color_map(rgb_img, cohere_map, join_path(self.color_dir, name_wo_ext,
                                                         name_wo_ext + "_color_coherency.png"), cbar_label="Coherency")

            vector_field = orient_vf(rgb_img, orient_map, cohere_map)
            Image.fromarray(vector_field).save(
                join_path(self.color_dir, name_wo_ext, name_wo_ext + "_orient_vf_wgts_coherency.png"))

            vector_field = orient_vf(rgb_img, orient_map, energy_map)
            Image.fromarray(vector_field).save(
                join_path(self.color_dir, name_wo_ext, name_wo_ext + "_orient_vf_wgts_energy.png"))
            vector_field = orient_vf(rgb_img, orient_map, None)
            Image.fromarray(vector_field).save(
                join_path(self.color_dir, name_wo_ext, name_wo_ext + "_orient_vf_constant.png"))

            color_length = info_color_map(rgb_img, length_map, cbar_label="Length (microns)", cmap="plasma", radius=1)
            Image.fromarray(color_length).save(join_path(self.color_dir, name_wo_ext, name_wo_ext + "_color_length.png"))

            curve_paths = glob(join_path(self.export_dir, name_wo_ext, name_wo_ext + "_Curve_Map *"))
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
        self.remove_oversized_imgs()
        self.generate_rois()

        if self.args["Initialization"]["Quantification"]:
            self.quantify_images()
            self.generate_visualizations()
            self.calc_fibre_areas()
            self.gap_analysis()
            self.correct_gap_analysis()
            self.combine_statistics()
            self.normalize_statistics()
            # self.collect_info_maps()
            self.generate_color_maps()
        else:
            Log.logger.info('Segmentation is done. No further analysis will be conducted.')
            # os._exit(0)
