import os
import csv
import cv2
import imagej
import shutil
import numpy as np
import pandas as pd
from glob import glob
from log import Log
from pathlib import Path
from utils import create_folder, join_path
from skimage.feature import peak_local_max
from sklearn.metrics.pairwise import euclidean_distances
from segmenter import parse_args, segment_single_image, visualize_fibres

from tkinter import *
from tkinter import filedialog


class Cabana:
    def __init__(self):
        self.program_name = "Twombli_v4.ijm"
        self.seg_param_file = "SegmenterParameters.txt"
        self.twombli_param_file = "TwombliParameters.txt"
        self.anamorf_param_file = "AnamorfProperties.xml"
        # self.resume = False
        # self.batch_size = 5
        # self.batch_number = -1
        self.args = parse_args()
        gui = Tk()
        gui.withdraw()
        self.program_folder = filedialog.askdirectory(initialdir=os.path.expanduser("~/Documents/"),
                                                      title="Choose Program Directory")
        if len(self.program_folder) == 0 or len(os.listdir(self.program_folder)) == 0:
            print("An empty path/folder has been selected. Aborting ...")
            os._exit(1)

        print(self.program_folder + " has been selected.")
        self.input_folder = filedialog.askdirectory(initialdir=os.path.expanduser("~/Documents/"),
                                                    title="Choose Input Directory")
        if len(self.input_folder) == 0 or len(os.listdir(self.input_folder)) == 0:
            print("An empty path/folder has been selected. Aborting ...")
            os._exit(1)

        print(self.input_folder + " has been selected.")
        self.output_folder = filedialog.askdirectory(initialdir=os.path.expanduser("~/Documents/"),
                                                     title="Choose Output Directory")
        if len(self.output_folder) == 0:
            print("An empty path has been selected. Aborting ...")
            os._exit(1)

        print(self.output_folder + " has been selected.")
        gui.destroy()

        # Create sub-folders in output directory
        self.roi_dir = join_path(self.output_folder, 'ROIs', "")
        self.bin_dir = join_path(self.output_folder, 'Bins', "")
        self.mask_dir = join_path(self.output_folder, 'Masks', "")
        self.hdm_dir = join_path(self.output_folder, 'HDM', "")
        self.road_dir = join_path(self.output_folder, 'Roads', "")
        self.ridge_dir = join_path(self.output_folder, 'Ridges', "")
        self.eligible_dir = join_path(self.output_folder, 'Eligible')

        create_folder(self.eligible_dir)
        create_folder(self.ridge_dir)
        create_folder(self.mask_dir)
        create_folder(self.hdm_dir)
        create_folder(self.road_dir)
        create_folder(self.roi_dir)
        create_folder(self.bin_dir)
        setattr(self.args, 'roi_dir', self.roi_dir)
        setattr(self.args, 'bin_dir', self.bin_dir)

    def check_running_status(self):
        all_content_exist = os.path.exists(self.eligible_dir) and os.path.exists(self.ridge_dir) and \
            os.path.exists(self.mask_dir) and os.path.exists(self.hdm_dir) and os.path.exists(self.road_dir) and \
                    os.path.exists(self.roi_dir) and os.path.exists(self.bin_dir) and \
                    os.path.exists(join_path(self.output_folder, '.check_point.txt'))

        if all_content_exist:
            input_folder = ""

            with open(join_path(self.output_folder, '.check_point.txt'), "r") as f:
                lines = f.readlines()
                for line in lines:
                    param_pair = line.rstrip().split(",")
                    key = param_pair[0]
                    value = param_pair[1]
                    if key == "Input Folder":
                        input_folder = value
                    elif key == "Batch Size":
                        self.batch_size = int(value)
                    elif key == "Batch Number":
                        self.batch_num = int(value)
                    else:
                        pass
            img_paths1 = glob(join_path(self.input_folder, '*.tif')) \
                        + glob(join_path(self.input_folder, '*.png')) \
                        + glob(join_path(self.input_folder, '*.jpg'))
            img_paths2 = glob(join_path(self.eligible_dir, '*.tif')) \
                         + glob(join_path(self.eligible_dir, '*.png')) \
                         + glob(join_path(self.eligible_dir, '*.jpg'))
            if os.path.samefile(input_folder, self.input_folder) and (len(img_paths1) == len(img_paths2)):
                len_roi = len(glob(join_path(self.roi_dir, '*.png')))
                len_bin = len(glob(join_path(self.bin_dir, '*.png')))
                len_road = len(glob(join_path(self.road_dir, '*.png')))
                len_hdm = len(glob(join_path(self.hdm_dir, '*.png')))
                len_ridge = len(glob(join_path(self.ridge_dir, '*.png')))
                len_mask = len(glob(join_path(self.mask_dir, '*.png')))
                len_list = [len_roi, len_mask, len_bin, len_ridge, len_road, len_hdm]
                if all(i >= self.batch_num * self.batch_size for i in len_list):
                    print("There seems a checkpoint file in the output folder.")
                    while True:
                        user_input = input("Do you want to resume from last checkpoint? ([y]/n): ")
                        if user_input.lower() == "y" or user_input == "":
                            print('Resuming from last check point...')
                            self.resume = True
                            break
                        elif user_input.lower() == "n":
                            print("Starting a new run...")
                            self.resume = False
                            break
                        else:
                            print("Invalid input. Please enter y or n.")
        else:
            print("Starting a new run...")
            self.resume = False

    def initialize_params(self):
        Log.init_log_path(join_path(self.program_folder, 'logs'))
        Log.logger.info('Logs folder: {}'.format(join_path(self.program_folder, 'logs')))
        seg_param_path = join_path(self.program_folder, self.seg_param_file)
        with open(seg_param_path) as f:
            lines = f.readlines()
            for line in lines:
                param_pair = line.rstrip().split(",")
                key = param_pair[0]
                value = param_pair[1]
                if key == "Number of Labels":
                    setattr(self.args, 'num_channels', int(value))
                elif key == "Max Iterations":
                    setattr(self.args, 'max_iter', int(value))
                elif key == "Normalized Hue Value":
                    setattr(self.args, 'hue_value', float(value))
                elif key == "Color Threshold":
                    setattr(self.args, 'rt', float(value))
                elif key == "Min Size":
                    setattr(self.args, 'min_size', int(value))
                elif key == "Max Size":
                    setattr(self.args, 'max_size', int(value))
                elif key == "Mode":
                    if value.lower() in ['both', 'twombli', 'segmentation']:
                        setattr(self.args, 'mode', value.lower())
                    else:
                        Log.logger.warning("Invalid mode parameter {}, set to default 'both'.".format(value))
                else:
                    Log.logger.warning('Invalid parameter {}'.format(key))
        Log.log_parameters(seg_param_path)
        # self.args = load_segmenter_parameters(self.args, seg_param_path)
        # remove_oversized_imgs(self.input_folder, self.eligible_dir, self.args.max_size)  # Remove over-sized images
        # self.input_folder = self.eligible_dir

    def remove_oversized_img(self):
        img_paths = glob(join_path(self.input_folder, '*.tif')) \
                    + glob(join_path(self.input_folder, '*.png')) \
                    + glob(join_path(self.input_folder, '*.jpg'))
        img_paths.sort()
        with open(join_path(self.eligible_dir, "Ignored_images.txt"), 'w') as f:
            count = 0
            for img_path in img_paths:
                img = cv2.imread(img_path, 0)
                bright_percent = np.sum(img > 20) / img.shape[0] / img.shape[1]
                if img.shape[0] * img.shape[1] <= self.args.max_size * self.args.max_size and bright_percent > 0.1:
                    img_name = os.path.basename(img_path)
                    shutil.copyfile(img_path, join_path(self.eligible_dir, img_name))
                else:
                    count += 1
                    f.write(img_path)
                    f.write("\n")
            Log.logger.info('{} over-sized images have been ignored.'.format(count))
        self.input_folder = self.eligible_dir

    def generate_rois(self):
        img_paths = glob(join_path(self.input_folder, '*.tif')) \
                    + glob(join_path(self.input_folder, '*.png')) \
                    + glob(join_path(self.input_folder, '*.jpg'))
        img_paths.sort()
        Log.logger.info('Segmenting {} images in {}'.format(len(img_paths), self.input_folder))
        for img_path in img_paths:
            setattr(self.args, 'input', img_path)
            if not self.args.mode == "twombli":
                segment_single_image(self.args)
            else:
                Log.logger.info("No segmentation is applied prior to image analysis.")
                img = cv2.imread(self.args.input)
                mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255
                img_name = os.path.splitext(os.path.basename(self.args.input))[0]
                cv2.imwrite(join_path(self.args.roi_dir, img_name + '_roi.png'), img)
                cv2.imwrite(join_path(self.args.bin_dir, img_name + '_mask.png'), mask)

        Log.logger.info('ROIs have been saved in {}'.format(self.args.roi_dir))
        Log.logger.info('Masks have been saved in {}'.format(self.args.bin_dir))

    def quantify_images(self):
        # Image analysis with twombli
        twombli_param_path = join_path(self.program_folder, self.twombli_param_file)
        Log.log_parameters(twombli_param_path)

        anamorf_param_path = join_path(self.program_folder, self.anamorf_param_file)

        # No space after comma!!!
        macro_args = "\"" + twombli_param_path + "," + self.roi_dir + "," + self.mask_dir + "," + self.hdm_dir + "," + self.road_dir + "," + anamorf_param_path + "\""
        full_macro = """runMacro(\"""" + join_path(self.program_folder, "") + \
                     self.program_name + """\", """ + macro_args + """);"""
        # print(full_macro)
        ij = imagej.init(r'C:\Program Files\fiji-win64\Fiji.app', mode='interactive')
        ij.py.run_macro(full_macro)
        Log.logger.info('Image analyses finished.')
        Log.logger.info('Appending FIJI log...')
        self.append_FIJI_log()

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
        img_paths = glob(join_path(self.input_folder, '*.tif')) \
                    + glob(join_path(self.input_folder, '*.png')) \
                    + glob(join_path(self.input_folder, '*.jpg'))
        img_paths.sort()
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            mask_path = join_path(self.mask_dir, img_name[:-4] + '_roi.png')
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            visualize_ridges(img, mask, join_path(self.ridge_dir, img_name[:-4] + '_ridges.png'), thickness)

    def calc_fibre_areas(self):
        img_paths = glob(join_path(self.bin_dir, '*.png'))
        img_paths.sort()
        with open(join_path(self.bin_dir, 'Results_ROI.csv'), 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Area', '% Black', 'Area (width)', '% Black (width)'])
            Log.logger.info('Calculating fibre areas for {} images'.format(len(img_paths)))
            for img_path in img_paths:
                img_mask = cv2.imread(img_path, 0)
                area = np.sum(img_mask > 128)
                percent_black = area / img_mask.shape[0] / img_mask.shape[1]
                name = os.path.basename(img_path)
                name = name[:-9] + '_roi.png'
                road_mask = cv2.imread(join_path(self.road_dir, name), 0)
                # print(join_path(self.road_dir, name))
                area_road = np.sum(road_mask < 128)
                percent_black_width = area_road / road_mask.shape[0] / road_mask.shape[1]
                data = [name, area, percent_black, area_road, percent_black_width]
                writer.writerow(data)

            Log.logger.info('Areas have been saved in {}'.format(join_path(self.bin_dir, 'Results_ROI.csv')))

    def gap_analysis(self):
        twombli_param_path = join_path(self.program_folder, self.twombli_param_file)
        min_gap_diameter = -1
        with open(twombli_param_path) as f:
            lines = f.readlines()
            for line in lines:
                param_pair = line.rstrip().split(",")
                key = param_pair[0]
                value = param_pair[1]
                if key == "Minimum Gap Diameter":
                    min_gap_diameter = int(value)
                    break
        if min_gap_diameter == -1:
            Log.logger.warning("'Minimum Gap Diameter' can not be found in {}.".format(twombli_param_path))
        elif min_gap_diameter == 0:
            Log.logger.warning("Skipping gap analysis.")
        else:
            gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
            Path(gap_result_dir).mkdir(parents=True, exist_ok=True)
            gap_analysis_file = join_path(gap_result_dir, "GapAnalysisSummary.txt")
            with open(gap_analysis_file, 'w+') as summary_file:
                summary_file.write("filename mean sd percentile5 median percentile95\n")
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
                                                  int(circle[2]), (0, 0, 255), 2)
                    img_name = os.path.basename(img_path)
                    cv2.imwrite(join_path(gap_result_dir, img_name[:-4] + "_GapImage.png"), final_result)
                    areas = np.pi * np.array(final_circles)[:, 2] ** 2
                    summary_file.write("{} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n".format(img_name,
                                                                                      np.mean(areas),
                                                                                      np.std(areas),
                                                                                      np.percentile(areas, 5),
                                                                                      np.median(areas),
                                                                                      np.percentile(areas, 95)))
                    final_circles = np.array(final_circles)
                    data = {'Area': areas,
                            'X': final_circles[:, 1],
                            'Y': final_circles[:, 0]
                            }
                    df = pd.DataFrame(data)
                    df.to_csv(join_path(gap_result_dir, "IndividualGaps_" + img_name[:-4] + ".csv"))

    def correct_gap_analysis(self):
        gap_result_dir = join_path(self.mask_dir, 'GapAnalysis')
        img_paths = glob(join_path(self.bin_dir, '*.png'))
        img_paths.sort()
        names = []
        means = []
        stds = []
        pct5s = []
        medians = []
        pct95s = []
        means_radius = []
        stds_radius = []
        pct5s_radius = []
        medians_radius = []
        pct95s_radius = []
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
            X = []
            Y = []
            for index, row in df_circles.iterrows():
                area, x, y = row['Area'], int(row['X']), int(row['Y'])
                radius = int(np.sqrt(area / 3.1416))
                if binary_mask[y, x] > 0:
                    color_img_ridge = cv2.circle(color_img_ridge, (x, y), radius, (0, 255, 0), 2)
                    areas.append(area)
                    X.append(x)
                    Y.append(y)

            areas = np.array(areas)
            radius = np.sqrt(areas / 3.1416).astype(np.int64)
            cv2.imwrite(join_path(gap_result_dir,
                                  base_name + "_roi_GapImage_corrected.png"), color_img_ridge)
            names.append(base_name + "_roi.png")

            means.append(np.mean(areas))
            stds.append(np.std(areas))
            medians.append(np.median(areas))
            pct5s.append(np.percentile(areas, 5))
            pct95s.append(np.percentile(areas, 95))

            means_radius.append(np.mean(radius))
            stds_radius.append(np.std(radius))
            medians_radius.append(np.median(radius))
            pct5s_radius.append(np.percentile(radius, 5))
            pct95s_radius.append(np.percentile(radius, 95))

        if names:
            data = {'Image': names,
                    'Mean (area)': means,
                    'Std (area)': stds,
                    'Median (area)': medians,
                    'Percentile5 (area)': pct5s,
                    'Percentile95 (area)': pct95s,
                    'Mean (radius)': means_radius,
                    'Std (radius)': stds_radius,
                    'Median (radius)': medians_radius,
                    'Percentile5 (radius)': pct5s_radius,
                    'Percentile95 (radius)': pct95s_radius}
            df = pd.DataFrame(data)
            df.to_csv(join_path(gap_result_dir, 'GapAnalysisSummaryCorrected.csv'))
        else:
            Log.logger.warning('There seems no gap analysis to correct.')

    def combine_statistics(self):
        Log.logger.info('Generating statistics...')

        twombli_csv = join_path(self.output_folder, 'Twombli_Results.csv')
        if not os.path.exists(twombli_csv):
            Log.logger.error(twombli_csv + " does not exist. Make sure TWOMBLI was run flawlessly.")
            os._exit(1)

        df_twombli = pd.read_csv(twombli_csv)

        segmenter_csv = join_path(self.bin_dir, 'Results_ROI.csv')
        df_segmenter = pd.read_csv(segmenter_csv)

        # Check if image names are unique
        twombli_img_names = df_twombli['Image'].values
        segmenter_img_names = df_segmenter['Image'].values

        if len(twombli_img_names) != len(set(twombli_img_names)):
            Log.logger.critical('Images names are not unique in ' + twombli_csv)
            os._exit(1)

        if len(segmenter_img_names) != len(set(segmenter_img_names)):
            Log.logger.critical('Images names are not unique in ' + segmenter_csv)
            os._exit(1)

        area_roi = []
        percent_blk = []
        area_road = []
        percent_blk_width = []
        for i in range(len(df_twombli)):
            img_name = df_twombli.loc[i, 'Image']
            if img_name[:-4] + ".tif" in df_segmenter['Image'].values:
                area_roi.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".tif")).idxmax(), 'Area'])
                percent_blk.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".tif")).idxmax(), '% Black'])
                area_road.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".tif")).idxmax(), 'Area (width)'])
                percent_blk_width.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".tif")).idxmax(), '% Black (width)'])
            elif img_name[:-4] + ".png" in df_segmenter['Image'].values:
                area_roi.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Area'])
                percent_blk.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".png")).idxmax(), '% Black'])
                area_road.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Area (width)'])
                percent_blk_width.append(
                    df_segmenter.loc[(df_segmenter['Image'] == (img_name[:-4] + ".png")).idxmax(), '% Black (width)'])
            else:
                Log.logger.info(img_name + ' cannot be found in ' + segmenter_csv)
                area_roi.append(0)
                percent_blk.append(0)
                area_road.append(0)
                percent_blk_width.append(0)

        df_twombli.insert(df_twombli.columns.get_loc("Area (microns^2)") + 1, "Area ROI", area_roi)
        df_twombli.insert(df_twombli.columns.get_loc("Area ROI") + 1, "Area (width)", area_road)
        df_twombli.insert(df_twombli.columns.get_loc("% High Density Matrix") + 1, "% Black ROI", percent_blk)
        df_twombli.insert(df_twombli.columns.get_loc("% Black ROI") + 1, "% Black ROI (width)", percent_blk_width)

        # Calculate other metrics
        percent = df_twombli.loc[:, '% High Density Matrix'].tolist()
        total_area = df_twombli.loc[:, 'TotalImageArea'].tolist()
        total_length = df_twombli.loc[:, 'Total Length (microns)'].tolist()
        num_endpoints = df_twombli.loc[:, 'Endpoints'].tolist()
        num_branchpoints = df_twombli.loc[:, 'Branchpoints'].tolist()

        avg_length = []
        for l, e, b in zip(total_length, num_endpoints, num_branchpoints):
            if (e + b) == 0:
                avg_length.append(np.NaN)
            else:
                avg_length.append((l * 2) / (e + b))

        avg_thickness = []
        for p, a, l in zip(percent, total_area, total_length):
            if l == 0:
                avg_thickness.append(np.NaN)
            else:
                avg_thickness.append(a * p / l)

        avg_thickness_roi = []
        for a, l in zip(area_roi, total_length):
            if l == 0:
                avg_thickness_roi.append(np.NaN)
            else:
                avg_thickness_roi.append(a / l)

        avg_thickness_road = []
        for a, l in zip(area_road, total_length):
            if l == 0:
                avg_thickness_road.append(np.NaN)
            else:
                avg_thickness_road.append(a / l)

        # Insert more metrics
        df_twombli.insert(df_twombli.columns.get_loc("Total Length (microns)") + 1, "Avg Length", avg_length)
        df_twombli.insert(df_twombli.columns.get_loc("Area (microns^2)") + 1, "Avg Thickness", avg_thickness)
        df_twombli.insert(df_twombli.columns.get_loc("Area ROI") + 1, "Avg Thickness ROI", avg_thickness_roi)
        df_twombli.insert(df_twombli.columns.get_loc("Avg Thickness ROI") + 1, "Avg Thickness (width)",
                          avg_thickness_road)

        means = []
        stds = []
        pct5s = []
        medians = []
        pct95s = []
        means_radius = []
        stds_radius = []
        pct5s_radius = []
        medians_radius = []
        pct95s_radius = []

        gaps_csv = join_path(self.output_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummaryCorrected.csv')
        if os.path.exists(gaps_csv):
            df_gaps = pd.read_csv(gaps_csv)
            gaps_img_names = df_gaps['Image'].values
            if len(gaps_img_names) != len(set(gaps_img_names)):
                Log.logger.critical('Images names are not unique in ' + gaps_csv)
                os._exit(1)

            for i in range(len(df_twombli)):
                img_name = df_twombli.loc[i, 'Image']
                if img_name[:-4] + ".png" in df_gaps['Image'].values:
                    means.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Mean (area)'])
                    stds.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Std (area)'])
                    medians.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Median (area)'])
                    pct5s.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Percentile5 (area)'])
                    pct95s.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Percentile95 (area)'])
                    means_radius.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Mean (radius)'])
                    stds_radius.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Std (radius)'])
                    medians_radius.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Median (radius)'])
                    pct5s_radius.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Percentile5 (radius)'])
                    pct95s_radius.append(
                        df_gaps.loc[(df_gaps['Image'] == (img_name[:-4] + ".png")).idxmax(), 'Percentile95 (radius)'])
                else:
                    Log.logger.error(img_name + ' cannot be found in ' + segmenter_csv)
                    means.append(0)
                    stds.append(0)
                    medians.append(0)
                    pct5s.append(0)
                    pct95s.append(0)
                    means_radius.append(0)
                    stds_radius.append(0)
                    medians_radius.append(0)
                    pct5s_radius.append(0)
                    pct95s_radius.append(0)
        else:
            Log.logger.warning("There seems no gap analysis result available.")
            means = [0] * len(df_twombli)
            stds = [0] * len(df_twombli)
            pct5s = [0] * len(df_twombli)
            medians = [0] * len(df_twombli)
            pct95s = [0] * len(df_twombli)
            means_radius = [0] * len(df_twombli)
            stds_radius = [0] * len(df_twombli)
            pct5s_radius = [0] * len(df_twombli)
            medians_radius = [0] * len(df_twombli)
            pct95s_radius = [0] * len(df_twombli)

        df_twombli.insert(len(df_twombli.columns), "Mean (area)", means)
        df_twombli.insert(len(df_twombli.columns), "Std (area)", stds)
        df_twombli.insert(len(df_twombli.columns), "Median (area)", medians)
        df_twombli.insert(len(df_twombli.columns), "Percentile5 (area)", pct5s)
        df_twombli.insert(len(df_twombli.columns), "Percentile95 (area)", pct95s)

        df_twombli.insert(len(df_twombli.columns), "Mean (radius)", means_radius)
        df_twombli.insert(len(df_twombli.columns), "Std (radius)", stds_radius)
        df_twombli.insert(len(df_twombli.columns), "Median (radius)", medians_radius)
        df_twombli.insert(len(df_twombli.columns), "Percentile5 (radius)", pct5s_radius)
        df_twombli.insert(len(df_twombli.columns), "Percentile95 (radius)", pct95s_radius)

        # Save to new csv file
        final_csv = join_path(self.output_folder, 'Twombli_Results_Final.csv')
        df_twombli.to_csv(final_csv)

        # combine_statistics(segmenter_csv, twombli_csv, gaps_csv, final_csv)
        Log.logger.info('Statistics have been saved to ' + final_csv)
        os._exit(0)

    def run(self):
        self.initialize_params()
        self.remove_oversized_img()
        self.generate_rois()
        if self.args.mode != "segmentation":
            self.quantify_images()
            self.generate_visualizations()
            self.calc_fibre_areas()
            self.gap_analysis()
            self.correct_gap_analysis()
            self.combine_statistics()
        else:
            Log.logger.info('Segmentation is done. No further analysis will be conducted.')
            os._exit(0)


if __name__ == '__main__':
    analyzer = Cabana()
    analyzer.run()
