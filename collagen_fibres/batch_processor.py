import os
import time
import shutil
import yaml
import pandas as pd
from glob import glob
from version_info import export_version_info
import xml.etree.ElementTree as ET


from pathlib import Path
from utils import split2batches, contains_oversized, export_parameters
from utils import create_folder, join_path, get_img_paths
from cabana import Cabana

from tkinter import *
from tkinter import filedialog
import torch


class BatchProcessor:
    def __init__(self, batch_size=5):
        self.batch_size = batch_size
        self.batch_num = -1
        self.resume = False
        self.ignore_oversized = True
        gui = Tk()
        gui.withdraw()
        self.program_folder = filedialog.askdirectory(initialdir=os.path.expanduser("~/Documents/"),
                                                      title="Choose Program Directory")
        if len(self.program_folder) == 0 or len(os.listdir(self.program_folder)) == 0:
            print("An empty path/folder has been selected. Aborting ...")
            os._exit(1)

        print(self.program_folder + " has been selected.")
        self.input_folder = filedialog.askdirectory(initialdir=os.path.dirname(self.program_folder),
                                                    title="Choose Input Directory")
        if len(self.input_folder) == 0 or len(os.listdir(self.input_folder)) == 0:
            print("An empty path/folder has been selected. Aborting ...")
            os._exit(1)

        print(self.input_folder + " has been selected.")
        self.output_folder = filedialog.askdirectory(initialdir=os.path.dirname(self.input_folder),
                                                     title="Choose Output Directory")
        if len(self.output_folder) == 0:
            print("An empty path has been selected. Aborting ...")
            os._exit(1)

        print(self.output_folder + " has been selected.")
        gui.destroy()

        self.args = None
        param_path = join_path(self.program_folder, "Parameters.yml")
        with open(param_path) as pf:
            try:
                self.args = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                print(exc)

    def check_running_status(self):
        if os.path.exists(join_path(self.output_folder, 'check_point.txt')):
            input_folder = ""
            batch_size = 5
            batch_num = 0
            ignore_oversized = False
            print("There seems a checkpoint file in the output folder.", end='')
            with open(join_path(self.output_folder, 'check_point.txt'), "r") as f:
                lines = f.readlines()
                for line in lines:
                    param_pair = line.rstrip().split(",")
                    key = param_pair[0]
                    value = param_pair[1]
                    if key == "Input Folder":
                        input_folder = value
                    elif key == "Batch Size":
                        batch_size = int(value)
                    elif key == "Batch Number":
                        batch_num = int(value)
                    elif key == "Ignore oversized":
                        ignore_oversized = True if value.lower() == 'true' else False
                    else:
                        pass
            if os.path.exists(input_folder):
                self.resume = os.path.samefile(input_folder, self.input_folder)
            else:
                self.resume = False

            # Briefly check if all sub-folders exist in the output folder
            for batch_idx in range(self.batch_num+1):
                if not os.path.exists(join_path(self.output_folder, 'Batches', 'batch_' + str(batch_idx))):
                    print('However, some necessary sub-folders are missing. A new run will start...')
                    self.resume = False
                    break

            while self.resume:
                user_input = input(" Do you want to resume from last checkpoint? ([y]/n): ")
                if user_input.lower() == "y" or user_input.lower() == "yes":
                    print('Resuming from last check point...')
                    self.resume = True
                    self.batch_size = batch_size
                    self.batch_num = batch_num
                    self.ignore_oversized = ignore_oversized
                    break
                elif user_input.lower() == "n" or user_input.lower() == "no":
                    print("Starting a new run...")
                    self.resume = False
                    break
                else:
                    print("Invalid input. Please enter y or n.")
        else:
            print("No checkpoint file found. Starting a new run...")
            self.resume = False

    def post_process(self):
        if not os.path.exists(join_path(self.output_folder, "Batches")) or len(os.listdir(join_path(self.output_folder, "Batches"))) == 0:
            print("No results found in output folder!")
            return

        print('Putting together everything...')
        sub_folders = ['ROIs', 'Bins', 'Masks', 'HDM', 'Exports', 'Ridges', 'Eligible', 'Colors']
        for sub_folder in sub_folders:
            create_folder(join_path(self.output_folder, sub_folder, ""))
        create_folder(join_path(self.output_folder, "Masks", "GapAnalysis"))

        # copy images to corresponding folders
        for batch_idx in range(self.batch_num+1):
            batch_folder = join_path(self.output_folder, 'Batches', "batch_"+str(batch_idx))
            for sub_folder in sub_folders:
                src_folder = join_path(batch_folder, sub_folder)
                dst_folder = join_path(self.output_folder, sub_folder)
                img_paths = glob(join_path(src_folder, '*.tif')) \
                            + glob(join_path(src_folder, '*.png')) \
                            + glob(join_path(src_folder, '*.jpg'))
                img_paths.sort()
                for img_path in img_paths:
                    shutil.copy(img_path, dst_folder)

            # copy gap analysis results
            src_folder = join_path(batch_folder, 'Masks', 'GapAnalysis')
            dst_folder = join_path(self.output_folder, 'Masks', 'GapAnalysis')
            img_paths = glob(join_path(src_folder, '*.png')) + glob(join_path(src_folder, '*.csv'))
            img_paths.sort()
            for img_path in img_paths:
                shutil.copy(img_path, dst_folder)

            # copy folders in Exports and Colors
            exports_folders = [f.name for f in os.scandir(join_path(batch_folder, "Exports")) if f.is_dir()]
            for folder in exports_folders:
                shutil.copytree(join_path(batch_folder, "Exports", folder),
                                join_path(self.output_folder, "Exports", folder), dirs_exist_ok=True)

            colors_folders = [f.name for f in os.scandir(join_path(batch_folder, "Colors")) if f.is_dir()]
            for folder in colors_folders:
                shutil.copytree(join_path(batch_folder, "Colors", folder),
                                join_path(self.output_folder, "Colors", folder), dirs_exist_ok=True)

        # copy ignored images
        ignored_images = []
        for batch_idx in range(self.batch_num+1):
            with open(join_path(self.output_folder,
                                'Batches', "batch_" + str(batch_idx), 'Eligible', 'Ignored_images.txt'), 'r') as f:
                lines = f.readlines()
            if len(lines) > 0:
                ignored_images.extend(lines)
        with open(join_path(self.output_folder, 'Eligible', 'Ignored_images.txt'), 'w') as f:
            f.writelines(ignored_images)

        # copy Results_ROI in Bins folder
        merged_df = pd.DataFrame()
        for batch_idx in range(self.batch_num+1):
            if os.path.exists(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx), 'Bins', 'Results_ROI.csv')):
                df = pd.read_csv(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx), 'Bins', 'Results_ROI.csv'))
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        merged_df.to_csv(join_path(self.output_folder, 'Bins', 'Results_ROI.csv'), index=False)

        # copy _ResultsHDM in HDM folder
        merged_df = pd.DataFrame()
        for batch_idx in range(self.batch_num+1):
            if os.path.exists(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx), 'HDM', '_ResultsHDM.csv')):
                df = pd.read_csv(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx), 'HDM', '_ResultsHDM.csv'))
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        merged_df.to_csv(join_path(self.output_folder, 'HDM', '_ResultsHDM.csv'), index=False)

        # copy summary text in GapAnalysis folder
        summary = ["filename mean sd percentile5 median percentile95\n"]
        for batch_idx in range(self.batch_num + 1):
            if os.path.exists(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx),
                                        'Masks', 'GapAnalysis', 'GapAnalysisSummary.txt')):
                with open(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx),
                                    'Masks', 'GapAnalysis', 'GapAnalysisSummary.txt'), 'r') as f:
                    lines = f.readlines()
                if len(lines) > 1:
                    summary.extend(lines[1:])

        if sum(1 for s in summary if "\n" in s) > 1:
            with open(join_path(self.output_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummary.txt'), 'w') as f:
                f.writelines(summary)

        # merge summary csv file in GapAnalysis folder
        merged_df = pd.DataFrame()
        for batch_idx in range(self.batch_num + 1):
            if os.path.exists(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx),
                                        'Masks', 'GapAnalysis', 'IntraGapAnalysisSummary.csv')):
                df = pd.read_csv(join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx),
                                           'Masks', 'GapAnalysis', 'IntraGapAnalysisSummary.csv'))
                merged_df = pd.concat([merged_df, df], ignore_index=True)

        if len(merged_df) > 0:
            merged_df.to_csv(
                join_path(self.output_folder, 'Masks', 'GapAnalysis', 'IntraGapAnalysisSummary.csv'), index=False)

        # merge Quantification_Results in output folder
        merged_df = pd.DataFrame()
        for batch_idx in range(self.batch_num + 1):
            if os.path.exists(join_path(self.output_folder,
                                        'Batches', "batch_" + str(batch_idx), 'Quantification_Results.csv')):
                df = pd.read_csv(join_path(self.output_folder,
                                           'Batches', "batch_" + str(batch_idx), 'Quantification_Results.csv'))
                merged_df = pd.concat([merged_df, df], ignore_index=True)
        merged_df.to_csv(join_path(self.output_folder, 'Quantification_Results.csv'), index=False)

        if os.path.exists(join_path(self.output_folder, 'check_point.txt')):
            os.remove(join_path(self.output_folder, 'check_point.txt'))

    def process(self):
        img_paths = get_img_paths(self.input_folder)
        if len(img_paths) == 0:
            print('No images found in the input folder. Only tif, png, and jpg images are supported!')
            return

        path_batches, res_batches = split2batches(img_paths, self.batch_size)
        if not self.resume:
            shutil.rmtree(self.output_folder)
            os.mkdir(self.output_folder)
            max_res = self.args['Segmentation']["Max Size"]
            if contains_oversized(img_paths, max_res):
                answer = input(f"Oversized (> {max_res:d}x{max_res:d} pixels) "
                               f"images have been detected. Do you want to ignore them? ([y]/n): ")
                self.ignore_oversized = False if answer.lower() == "no" or answer.lower() == "n" else True

            with open(join_path(self.output_folder, 'check_point.txt'), 'w') as ckpt:
                ckpt.write('Input Folder,{}\n'.format(self.input_folder))
                ckpt.write('Batch Size,{}\n'.format(self.batch_size))
                ckpt.write('Batch Number,-1\n'.format(self.batch_num))
                ckpt.write('Ignore Oversized,{}\n'.format(str(self.ignore_oversized)))

        # export version info before processing,
        # so that version info is available even if the subsequent analysis goes wrong
        version_info_file = join_path(self.output_folder, 'version_params.yaml')
        with open(version_info_file, 'w') as file:
            try:
                yaml.dump(self.args, file)
            except yaml.YAMLError as exc:
                print(exc)

        start_batch_idx = self.batch_num+1 if self.resume else 0
        end_batch_idx = len(path_batches)  # int(np.ceil(len(img_paths) / self.batch_size))
        for batch_idx in range(start_batch_idx, end_batch_idx):
            print('Processing batch {}/{} with size {} and res. {} um/pixel ...'.format(batch_idx+1,
                                                                                     end_batch_idx,
                                                                                     len(path_batches[batch_idx]),
                                                                                     res_batches[batch_idx]))
            self.batch_num = batch_idx
            batch_folder = join_path(self.output_folder, 'Batches', 'batch_' + str(batch_idx))
            Path(batch_folder).mkdir(parents=True, exist_ok=True)
            batch_cabana = Cabana(self.program_folder, self.input_folder,
                                  batch_folder, self.batch_size, batch_idx, self.ignore_oversized)
            batch_cabana.run()
            with open(join_path(self.output_folder, 'check_point.txt'), 'r') as ckpt:
                lines = ckpt.readlines()
            lines[2] = 'Batch Number,{}\n'.format(batch_idx)
            with open(join_path(self.output_folder, 'check_point.txt'), 'w') as ckpt:
                ckpt.writelines(lines)

    def run(self):
        self.check_running_status()
        self.process()
        self.post_process()


if __name__ == "__main__":
    start_time = time.time()
    batch_processor = BatchProcessor(5)
    batch_processor.run()
    time_secs = time.time() - start_time
    hours = time_secs // 3600
    minutes = (time_secs % 3600) // 60
    seconds = (time_secs % 3600) % 60
    print("--- {:.0f} hours {:.0f} mins {:.0f} seconds ---".format(hours, minutes, seconds))
    os._exit(0)
