import os
import time
import shutil
import yaml
import pandas as pd
from glob import glob
from pathlib import Path
from utils import split2batches, contains_oversized
from utils import create_folder, join_path, get_img_paths
from version_info import get_version_info
from batch import BatchCabana

from tkinter import Tk
from tkinter import filedialog
import getpass
import datetime
from log import Log


class BatchProcessor:
    def __init__(self, batch_size=5):
        self.batch_size = batch_size
        self.batch_num = -1
        self.resume = False
        self.ignore_large = True
        gui = Tk()
        gui.withdraw()
        self.param_file = filedialog.askopenfilename(initialdir=os.path.expanduser("~/Documents/"),
                                                      title="Choose Parameter File")
        if len(self.param_file) == 0 or os.path.exists(self.param_file) is False:
            print("No path/folder has been selected. Abort!")
            os._exit(1)
        print(self.param_file + " has been selected.")

        self.input_folder = filedialog.askdirectory(initialdir=Path(self.param_file).parent.parent,
                                                    title="Choose Input Directory")
        if len(self.input_folder) == 0 or len(os.listdir(self.input_folder)) == 0:
            print("No path/folder has been selected. Abort.")
            os._exit(1)
        print(self.input_folder + " has been selected.")

        self.output_folder = filedialog.askdirectory(initialdir=os.path.dirname(self.input_folder),
                                                     title="Choose Output Directory")
        if len(self.output_folder) == 0:
            print("No path/folder has been selected. Abort.")
            os._exit(1)
        print(self.output_folder + " has been selected.")
        gui.destroy()

        # Create 'Logs' folder and print out parameters
        Log.init_log_path(join_path(os.path.dirname(self.input_folder), 'Logs'))
        Log.logger.info('Logs folder: {}'.format(join_path(os.path.dirname(self.output_folder), 'Logs')))

        # print out parameters
        self.args = None
        with open(self.param_file) as pf:
            try:
                self.args = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                Log.logger.error(exc)
        Log.log_parameters(self.param_file)

    def check_running_status(self):
        if os.path.exists(join_path(self.output_folder, '.CheckPoint.txt')):
            input_folder = ""
            batch_size = 5
            batch_num = 0
            ignore_large = False
            Log.logger.warning("A checkpoint file exists in the output folder.")
            with open(join_path(self.output_folder, '.CheckPoint.txt'), "r") as f:
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
                    elif key == "Ignore Large":
                        ignore_large = True if value.lower() == 'true' else False
                    else:
                        pass
            if os.path.exists(input_folder):
                self.resume = os.path.samefile(input_folder, self.input_folder)
            else:
                self.resume = False

            # Briefly check if all sub-folders exist in the output folder
            for batch_idx in range(self.batch_num+1):
                if not os.path.exists(join_path(self.output_folder, 'Batches', 'batch_' + str(batch_idx))):
                    Log.logger.warning('However, some necessary sub-folders are missing. A new run will start.')
                    self.resume = False
                    break

            while self.resume:
                user_input = input("Do you want to resume from last checkpoint? ([y]/n): ")
                if user_input.lower() == "y" or user_input.lower() == "yes":
                    Log.logger.info('Resuming from last check point.')
                    self.resume = True
                    self.batch_size = batch_size
                    self.batch_num = batch_num
                    self.ignore_large = ignore_large
                    break
                elif user_input.lower() == "n" or user_input.lower() == "no":
                    Log.logger.info("Starting a new run.")
                    self.resume = False
                    break
                else:
                    Log.logger.warning("Invalid input. Please enter y or n.")
        else:
            Log.logger.info("No checkpoint file found. Starting a new run.")
            self.resume = False

    def post_process(self):
        if not os.path.exists(join_path(self.output_folder, "Batches")) or len(os.listdir(join_path(self.output_folder, "Batches"))) == 0:
            Log.logger.warning("No results found in output folder!")
            return

        Log.logger.info('Putting together everything.')
        sub_folders = ['ROIs', 'Bins', 'Masks', 'HDM', 'Exports', 'Fibres', 'Eligible', 'Colors']
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
                                'Batches', "batch_" + str(batch_idx), 'Eligible', 'IgnoredImages.txt'), 'r') as f:
                lines = f.readlines()
            if len(lines) > 0:
                ignored_images.extend(lines)
        with open(join_path(self.output_folder, 'Eligible', 'IgnoredImages.txt'), 'w') as f:
            f.writelines(ignored_images)

        # copy Results_ROI in Bins folder
        roi_df = []
        for batch_idx in range(self.batch_num+1):
            batch_folder = join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx))
            if os.path.exists(join_path(batch_folder, 'Bins', 'ResultsROI.csv')):
                roi_df.append(pd.read_csv(join_path(batch_folder, 'Bins', 'ResultsROI.csv')))
        if len(roi_df) > 0:
            merged_df = pd.concat(roi_df, ignore_index=True)
            merged_df.to_csv(join_path(self.output_folder, 'Bins', 'ResultsROI.csv'), index=False)

        # copy ResultsHDM in HDM folder
        hdm_df = []
        for batch_idx in range(self.batch_num+1):
            batch_folder = join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx))
            if os.path.exists(join_path(batch_folder, 'HDM', 'ResultsHDM.csv')):
                hdm_df.append(pd.read_csv(join_path(batch_folder, 'HDM', 'ResultsHDM.csv')))

        if len(hdm_df) > 0:
            merged_df = pd.concat(hdm_df, ignore_index=True)
            merged_df.to_csv(join_path(self.output_folder, 'HDM', 'ResultsHDM.csv'), index=False)

        # copy summary of all gap circles in GapAnalysis folder
        summary_df = []
        for batch_idx in range(self.batch_num + 1):
            batch_folder = join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx))
            if os.path.exists(join_path(batch_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummary.csv')):
                summary_df.append(pd.read_csv(
                    join_path(batch_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummary.csv')))

        if len(summary_df) > 0:
            merged_df = pd.concat(summary_df, ignore_index=True)
            merged_df.to_csv(
                join_path(self.output_folder, 'Masks', 'GapAnalysis', 'GapAnalysisSummary.csv'), index=False)

        # merge summary of intra gap circles in GapAnalysis folder
        intra_summary_df = []
        for batch_idx in range(self.batch_num + 1):
            batch_folder = join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx))
            if os.path.exists(join_path(batch_folder, 'Masks', 'GapAnalysis', 'IntraGapAnalysisSummary.csv')):
                intra_summary_df.append(pd.read_csv(
                    join_path(batch_folder, 'Masks', 'GapAnalysis', 'IntraGapAnalysisSummary.csv')))

        if len(intra_summary_df) > 0:
            merged_df = pd.concat(intra_summary_df, ignore_index=True)
            merged_df.to_csv(
                join_path(self.output_folder, 'Masks', 'GapAnalysis', 'IntraGapAnalysisSummary.csv'), index=False)

        # merge QuantificationResults in output folder
        results_df = []
        for batch_idx in range(self.batch_num + 1):
            batch_folder = join_path(self.output_folder, 'Batches', "batch_" + str(batch_idx))
            if os.path.exists(join_path(batch_folder, 'QuantificationResults.csv')):
                results_df.append(pd.read_csv(
                    join_path(batch_folder, 'QuantificationResults.csv')))
        if len(results_df) > 0:
            merged_df = pd.concat(results_df, ignore_index=True)
            merged_df.to_csv(join_path(self.output_folder, 'QuantificationResults.csv'), index=False)

        # remove CheckPoint.txt if program is run successfully
        if os.path.exists(join_path(self.output_folder, '.CheckPoint.txt')):
            os.remove(join_path(self.output_folder, '.CheckPoint.txt'))

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
                self.ignore_large = False if answer.lower() == "no" or answer.lower() == "n" else True

            with open(join_path(self.output_folder, '.CheckPoint.txt'), 'w') as ckpt:
                ckpt.write('Input Folder,{}\n'.format(self.input_folder))
                ckpt.write('Batch Size,{}\n'.format(self.batch_size))
                ckpt.write('Batch Number,-1\n')
                ckpt.write('Ignore Large,{}\n'.format(str(self.ignore_large)))

        # export version info before processing,
        # # so that version info is available even if the subsequent analysis goes wrong
        version_info_file = join_path(self.output_folder, 'version_params.yaml')
        # with open(version_info_file, 'w') as file:
        #     try:
        #         yaml.dump(self.args, file)
        #     except yaml.YAMLError as exc:
        #         Log.logger.error(exc)
        with open(version_info_file, 'w') as file:
            try:
                # Get metadata
                username = getpass.getuser()
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                version_info = get_version_info()

                # Create metadata dictionary
                metadata = {
                    "execution_info": {
                        "user": username,
                        "datetime": timestamp,
                        "version": version_info["version"],
                        "git_info": {
                            "commit": version_info.get("git_hash", "unknown"),
                            "branch": version_info.get("git_branch", "unknown"),
                            "tag": version_info.get("latest_tag", "unknown")
                        }
                    }
                }

                # Write metadata first, then args
                yaml.dump(metadata, file, default_flow_style=False)
                file.write("\n# Program Arguments\n")
                yaml.dump(self.args, file, default_flow_style=False)
            except yaml.YAMLError as exc:
                Log.logger.error(exc)

        start_batch_idx = self.batch_num+1 if self.resume else 0
        end_batch_idx = len(path_batches)  # int(np.ceil(len(img_paths) / self.batch_size))
        for batch_idx in range(start_batch_idx, end_batch_idx):
            Log.logger.info(f'Processing batch {batch_idx+1}/{end_batch_idx} '
                            f'of {len(path_batches[batch_idx])} images '
                            f'with resolution {res_batches[batch_idx]}um/pixel.')
            self.batch_num = batch_idx
            batch_folder = join_path(self.output_folder, 'Batches', 'batch_' + str(batch_idx))
            Path(batch_folder).mkdir(parents=True, exist_ok=True)
            batch_cabana = BatchCabana(self.param_file, self.input_folder,
                                  batch_folder, self.batch_size, batch_idx, self.ignore_large)
            batch_cabana.run()
            with open(join_path(self.output_folder, '.CheckPoint.txt'), 'r') as ckpt:
                lines = ckpt.readlines()
            lines[2] = 'Batch Number,{}\n'.format(batch_idx)
            with open(join_path(self.output_folder, '.CheckPoint.txt'), 'w') as ckpt:
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
    Log.logger.info("--- {:.0f} hours {:.0f} mins {:.0f} seconds ---".format(hours, minutes, seconds))
    os._exit(0)
