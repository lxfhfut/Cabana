import os
import time
import shutil
import yaml
from pathlib import Path
from utils import split2batches, contains_oversized
from utils import join_path, get_img_paths
from version_info import get_version_info
from cabana import Cabana

import getpass
import datetime
from log import Log

class BatchProcessor():
    def __init__(self, param_folder, input_folder, output_folder, batch_size=5):
        """
        Initialize a BatchProcessor with the given parameters.

        Parameters:
        ----------
        param_folder : str
            Path to the folder containing Parameters.yml
        input_folder : str
            Path to the folder containing input images
        output_folder : str
            Path to the output folder
        batch_size : int, optional
            Size of batches for processing, by default 5
        """
        self.batch_size = batch_size
        self.batch_num = -1
        self.resume = False
        self.ignore_large = True
        self.param_folder = param_folder
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.progress_callback = None  # Add a callback for progress updates

        # Validate inputs
        if not os.path.exists(self.param_folder) or not os.listdir(self.param_folder):
            print("Invalid parameter directory. Abort!")
            os._exit(1)

        if not os.path.exists(self.input_folder) or not os.listdir(self.input_folder):
            print("Invalid input directory. Abort!")
            os._exit(1)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder, exist_ok=True)

        # Create 'Logs' folder and print out parameters
        Log.init_log_path(join_path(os.path.dirname(self.input_folder), 'Logs'))
        Log.logger.info('Logs folder: {}'.format(join_path(os.path.dirname(self.output_folder), 'Logs')))

        # print out parameters
        self.args = None
        param_path = join_path(self.param_folder, "Parameters.yml")
        with open(param_path) as pf:
            try:
                self.args = yaml.safe_load(pf)
            except yaml.YAMLError as exc:
                Log.logger.error(exc)
        Log.log_parameters(param_path)

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

    # Add this method to the BatchProcessor class for progress callback
    def update_progress(self, value):
        """
        Update the progress callback if available.

        Parameters:
        ----------
        value : int
            Progress value (0-100)
        """
        if self.progress_callback:
            self.progress_callback(value)

    # Modify the process method to report progress
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
                # For GUI mode, we'll default to ignoring large images
                self.ignore_large = True
                Log.logger.warning(f"Oversized (> {max_res:d}x{max_res:d} pixels) images will be ignored.")

            with open(join_path(self.output_folder, '.CheckPoint.txt'), 'w') as ckpt:
                ckpt.write('Input Folder,{}\n'.format(self.input_folder))
                ckpt.write('Batch Size,{}\n'.format(self.batch_size))
                ckpt.write('Batch Number,-1\n')
                ckpt.write('Ignore Large,{}\n'.format(str(self.ignore_large)))

        # Initial progress update
        self.update_progress(5)

        # Export version info
        version_info_file = join_path(self.output_folder, 'version_params.yaml')
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

        # Update progress after setup
        self.update_progress(10)

        start_batch_idx = self.batch_num + 1 if self.resume else 0
        end_batch_idx = len(path_batches)  # int(np.ceil(len(img_paths) / self.batch_size))
        for batch_idx in range(start_batch_idx, end_batch_idx):
            Log.logger.info(f'Processing batch {batch_idx + 1}/{end_batch_idx} '
                            f'of {len(path_batches[batch_idx])} images '
                            f'with resolution {res_batches[batch_idx]}um/pixel.')
            self.batch_num = batch_idx
            batch_folder = join_path(self.output_folder, 'Batches', 'batch_' + str(batch_idx))
            Path(batch_folder).mkdir(parents=True, exist_ok=True)
            batch_cabana = Cabana(self.param_folder, self.input_folder,
                                  batch_folder, self.batch_size, batch_idx, self.ignore_large)
            batch_cabana.run()
            with open(join_path(self.output_folder, '.CheckPoint.txt'), 'r') as ckpt:
                lines = ckpt.readlines()
            lines[2] = 'Batch Number,{}\n'.format(batch_idx)
            with open(join_path(self.output_folder, '.CheckPoint.txt'), 'w') as ckpt:
                ckpt.writelines(lines)

            # Calculate and report progress (scale from 10-90%)
            progress = int(10 + (batch_idx + 1) / end_batch_idx * 80)
            self.update_progress(progress)

    # Modify run method to update progress at completion
    def run(self):
        self.check_running_status()
        print("before process")
        self.process()
        print("before post process")
        self.post_process()
        # Final progress update
        self.update_progress(100)