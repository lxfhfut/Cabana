# Installation

## System Requirements

1. **Processor (CPU):** Intel Core i7 or AMD Ryzen 7, or higher for better performance
2. **Memory (RAM):** Minimum 16 GB, with 32 GB preferred
3. **Graphics Processing Unit (GPU):** NVIDIA GeForce GTX/RTX series GPU with at least 10 GB VRAM

## Installation Instructions

1. Install miniconda. The default installation location is `C:\\ProgramData\\miniconda3` on Windows. Verify the installation by running `conda env list` in the command line. If the installation was successful, you should be able to see a `base` env and the path to the env.

2. Create a conda env named `cabana` (you can change it to any name you want) by running `conda create -n cabana python=3.10` in the command line. If the env is created correctly, activate the created conda env by running `conda activate cabana`.

3. Install Cabana by running `pip install -U cabana`. If the installation is successful, you can start Cabana GUI by running `python -m cabana`. Alternatively, you can import Cabana in your code for more customized analysis (see examples in <a href="_static/tutorial.ipynb" target="_blank" rel="noopener">tutorial.ipynb</a>).

If you want to create a shortcut on Desktop, follow the steps below:

1) Make sure that Git is installed. If not, download it from this link and install it. Verify the installation by running `git --version` in the command line. 

2) Run `git clone https://github.com/lxfhfut/Cabana.git` in the command line. You will be prompted to log in using your GitHub account. If authentication is successful, a `Cabana` folder will be downloaded to your current working directory. You can move (copy-and-paste) the folder to any installation location and change the current working direction to `Cabana` folder.

3) Modify the content in cabana.bat. Make sure CONDAPATH is set to the folder where miniconda was installed, ENVNAME is set to `cabana` and the command on line 19 is set to `python -m cabana`.

4) Create a `Cabana` shortcut on `C:\\Users\\Public\\Desktop` for `cabana.bat`. Change the icon of Cabana to any icon you want, or you can use the `cabana-logo.ico` in the downloaded `Cabana` folder from Github. You are now ready to use Cabana by double-clicking the shortcut.
