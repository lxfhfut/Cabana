@echo OFF
rem How to run a Python script in a given conda environment from a batch file.

rem It doesn't require:
rem - conda to be in the PATH
rem - cmd.exe to be initialized with conda init

rem Define here the path to your conda installation
set CONDAPATH=C:\Users\admin-astmag\anaconda3\
rem Define here the name of the environment
set ENVNAME=collagen_fibres
set ENVPATH=%CONDAPATH%\envs\%ENVNAME%

rem Activate the conda environment
rem Using call is required here, see: https://stackoverflow.com/questions/24678144/conda-environments-and-bat-files
call %CONDAPATH%\Scripts\activate.bat %ENVPATH%

rem run program update is needed
rem cd "C:\Program Files\P-Cabana\collagen_fibres"
rem python "C:\Program Files\P-Cabana\collagen_fibres\version_info.py"

rem Run a python script in that environment
streamlit run "D:\P-Cabana\collagen_fibres\app.py"

rem Deactivate the environment
call conda deactivate

pause
rem If conda is directly available from the command line then the following code works.
rem call activate someenv
rem python script.py
rem conda deactivate

rem One could also use the conda run command
rem conda run -n someenv python script.py