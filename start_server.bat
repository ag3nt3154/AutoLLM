@echo off
REM Open Anaconda Prompt, activate environment, navigate, and start Uvicorn

REM Set the conda environment name
set CONDA_ENV=llm-env

REM Set the directory where the server script is located
set SERVER_DIR=C:\Users\alexr\AutoLLM

REM Set the Uvicorn server command
set UVICORN_CMD=uvicorn start_server:app --host 0.0.0.0 --port 8000

REM Activate the conda environment
call "%USERPROFILE%\anaconda3\condabin\conda.bat" activate %CONDA_ENV%

REM Navigate to the specified directory
cd /d %SERVER_DIR%

REM Start the Uvicorn server
%UVICORN_CMD%

REM Keep the prompt open
pause
