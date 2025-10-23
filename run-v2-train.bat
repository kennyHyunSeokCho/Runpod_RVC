@echo off
setlocal

set SCRIPT_DIR=%~dp0
set ROOT=%SCRIPT_DIR%..\..
pushd "%ROOT%"

REM 한글 주석: 실행법
REM   run-v2-train.bat [MODEL_NAME] [DATASET_PATH]
REM   run-v2-train.bat --url [MODEL_NAME] [DATASET_ZIP_URL]
REM 예) run-v2-train.bat my_model assets\datasets\my_model
REM     run-v2-train.bat --url my_model https://example.com/my_dataset.zip

set ARG1=%1
if /I "%ARG1%"=="--url" goto :mode_url

set MODEL_NAME=%1
if "%MODEL_NAME%"=="" set MODEL_NAME=my_model
set DATASET_PATH=%2
if "%DATASET_PATH%"=="" set DATASET_PATH=assets\datasets\%MODEL_NAME%
goto :do_pipeline

:mode_url
set MODEL_NAME=%2
set DATASET_URL=%3
if "%MODEL_NAME%"=="" set MODEL_NAME=my_model
if "%DATASET_URL%"=="" (
  echo URL 모드: DATASET_URL 이 필요합니다.
  exit /b 1
)

set CPU_CORES=%NUMBER_OF_PROCESSORS%
set GPU=0
set BATCH_SIZE=8
set PYTHON=python

if /I "%ARG1%"=="--url" goto :auto_all

echo ==== PREPROCESS (%MODEL_NAME%) ====
%PYTHON% rvc\v2_core\core.py preprocess --model_name "%MODEL_NAME%" --dataset_path "%DATASET_PATH%" --sample_rate 48000 --cpu_cores %CPU_CORES%
if errorlevel 1 goto :error

echo ==== EXTRACT FEATURES (%MODEL_NAME%) ====
%PYTHON% rvc\v2_core\core.py extract --model_name "%MODEL_NAME%" --f0_method rmvpe --cpu_cores %CPU_CORES% --gpu %GPU% --sample_rate 48000 --embedder_model contentvec --include_mutes 2
if errorlevel 1 goto :error

echo ==== TRAIN (48kHz, 500 epochs, no mid-save) ====
%PYTHON% rvc\v2_core\core.py train --model_name "%MODEL_NAME%" --batch_size %BATCH_SIZE% --gpu %GPU% --index_algorithm Auto
goto :done

:auto_all
echo ==== AUTO DOWNLOAD→PREPROCESS→EXTRACT→TRAIN (%MODEL_NAME%) ====
%PYTHON% rvc\v2_core\auto_train.py --model_name "%MODEL_NAME%" --dataset_url "%DATASET_URL%" --batch_size %BATCH_SIZE% --gpu %GPU% --index_algorithm Auto
if errorlevel 1 goto :error
goto :done
if errorlevel 1 goto :error

:done
echo ==== DONE ====
echo - Logs:      %ROOT%logs\%MODEL_NAME%
echo - Index:     %ROOT%logs\%MODEL_NAME%\%MODEL_NAME%.index (자동 생성)
echo - Datasets:  %ROOT%%DATASET_PATH%
goto :eof

:error
echo 실패했습니다. 위 에러 로그를 확인하세요.
exit /b 1

endlocal

popd

