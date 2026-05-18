@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PROJECT_C=%ROOT%\project_c_intrinsic"
set "TESTS_DIR=%ROOT%\project_a\tests"
set "MODELS_FILE=%~dp0project_c_hf_models_20260515.txt"
set "PYTHON=%ROOT%\.venv_project_c\Scripts\python.exe"
set "BOOTSTRAP_PY=%ROOT%\.venv\Scripts\python.exe"

if not defined C_AUTO_INSTALL set "C_AUTO_INSTALL=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
set "HF_HOME=%ROOT%\.hf_cache"
set "HF_HUB_CACHE=%ROOT%\.hf_cache\hub"
set "HUGGINGFACE_HUB_CACHE=%ROOT%\.hf_cache\hub"
set "TRANSFORMERS_CACHE=%ROOT%\.hf_cache\transformers"
set "HF_XET_CACHE=%ROOT%\.hf_cache\xet"
set "C_OFFLOAD_ROOT=%ROOT%\.hf_offload"

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
set "CHECK_DIR=%ROOT%\runs\C_model_checks\%TS%"

echo.
echo ============================================================
echo  PROJECT C - HuggingFace model check
echo ============================================================
echo ROOT=%ROOT%
echo PROJECT_C=%PROJECT_C%
echo MODELS_FILE=%MODELS_FILE%
echo CHECK_DIR=%CHECK_DIR%
echo C_AUTO_INSTALL=%C_AUTO_INSTALL%
echo.

call :ensure_python_env
if errorlevel 1 goto fail

if not exist "%PROJECT_C%\check_hf_models.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_C%\check_hf_models.py
    goto fail
)

if not exist "%PROJECT_C%\intrinsic_evaluator.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_C%\intrinsic_evaluator.py
    goto fail
)

if not exist "%MODELS_FILE%" (
    echo [KLAIDA] Nerastas modeliu failas:
    echo %MODELS_FILE%
    goto fail
)

mkdir "%CHECK_DIR%" >nul 2>nul
mkdir "%HF_HOME%" >nul 2>nul
mkdir "%HF_HUB_CACHE%" >nul 2>nul
mkdir "%TRANSFORMERS_CACHE%" >nul 2>nul
mkdir "%HF_XET_CACHE%" >nul 2>nul
mkdir "%C_OFFLOAD_ROOT%" >nul 2>nul

"%PYTHON%" "%PROJECT_C%\check_hf_models.py" ^
    --models-file "%MODELS_FILE%" ^
    --evaluator "%PROJECT_C%\intrinsic_evaluator.py" ^
    --tests-dir "%TESTS_DIR%" ^
    --output-dir "%CHECK_DIR%" ^
    --python "%PYTHON%" ^
    --min-working 3 ^
    --top-n 3 ^
    --timeout 1800 ^
    --max-new-tokens 32 ^
    --max-memory-gpu 8GiB ^
    --max-memory-cpu 48GiB ^
    --offload-root "%C_OFFLOAD_ROOT%" ^
    --attn-implementation eager

if errorlevel 1 goto fail

echo.
echo ============================================================
echo  PROJECT C MODEL CHECK OK
echo ============================================================
echo Working top 3:
type "%CHECK_DIR%\working_models_top3.txt"
echo.
echo Visi rezultatai:
echo %CHECK_DIR%
goto end


:ensure_python_env
if exist "%PYTHON%" goto check_deps

if exist "%BOOTSTRAP_PY%" (
    "%BOOTSTRAP_PY%" -m venv "%ROOT%\.venv_project_c"
) else (
    py -3.12 -m venv "%ROOT%\.venv_project_c"
)
if errorlevel 1 (
    echo [KLAIDA] Nepavyko sukurti .venv_project_c
    exit /b 1
)

:check_deps
"%PYTHON%" -c "import torch, transformers, accelerate, pandas, numpy" >nul 2>nul
if not errorlevel 1 exit /b 0

if not "%C_AUTO_INSTALL%"=="1" (
    echo [KLAIDA] Project C priklausomybes neinstaliuotos.
    echo Paleisk:
    echo set C_AUTO_INSTALL=1
    echo check_project_C_hf_models_20260515.bat
    exit /b 1
)

echo.
echo === Installing Project C dependencies ===
"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

"%PYTHON%" -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 torch
if errorlevel 1 exit /b 1

"%PYTHON%" -m pip install --upgrade transformers accelerate bitsandbytes pandas numpy scipy matplotlib
if errorlevel 1 exit /b 1

"%PYTHON%" -c "import torch, transformers, accelerate, pandas, numpy; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
exit /b %ERRORLEVEL%


:fail
echo.
echo *** PROJECT C MODEL CHECK NEPAVYKO ***
echo Jei HuggingFace gated modelis nepraeina, prisijunk:
echo   huggingface-cli login
echo arba nustatyk:
echo   set HF_TOKEN=hf_...
exit /b 1

:end
endlocal
exit /b 0
