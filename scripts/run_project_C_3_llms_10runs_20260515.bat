@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PROJECT_C=%ROOT%\project_c_intrinsic"
set "TESTS_DIR=%ROOT%\project_a\tests"
set "MODELS_FILE=%~dp0project_c_hf_models_20260515.txt"
set "PYTHON=%ROOT%\.venv_project_c\Scripts\python.exe"
set "BOOTSTRAP_PY=%ROOT%\.venv\Scripts\python.exe"

if not defined C_AUTO_INSTALL set "C_AUTO_INSTALL=1"
if not defined C_RUNS set "C_RUNS=10"
if not defined C_TESTS_PER_MODEL set "C_TESTS_PER_MODEL=40"
set /a C_EXPECTED_PER_MODEL=%C_RUNS% * %C_TESTS_PER_MODEL%
if not defined C_MAX_NEW_TOKENS set "C_MAX_NEW_TOKENS=512"
if not defined C_TEMPERATURE set "C_TEMPERATURE=0.0"
if not defined C_KEEP_FULL_TOKENS set "C_KEEP_FULL_TOKENS=1"
if not defined C_MAX_RETRIES set "C_MAX_RETRIES=2"
if not defined C_RETRY_SLEEP set "C_RETRY_SLEEP=10"

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
if not defined C_MAX_MEMORY_GPU set "C_MAX_MEMORY_GPU=8GiB"
if not defined C_MAX_MEMORY_CPU set "C_MAX_MEMORY_CPU=48GiB"
if not defined C_ATTN_IMPLEMENTATION set "C_ATTN_IMPLEMENTATION=eager"

if defined RESUME_TS (
    set "TS=%RESUME_TS%"
) else (
    for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
)

set "CHECK_DIR=%ROOT%\runs\C_model_checks\%TS%"
set "RUN_DIR=%ROOT%\runs\C_full_3llms_10runs_%TS%"
set "C_DIR=%RUN_DIR%\C"
set "LOG_DIR=%C_DIR%\logs"
set "STATE_DIR=%C_DIR%\progress"
set "FAIL_DIR=%C_DIR%\failures"
set "ANALYSIS_ROOT=%C_DIR%\analysis"
set "MANIFEST=%RUN_DIR%\project_c_manifest.txt"

echo.
echo ============================================================
echo  PROJECT C FULL - 3 LLMs, 10 runs
echo ============================================================
echo ROOT=%ROOT%
echo RUN_DIR=%RUN_DIR%
echo C_RUNS=%C_RUNS%
echo C_EXPECTED_PER_MODEL=%C_EXPECTED_PER_MODEL%
echo C_TEMPERATURE=%C_TEMPERATURE%
echo C_KEEP_FULL_TOKENS=%C_KEEP_FULL_TOKENS%
echo C_MAX_MEMORY_GPU=%C_MAX_MEMORY_GPU%
echo C_AUTO_INSTALL=%C_AUTO_INSTALL%
echo.

call :ensure_python_env
if errorlevel 1 goto fail

if not exist "%PROJECT_C%\check_hf_models.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_C%\check_hf_models.py
    goto fail
)

if not exist "%PROJECT_C%\intrinsic_evaluator_resilient.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_C%\intrinsic_evaluator_resilient.py
    goto fail
)

if not exist "%PROJECT_C%\entropy_analyzer.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_C%\entropy_analyzer.py
    goto fail
)

mkdir "%C_DIR%" >nul 2>nul
mkdir "%LOG_DIR%" >nul 2>nul
mkdir "%STATE_DIR%" >nul 2>nul
mkdir "%FAIL_DIR%" >nul 2>nul
mkdir "%ANALYSIS_ROOT%" >nul 2>nul
mkdir "%HF_HOME%" >nul 2>nul
mkdir "%HF_HUB_CACHE%" >nul 2>nul
mkdir "%TRANSFORMERS_CACHE%" >nul 2>nul
mkdir "%HF_XET_CACHE%" >nul 2>nul
mkdir "%C_OFFLOAD_ROOT%" >nul 2>nul

echo.
echo === Step 1: Project C model check ===
mkdir "%CHECK_DIR%" >nul 2>nul

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
    --max-memory-gpu %C_MAX_MEMORY_GPU% ^
    --max-memory-cpu %C_MAX_MEMORY_CPU% ^
    --offload-root "%C_OFFLOAD_ROOT%" ^
    --attn-implementation %C_ATTN_IMPLEMENTATION%

if errorlevel 1 goto fail

set "WORKING_TOP3=%CHECK_DIR%\working_models_top3.txt"
copy "%WORKING_TOP3%" "%C_DIR%\working_project_c_models_top3.txt" >nul

(
echo PROJECT C FULL - 3 LLMs, 10 runs
echo Started=%DATE% %TIME%
echo ROOT=%ROOT%
echo PROJECT_C=%PROJECT_C%
echo TESTS_DIR=%TESTS_DIR%
echo RUN_DIR=%RUN_DIR%
echo CHECK_DIR=%CHECK_DIR%
echo WORKING_TOP3=%WORKING_TOP3%
echo C_RUNS=%C_RUNS%
echo C_EXPECTED_PER_MODEL=%C_EXPECTED_PER_MODEL%
echo C_MAX_NEW_TOKENS=%C_MAX_NEW_TOKENS%
echo C_TEMPERATURE=%C_TEMPERATURE%
echo C_KEEP_FULL_TOKENS=%C_KEEP_FULL_TOKENS%
echo C_MAX_MEMORY_GPU=%C_MAX_MEMORY_GPU%
echo C_MAX_MEMORY_CPU=%C_MAX_MEMORY_CPU%
echo C_OFFLOAD_ROOT=%C_OFFLOAD_ROOT%
echo C_ATTN_IMPLEMENTATION=%C_ATTN_IMPLEMENTATION%
) > "%MANIFEST%"

echo.
echo Naudojami Project C modeliai:
type "%WORKING_TOP3%"

echo.
echo === Step 2: Intrinsic runs ===

for /f "usebackq tokens=1-6 delims=|" %%A in ("%WORKING_TOP3%") do (
    call :run_model "%%A" "%%B" "%%C" "%%D" "%%E" "%%F"
)

echo.
echo Finished=%DATE% %TIME%>>"%MANIFEST%"
echo ============================================================
echo  PROJECT C FULL BAIGTA
echo ============================================================
echo Manifest: %MANIFEST%
echo Logs:     %LOG_DIR%
echo Progress: %STATE_DIR%
echo Failures: %FAIL_DIR%
echo Analysis: %ANALYSIS_ROOT%
goto end


:run_model
set "MODEL_LABEL=%~1"
set "MODEL_ID=%~2"
set "MODEL_QUANT=%~3"
set "MODEL_DTYPE=%~4"
set "MODEL_DEVICE=%~5"
set "MODEL_TRUST_REMOTE_CODE=%~6"
if not defined MODEL_TRUST_REMOTE_CODE set "MODEL_TRUST_REMOTE_CODE=0"

set "OUT_JSONL=%C_DIR%\%MODEL_LABEL%.jsonl"
set "FAIL_OUT=%FAIL_DIR%\%MODEL_LABEL%.failures.jsonl"
set "PROGRESS_OUT=%STATE_DIR%\%MODEL_LABEL%.progress.json"
set "MODEL_LOG=%LOG_DIR%\%MODEL_LABEL%.log"
set "MODEL_ANALYSIS=%ANALYSIS_ROOT%\%MODEL_LABEL%"
set "MODEL_OFFLOAD=%C_OFFLOAD_ROOT%\%MODEL_LABEL%"
mkdir "%MODEL_OFFLOAD%" >nul 2>nul

set "QUANT_ARGS="
if "%MODEL_QUANT%"=="4bit" set "QUANT_ARGS=--load-in-4bit"
if "%MODEL_QUANT%"=="8bit" set "QUANT_ARGS=--load-in-8bit"

set "TOKEN_ARGS="
if "%C_KEEP_FULL_TOKENS%"=="1" set "TOKEN_ARGS=--keep-full-tokens"

set "TRUST_ARGS="
if "%MODEL_TRUST_REMOTE_CODE%"=="1" set "TRUST_ARGS=--trust-remote-code"
if "%MODEL_TRUST_REMOTE_CODE%"=="true" set "TRUST_ARGS=--trust-remote-code"
if "%MODEL_TRUST_REMOTE_CODE%"=="yes" set "TRUST_ARGS=--trust-remote-code"

echo.
echo ------------------------------------------------------------
echo Model: %MODEL_LABEL% = %MODEL_ID%
echo Quant: %MODEL_QUANT%, dtype=%MODEL_DTYPE%, device=%MODEL_DEVICE%
echo Output: %OUT_JSONL%
echo Log:    %MODEL_LOG%
echo ------------------------------------------------------------

(
echo ------------------------------------------------------------
echo Started=%DATE% %TIME%
echo MODEL_LABEL=%MODEL_LABEL%
echo MODEL_ID=%MODEL_ID%
echo MODEL_QUANT=%MODEL_QUANT%
echo MODEL_DTYPE=%MODEL_DTYPE%
echo MODEL_DEVICE=%MODEL_DEVICE%
echo MODEL_TRUST_REMOTE_CODE=%MODEL_TRUST_REMOTE_CODE%
echo OUT_JSONL=%OUT_JSONL%
echo FAIL_OUT=%FAIL_OUT%
echo PROGRESS_OUT=%PROGRESS_OUT%
echo MODEL_ANALYSIS=%MODEL_ANALYSIS%
echo MODEL_OFFLOAD=%MODEL_OFFLOAD%
echo ------------------------------------------------------------
) > "%MODEL_LOG%"

"%PYTHON%" "%PROJECT_C%\intrinsic_evaluator_resilient.py" ^
    --model "%MODEL_ID%" ^
    --model-label "%MODEL_LABEL%" ^
    --tests-dir "%TESTS_DIR%" ^
    --output "%OUT_JSONL%" ^
    --runs %C_RUNS% ^
    --max-new-tokens %C_MAX_NEW_TOKENS% ^
    --temperature %C_TEMPERATURE% ^
    --device %MODEL_DEVICE% ^
    --dtype %MODEL_DTYPE% ^
    --failure-output "%FAIL_OUT%" ^
    --progress-output "%PROGRESS_OUT%" ^
    --require-items %C_EXPECTED_PER_MODEL% ^
    --max-memory-gpu %C_MAX_MEMORY_GPU% ^
    --max-memory-cpu %C_MAX_MEMORY_CPU% ^
    --offload-folder "%MODEL_OFFLOAD%" ^
    --attn-implementation %C_ATTN_IMPLEMENTATION% ^
    --max-retries %C_MAX_RETRIES% ^
    --retry-sleep %C_RETRY_SLEEP% ^
    %TOKEN_ARGS% ^
    %TRUST_ARGS% ^
    %QUANT_ARGS% ^
    >> "%MODEL_LOG%" 2>&1

set "RC=!ERRORLEVEL!"
echo RC=!RC! model=%MODEL_LABEL% output=%OUT_JSONL% log=%MODEL_LOG%>>"%MANIFEST%"

if not "!RC!"=="0" (
    echo [WARN] intrinsic run baigesi su RC=!RC!: %MODEL_LABEL%
    echo        Ziurek log: %MODEL_LOG%
)

if exist "%OUT_JSONL%" (
    "%PYTHON%" "%PROJECT_C%\entropy_analyzer.py" ^
        --intrinsic "%OUT_JSONL%" ^
        --output-dir "%MODEL_ANALYSIS%" ^
        >> "%MODEL_LOG%" 2>&1
    echo ANALYSIS_RC=!ERRORLEVEL! model=%MODEL_LABEL% analysis=%MODEL_ANALYSIS%>>"%MANIFEST%"
)
exit /b 0


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
    echo run_project_C_3_llms_10runs_20260515.bat
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
echo *** PROJECT C FULL NEPAVYKO ***
echo RUN_DIR:
echo %RUN_DIR%
echo.
echo Jei HuggingFace gated modelis nepraeina, prisijunk:
echo   huggingface-cli login
echo arba nustatyk:
echo   set HF_TOKEN=hf_...
exit /b 1

:end
endlocal
exit /b 0
