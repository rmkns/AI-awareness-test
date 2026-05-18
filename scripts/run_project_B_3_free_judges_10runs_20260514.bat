@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PROJECT_B=%ROOT%\project_b_judge"
if not defined PYTHON set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "MODELS_FILE=%~dp0openrouter_free_judges_20260514.txt"

if not defined B_RUNS set "B_RUNS=10"
if not defined B_TESTS_PER_MODEL set "B_TESTS_PER_MODEL=40"
set /a B_EXPECTED_PER_SOURCE=%B_RUNS% * %B_TESTS_PER_MODEL%
if not defined B_SLEEP set "B_SLEEP=5"
if not defined B_MAX_RETRIES set "B_MAX_RETRIES=12"
if not defined B_RETRY_BASE set "B_RETRY_BASE=30"
if not defined B_RETRY_MAX set "B_RETRY_MAX=180"
if not defined B_STOP_AFTER_CONSECUTIVE_FAILURES set "B_STOP_AFTER_CONSECUTIVE_FAILURES=15"

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
set "CHECK_DIR=%ROOT%\runs\openrouter_model_checks\%TS%"
set "TEMP_A_ENV=%TEMP%\project_b_a_run_%TS%.bat"
set "TEMP_A_JSON=%ROOT%\runs\project_b_a_inventory_%TS%.json"

echo.
echo ============================================================
echo  PROJECT B FULL - 3 free judges, 10 runs
echo ============================================================
echo ROOT=%ROOT%
echo B_RUNS=%B_RUNS%
echo B_EXPECTED_PER_SOURCE=%B_EXPECTED_PER_SOURCE%
echo.

if not exist "%PYTHON%" (
    echo [KLAIDA] Python nerastas:
    echo %PYTHON%
    goto fail
)

if not exist "%PROJECT_B%\evaluate_llm_judge_resilient.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_B%\evaluate_llm_judge_resilient.py
    goto fail
)

if not exist "%PROJECT_B%\verify_project_a_run.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_B%\verify_project_a_run.py
    goto fail
)

if not exist "%PROJECT_B%\check_openrouter_models.py" (
    echo [KLAIDA] Nerastas:
    echo %PROJECT_B%\check_openrouter_models.py
    goto fail
)

if not exist "%MODELS_FILE%" (
    echo [KLAIDA] Nerastas modeliu failas:
    echo %MODELS_FILE%
    goto fail
)

if not defined OPENROUTER_API_KEY (
    if exist "%~dp0_apikey.bat" (
        call "%~dp0_apikey.bat"
    )
)
if not defined OPENROUTER_API_KEY (
    echo [KLAIDA] OPENROUTER_API_KEY nenustatytas ir nerasta:
    echo "%~dp0_apikey.bat"
    goto fail
)

echo.
echo === Step 1: OpenRouter modeliu patikra ===
mkdir "%CHECK_DIR%" >nul 2>nul

"%PYTHON%" "%PROJECT_B%\check_openrouter_models.py" ^
    --models-file "%MODELS_FILE%" ^
    --output-dir "%CHECK_DIR%" ^
    --min-working 3 ^
    --top-n 3 ^
    --sleep 2 ^
    --timeout 90

if errorlevel 1 (
    echo [KLAIDA] Maziau nei 3 veikiantys nemokami judge modeliai.
    echo Patikros katalogas:
    echo %CHECK_DIR%
    goto fail
)

set "WORKING_TOP3=%CHECK_DIR%\working_models_top3.txt"
if not exist "%WORKING_TOP3%" (
    echo [KLAIDA] Nerastas:
    echo %WORKING_TOP3%
    goto fail
)

echo.
echo Naudojami judge modeliai:
type "%WORKING_TOP3%"

echo.
echo === Step 2: Project A 10-run rinkinio patikra ===

if defined A_RUN_DIR (
    "%PYTHON%" "%PROJECT_B%\verify_project_a_run.py" ^
        --a-run-dir "%A_RUN_DIR%" ^
        --runs %B_RUNS% ^
        --tests-per-model %B_TESTS_PER_MODEL% ^
        --output-json "%TEMP_A_JSON%" ^
        --write-env "%TEMP_A_ENV%"
) else (
    "%PYTHON%" "%PROJECT_B%\verify_project_a_run.py" ^
        --runs-root "%ROOT%\runs" ^
        --runs %B_RUNS% ^
        --tests-per-model %B_TESTS_PER_MODEL% ^
        --output-json "%TEMP_A_JSON%" ^
        --write-env "%TEMP_A_ENV%"
)

if errorlevel 1 (
    echo.
    echo [KLAIDA] Nerastas pilnas Project A 10-run rinkinys.
    echo Reikia: 5 modeliai * 40 testu * 10 runu = 2000 A atsakymu.
    echo Inventorizacija:
    echo %TEMP_A_JSON%
    goto fail
)

call "%TEMP_A_ENV%"

set "B_ROOT=%A_RUN_DIR%\B"
set "LOG_DIR=%B_ROOT%\logs_3free_10runs\%TS%"
set "STATE_DIR=%B_ROOT%\progress_3free_10runs\%TS%"
set "FAIL_DIR=%B_ROOT%\failures_3free_10runs\%TS%"
set "MANIFEST=%B_ROOT%\project_B_3free_10runs_manifest_%TS%.txt"

mkdir "%LOG_DIR%" >nul 2>nul
mkdir "%STATE_DIR%" >nul 2>nul
mkdir "%FAIL_DIR%" >nul 2>nul

copy "%WORKING_TOP3%" "%B_ROOT%\working_judges_%TS%.txt" >nul

(
echo PROJECT B FULL - 3 free judges, 10 runs
echo Started=%DATE% %TIME%
echo ROOT=%ROOT%
echo A_RUN_DIR=%A_RUN_DIR%
echo CHECK_DIR=%CHECK_DIR%
echo WORKING_TOP3=%WORKING_TOP3%
echo LOG_DIR=%LOG_DIR%
echo STATE_DIR=%STATE_DIR%
echo FAIL_DIR=%FAIL_DIR%
echo B_RUNS=%B_RUNS%
echo B_EXPECTED_PER_SOURCE=%B_EXPECTED_PER_SOURCE%
) > "%MANIFEST%"

echo.
echo === Step 3: Project B vertinimas ===
echo A_RUN_DIR=%A_RUN_DIR%
echo MANIFEST=%MANIFEST%
echo LOG_DIR=%LOG_DIR%
echo.

for /f "usebackq tokens=1* delims=|" %%L in ("%WORKING_TOP3%") do (
    call :run_judge "%%L" "%%M"
)

echo.
echo Finished=%DATE% %TIME%>>"%MANIFEST%"
echo ============================================================
echo  PROJECT B FULL PALEIDIMAS BAIGTAS
echo ============================================================
echo Manifest: %MANIFEST%
echo Logs:     %LOG_DIR%
echo Progress: %STATE_DIR%
echo Failures: %FAIL_DIR%
echo.
goto end


:run_judge
set "JUDGE_LABEL=%~1"
set "JUDGE_MODEL=%~2"

echo.
echo ------------------------------------------------------------
echo Judge: %JUDGE_LABEL% = %JUDGE_MODEL%
echo ------------------------------------------------------------
echo Judge=%JUDGE_LABEL% ^| %JUDGE_MODEL%>>"%MANIFEST%"

call :run_source "gemma3"
call :run_source "llama32"
call :run_source "gptoss"
call :run_source "deepseek"
call :run_source "qwen3"
exit /b 0


:run_source
set "SRC=%~1"
set "SESSION_VAR=SESSION_%SRC%"
call set "INPUT_DIR=%%%SESSION_VAR%%%"

set "OUT_DIR=%B_ROOT%\judges\%JUDGE_LABEL%"
set "B_OUT=%OUT_DIR%\%SRC%.jsonl"
set "FAIL_OUT=%FAIL_DIR%\%JUDGE_LABEL%_%SRC%.failures.jsonl"
set "PROGRESS_OUT=%STATE_DIR%\%JUDGE_LABEL%_%SRC%.progress.json"
set "SRC_LOG=%LOG_DIR%\%JUDGE_LABEL%_%SRC%.log"

mkdir "%OUT_DIR%" >nul 2>nul

echo.
echo --- B: judge=%JUDGE_LABEL% source=%SRC% ---
echo input=%INPUT_DIR%
echo output=%B_OUT%
echo log=%SRC_LOG%

(
echo ------------------------------------------------------------
echo Started=%DATE% %TIME%
echo Judge=%JUDGE_LABEL%
echo JudgeModel=%JUDGE_MODEL%
echo Source=%SRC%
echo Input=%INPUT_DIR%
echo Output=%B_OUT%
echo FailureOutput=%FAIL_OUT%
echo ProgressOutput=%PROGRESS_OUT%
echo ------------------------------------------------------------
) > "%SRC_LOG%"

"%PYTHON%" "%PROJECT_B%\evaluate_llm_judge_resilient.py" ^
    --input-dir "%INPUT_DIR%" ^
    --output "%B_OUT%" ^
    --judge-model "%JUDGE_MODEL%" ^
    --api-base "https://openrouter.ai/api/v1" ^
    --api-key "%OPENROUTER_API_KEY%" ^
    --runs %B_RUNS% ^
    --require-items %B_EXPECTED_PER_SOURCE% ^
    --failure-output "%FAIL_OUT%" ^
    --progress-output "%PROGRESS_OUT%" ^
    --max-retries %B_MAX_RETRIES% ^
    --retry-base %B_RETRY_BASE% ^
    --retry-max %B_RETRY_MAX% ^
    --sleep %B_SLEEP% ^
    --stop-after-consecutive-failures %B_STOP_AFTER_CONSECUTIVE_FAILURES% ^
    >> "%SRC_LOG%" 2>&1

set "RC=!ERRORLEVEL!"
echo RC=!RC! judge=%JUDGE_LABEL% source=%SRC% output=%B_OUT% log=%SRC_LOG%>>"%MANIFEST%"

if not "!RC!"=="0" (
    echo [WARN] judge=%JUDGE_LABEL% source=%SRC% baigesi su RC=!RC!
    echo        Ziurek log: %SRC_LOG%
)
exit /b 0


:fail
echo.
echo *** PROJECT B FULL NEPAVYKO ***
exit /b 1

:end
endlocal
exit /b 0
