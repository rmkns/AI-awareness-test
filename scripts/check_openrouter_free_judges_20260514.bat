@echo off
setlocal EnableExtensions EnableDelayedExpansion

for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PROJECT_B=%ROOT%\project_b_judge"
if not defined PYTHON set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
set "MODELS_FILE=%~dp0openrouter_free_judges_20260514.txt"

for /f %%T in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%T"
set "CHECK_DIR=%ROOT%\runs\openrouter_model_checks\%TS%"

echo.
echo ============================================================
echo  OpenRouter free judge model check
echo ============================================================
echo ROOT=%ROOT%
echo MODELS_FILE=%MODELS_FILE%
echo CHECK_DIR=%CHECK_DIR%
echo.

if not exist "%PYTHON%" (
    echo [KLAIDA] Python nerastas:
    echo %PYTHON%
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

mkdir "%CHECK_DIR%" >nul 2>nul

"%PYTHON%" "%PROJECT_B%\check_openrouter_models.py" ^
    --models-file "%MODELS_FILE%" ^
    --output-dir "%CHECK_DIR%" ^
    --min-working 3 ^
    --top-n 3 ^
    --sleep 2 ^
    --timeout 90

if errorlevel 1 goto fail

echo.
echo ============================================================
echo  MODEL CHECK OK
echo ============================================================
echo Working top 3:
type "%CHECK_DIR%\working_models_top3.txt"
echo.
echo Visi rezultatai:
echo %CHECK_DIR%
goto end

:fail
echo.
echo *** MODEL CHECK NEPAVYKO ***
echo Jei matai 429, palauk ir paleisk is naujo.
exit /b 1

:end
endlocal
exit /b 0
