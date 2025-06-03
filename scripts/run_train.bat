@REM @echo off
@REM setlocal


@REM set config=%1
@REM if not exist %config% (
@REM     echo Need a config file
@REM     exit /b
@REM )

@REM rem Determine the directory where this batch file is located
@REM set script_dir=%~dp0

@REM rem Use the helper script to extract values - with full path
@REM for /f "usebackq tokens=*" %%a in (`python "%script_dir%extract_config.py" "%config%"`) do (
@REM     if not defined type (
@REM         set type=%%a
@REM     ) else if not defined arch (
@REM         set arch=%%a
@REM     ) else if not defined dataset (
@REM         set dataset=%%a
@REM     )
@REM )

@REM rem Replace forward slash with hyphen for directory path
@REM set "arch_dir=%arch:/=-%"

@REM echo Type: %type%
@REM echo Arch: %arch%
@REM echo Arch Dir: %arch_dir%
@REM echo Dataset: %dataset%

@REM rem Format current date and time as YYYYMMDD_HHMMSS
@REM for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
@REM for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set mytime=%%a%%b)
@REM set now=%mydate%_%mytime%

@REM echo Creating directory structure: exp\%type%\%arch_dir%\%dataset%\%now%

@REM rem Create directory structure
@REM if not exist "exp\%type%\%arch_dir%\%dataset%\%now%" mkdir "exp\%type%\%arch_dir%\%dataset%\%now%"

@REM echo Running training script
@REM python train.py --config "%config%" --log_time %now% > "exp\%type%\%arch_dir%\%dataset%\%now%\%now%.log" 2>&1

@REM endlocal

@REM New Batch file updated
@echo off
REM Corrected batch file with proper ViT-B\16 path format
setlocal enabledelayedexpansion

if "%1"=="" (
    echo Usage: scripts\run_train.bat .\configs\rarp50\rarp50_ft.yaml [resume] ["path\to\checkpoint.pt"]
    pause
    exit /b 1
)

set CONFIG_FILE=%1

if "%2"=="list" (
    echo ========================================
    echo AVAILABLE CHECKPOINTS
    echo ========================================
    set FOUND_ANY=0
    
    REM Check the correct path format: ViT-B\16
    if exist ".\exp\clip_surgery\ViT-B\16\rarp50" (
        echo Searching in: .\exp\clip_surgery\ViT-B\16\rarp50
        for /d %%d in (".\exp\clip_surgery\ViT-B\16\rarp50\*") do (
            echo Checking directory: %%d
            if exist "%%d\*.pt" (
                echo   Found .pt files in: %%d
                for %%f in ("%%d\*.pt") do (
                    echo     %%f
                )
                set FOUND_ANY=1
            ) else (
                echo   No .pt files found in: %%d
            )
        )
    ) else (
        echo Directory does not exist: .\exp\clip_surgery\ViT-B\16\rarp50
    )
    
    if !FOUND_ANY!==0 (
        echo No .pt checkpoint files found!
    )
    
    echo ========================================
    pause
    exit /b 0
)

REM Generate timestamp
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set DATE_STR=%%c%%a%%b
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set TIME_STR=%%a%%b
set TIMESTAMP=%DATE_STR%_%TIME_STR%
set TIMESTAMP=%TIMESTAMP: =%
set TIMESTAMP=%TIMESTAMP::=%

REM Correct experiment directory path
set EXP_DIR=.\exp\clip_surgery\ViT-B\16\rarp50\%TIMESTAMP%

if "%2"=="resume" (
    echo ========================================
    echo RESUME MODE
    echo ========================================
    
    if NOT "%3"=="" (
        set MANUAL_CHECKPOINT=%3
        set MANUAL_CHECKPOINT=!MANUAL_CHECKPOINT:"=!
        
        if exist "!MANUAL_CHECKPOINT!" (
            echo Using: !MANUAL_CHECKPOINT!
            set RESUME_ARG=--resume "!MANUAL_CHECKPOINT!"
        ) else (
            echo Error: Checkpoint not found: !MANUAL_CHECKPOINT!
            pause
            exit /b 1
        )
    ) else (
        echo Error: No checkpoint specified for resume!
        echo Run: scripts\run_train.bat %CONFIG_FILE% list
        echo Then: scripts\run_train.bat %CONFIG_FILE% resume "path\to\checkpoint.pt"
        pause
        exit /b 1
    )
) else (
    echo ========================================
    echo FRESH TRAINING
    echo ========================================
    set RESUME_ARG=
)

echo Config: %CONFIG_FILE%
echo Working Dir: %EXP_DIR%
echo Resume: %RESUME_ARG%
echo ========================================

call conda activate bridge-prompt

if not exist "%EXP_DIR%" mkdir "%EXP_DIR%"

python train.py --config "%CONFIG_FILE%" --log_time "%TIMESTAMP%" %RESUME_ARG%

pause