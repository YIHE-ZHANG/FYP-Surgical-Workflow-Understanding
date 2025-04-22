@echo off
setlocal


set config=%1
if not exist %config% (
    echo Need a config file
    exit /b
)

rem Determine the directory where this batch file is located
set script_dir=%~dp0

rem Use the helper script to extract values - with full path
for /f "usebackq tokens=*" %%a in (`python "%script_dir%extract_config.py" "%config%"`) do (
    if not defined type (
        set type=%%a
    ) else if not defined arch (
        set arch=%%a
    ) else if not defined dataset (
        set dataset=%%a
    )
)

rem Replace forward slash with hyphen for directory path
set "arch_dir=%arch:/=-%"

echo Type: %type%
echo Arch: %arch%
echo Arch Dir: %arch_dir%
echo Dataset: %dataset%

rem Format current date and time as YYYYMMDD_HHMMSS
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set mytime=%%a%%b)
set now=%mydate%_%mytime%

echo Creating directory structure: exp\%type%\%arch_dir%\%dataset%\%now%

rem Create directory structure
if not exist "exp\%type%\%arch_dir%\%dataset%\%now%" mkdir "exp\%type%\%arch_dir%\%dataset%\%now%"

echo Running training script
python train.py --config "%config%" --log_time %now% > "exp\%type%\%arch_dir%\%dataset%\%now%\%now%.log" 2>&1

endlocal