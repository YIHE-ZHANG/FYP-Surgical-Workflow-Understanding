@echo off
setlocal

rem Check if config file is provided
set config=%1
if not exist %config% (
    echo Need a config file
    exit /b
)

rem Determine script directory
set "script_dir=%~dp0"

rem Extract configuration values
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
echo Arch: %arch_dir%
echo Dataset: %dataset%

rem Format current date and time as YYYYMMDD_HHMMSS
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c%%a%%b)
for /f "tokens=1-2 delims=: " %%a in ('time /t') do (set mytime=%%a%%b)
set now=%mydate%_%mytime%

rem Create output directory
if not exist "exp\%type%\%arch_dir%\%dataset%\%now%" mkdir "exp\%type%\%arch_dir%\%dataset%\%now%"

rem Run the test
echo Running test script...
python test.py --config "%config%" --log_time %now% --dataset %dataset% > "exp\%type%\%arch_dir%\%dataset%\%now%\%now%.log" 2>&1

rem Check if test was successful
if %ERRORLEVEL% EQU 0 (
    echo Test completed successfully!
    echo Log file: exp\%type%\%arch_dir%\%dataset%\%now%\%now%.log
    
    rem Run analysis script if it exists
    if exist analyze_rarp50_results.py (
        echo Analyzing results...
        python analyze_rarp50_results.py
    )
) else (
    echo Test failed. Check the log file for details.
)

endlocal