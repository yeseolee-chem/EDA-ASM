@echo off
REM =====================================================================
REM Windows batch script to run Gaussian AM1 SPE on 1015 .gjf files.
REM
REM Prereq:
REM   - Gaussian 16 (or 09) installed and on PATH (`g16` or `g09` command)
REM   - CD to the paper_reproduction_203/ root before running
REM
REM Usage:
REM   scripts\run_all_am1_windows.bat
REM
REM Output:
REM   gaussian_inputs\<type>\<name>.log  (Gaussian output)
REM   gaussian_inputs\<type>\<name>.chk  (checkpoint, auto-produced)
REM
REM Runtime: ~30 min - 4 h depending on machine.
REM =====================================================================

setlocal enabledelayedexpansion
set GAUSS_CMD=g16
where %GAUSS_CMD% >nul 2>&1
if errorlevel 1 (
    set GAUSS_CMD=g09
    where %GAUSS_CMD% >nul 2>&1
    if errorlevel 1 (
        echo ERROR: neither g16 nor g09 found on PATH. Install Gaussian.
        exit /b 1
    )
)
echo Using Gaussian command: %GAUSS_CMD%
echo.

set N_TOTAL=1015
set N_DONE=0
set N_FAIL=0

for %%T in (ts gs dist_gs) do (
    echo === Running %%T jobs ===
    cd gaussian_inputs\%%T
    for %%F in (*.gjf) do (
        set /a N_DONE+=1
        set STEM=%%~nF
        if exist "!STEM!.log" (
            findstr /C:"Normal termination" "!STEM!.log" >nul 2>&1
            if !errorlevel! equ 0 (
                echo   [SKIP] !STEM!.log already done ^(!N_DONE!/%N_TOTAL%^)
            ) else (
                echo   [RUN]  !STEM! ^(!N_DONE!/%N_TOTAL%^)
                %GAUSS_CMD% "%%F"
                if errorlevel 1 set /a N_FAIL+=1
            )
        ) else (
            echo   [RUN]  !STEM! ^(!N_DONE!/%N_TOTAL%^)
            %GAUSS_CMD% "%%F"
            if errorlevel 1 set /a N_FAIL+=1
        )
    )
    cd ..\..
)

echo.
echo === DONE: !N_DONE! attempted, !N_FAIL! failures ===
echo Check with: python scripts\verify_am1_results.py
endlocal
