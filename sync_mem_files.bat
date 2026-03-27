@echo off
REM ============================================================================
REM Sync Memory Files for BNN MNIST FPGA Project
REM
REM Copies weight and test image files from mem_files/ to Vivado simulation
REM directories for XSim.
REM ============================================================================

echo.
echo ========================================
echo BNN MNIST Memory File Synchronization
echo ========================================
echo.

set SRC_DIR=mem_files
set DST1=mnist_verilog\mem_files
set DST2=mnist_verilog\project_1\project_1.sim\sim_1\behav\xsim\mem_files

REM Check source directory exists
if not exist "%SRC_DIR%" (
    echo ERROR: Source directory '%SRC_DIR%' not found!
    echo Run mnist_bnn_v2.py first to generate weight files.
    exit /b 1
)

REM Create destination directories
echo Creating destination directories...
if not exist "%DST1%" mkdir "%DST1%"
if not exist "%DST2%" mkdir "%DST2%"

REM Copy all files
echo Copying memory files...
xcopy /Y /Q "%SRC_DIR%\*" "%DST1%\" >nul
xcopy /Y /Q "%SRC_DIR%\*" "%DST2%\" >nul

REM Count files
set /a count=0
for %%f in ("%DST1%\*.mem") do set /a count+=1

echo.
echo ========================================
echo Synchronization Complete
echo ========================================
echo   Files copied: %count%
echo   Destination 1: %DST1%
echo   Destination 2: %DST2%
echo.
echo Next steps:
echo   1. Open Vivado project (mnist_verilog\project_1\project_1.xpr)
echo   2. Run behavioral simulation
echo   3. Check tb_bnn_system testbench results
echo.

pause
