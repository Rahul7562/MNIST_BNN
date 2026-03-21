@echo off
REM Sync mem_files to all Vivado simulation directories
echo Syncing mem_files to simulation directories...

xcopy /Y /I "mem_files\*" "mnist_verilog\mem_files\"
xcopy /Y /I "mem_files\*" "mnist_verilog\project_1\project_1.sim\sim_1\behav\xsim\mem_files\"

echo Done! mem_files synced to:
echo   - mnist_verilog\mem_files\
echo   - mnist_verilog\project_1\project_1.sim\sim_1\behav\xsim\mem_files\
pause
