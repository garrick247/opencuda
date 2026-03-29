@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" > nul 2>&1
cd /d "C:\users\kraken\opencuda\opencuda\runtime"
cl /W3 /O2 windows_d3dkmt.c /Fe:opencuda_runtime.exe > C:\users\kraken\opencuda\opencuda\runtime\build_out.txt 2>&1
echo EXIT=%ERRORLEVEL% >> C:\users\kraken\opencuda\opencuda\runtime\build_out.txt
