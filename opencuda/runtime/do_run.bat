@echo off
cd /d "C:\users\kraken\opencuda\opencuda\runtime"
C:\users\kraken\opencuda\opencuda\runtime\opencuda_runtime.exe > C:\users\kraken\opencuda\opencuda\runtime\run_out.txt 2>&1
echo EXIT=%ERRORLEVEL% >> C:\users\kraken\opencuda\opencuda\runtime\run_out.txt
