@echo off
echo TEST_RAN_%TIME% > C:\users\kraken\opencuda\opencuda\runtime\test_build_out.txt
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" >> C:\users\kraken\opencuda\opencuda\runtime\test_build_out.txt 2>&1
cd /d "C:\users\kraken\opencuda\opencuda\runtime"
echo CWD=%CD% >> C:\users\kraken\opencuda\opencuda\runtime\test_build_out.txt
where cl.exe >> C:\users\kraken\opencuda\opencuda\runtime\test_build_out.txt 2>&1
cl /W3 /O2 windows_d3dkmt.c /Fe:opencuda_runtime.exe >> C:\users\kraken\opencuda\opencuda\runtime\test_build_out.txt 2>&1
echo EXIT=%ERRORLEVEL% >> C:\users\kraken\opencuda\opencuda\runtime\test_build_out.txt
