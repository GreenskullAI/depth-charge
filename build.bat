@echo off
setlocal
cd /d "%~dp0"
call "..\DLSS5-Feeder-main\tools\vcvars.bat" x64 || exit /b 1
if not exist build mkdir build
cl /nologo /LD /EHsc /O2 /MD /W3 /std:c++20 /I"..\DLSS5-Feeder-main\external\reshade\include" /I"..\DLSS5-Feeder-main\external\imgui" /Fobuild\ /Fdbuild\ depth-charge.cpp /link /OUT:build\depth-charge.addon64 kernel32.lib user32.lib d3dcompiler.lib
if errorlevel 1 (echo BUILD FAILED & exit /b 1)
echo.
echo Built build\depth-charge.addon64
endlocal
