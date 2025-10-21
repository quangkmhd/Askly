@echo off
REM ============================================================================
REM STOP ALL SERVICES - WINDOWS
REM ============================================================================

echo.
echo 🛑 Stopping Askly services...
echo.

REM Kill Python processes (Backend)
echo Stopping Backend...
taskkill /F /IM python.exe /T >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Backend stopped
) else (
    echo ⚠️  No Backend process running
)

REM Kill Node processes (Frontend)  
echo Stopping Frontend...
taskkill /F /IM node.exe /T >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Frontend stopped
) else (
    echo ⚠️  No Frontend process running
)

echo.
echo ✅ All services stopped
echo.
pause

