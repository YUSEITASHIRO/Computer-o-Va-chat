@echo off
setlocal enabledelayedexpansion
title sayoko-ui-launcher
rem ---------------------------------------------------------------------------
rem  NOTE: This launcher is intentionally ASCII-only, CRLF, and calls
rem  timeout.exe by full path.
rem   - cmd.exe parses .bat with the system ANSI codepage (CP932 here), so
rem     UTF-8 Japanese inside a .bat corrupts parsing ("chcp 65001" does not
rem     help; it only affects console output).
rem   - LF-only line endings break label/goto handling.
rem   - "timeout.exe" refuses to run when stdin is redirected, and a bare
rem     "timeout" can resolve to GNU timeout if Git Bash is on PATH, so we
rem     sleep with ping instead (full path, no stdin dependency).
rem ---------------------------------------------------------------------------
set "PING=%SystemRoot%\System32\ping.exe"

echo ============================================
echo   Sayoko voice chat - starting up
echo ============================================
echo.

where ssh >nul 2>&1
if errorlevel 1 (
  echo [ERROR] "ssh" not found. Enable the Windows OpenSSH client.
  goto :failed
)

rem --- Read OPENAI_KEY from .env -------------------------------------------
rem The key is handed to the browser through the URL fragment (#k=). Fragments
rem are never sent in HTTP requests, so the key never reaches the chat server.
set "KEY="
for %%F in (
  "%~dp0.env"
  "%~dp0..\.env"
  "%USERPROFILE%\Desktop\otamesi\Va-chan\.env"
  "%USERPROFILE%\Desktop\Va-chan\.env"
) do (
  if not defined KEY if exist "%%~F" (
    for /f "usebackq tokens=1,* delims==" %%a in ("%%~F") do (
      if /i "%%a"=="OPENAI_KEY" set "KEY=%%b"
      if /i "%%a"=="OPENAI_API_KEY" set "KEY=%%b"
    )
    if defined KEY echo [info] API key loaded from %%~F
  )
)
if not defined KEY echo [info] No OPENAI_KEY in .env - LLM injection disabled.

echo [1/3] Starting server on g24 and forwarding port 8998 ...
start "sayoko-server (close to quit)" ssh -L 8998:localhost:8998 g24 "bash ~/sayoko-fullduplex/ui/server/serve_ui.sh"

echo [2/3] Waiting for the model to load (60-90 seconds) ...
set /a TRY=0
:wait
"%PING%" -n 6 127.0.0.1 >nul
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://localhost:8998/healthz -TimeoutSec 3; exit ($r.StatusCode -ne 200) } catch { exit 1 }"
if not errorlevel 1 goto :ready
set /a TRY+=1
if !TRY! LSS 36 (
  echo   ... waiting !TRY!/36
  goto :wait
)
echo [ERROR] Server did not come up. Check the "sayoko-server" window.
goto :failed

:ready
echo [3/3] Opening the browser (Chrome recommended)
if defined KEY (
  start "" "http://localhost:8998/#k=!KEY!"
) else (
  start "" "http://localhost:8998/"
)
echo.
echo Ready. Close the browser tab to finish (the server shuts down by itself).
"%PING%" -n 11 127.0.0.1 >nul
exit /b 0

:failed
echo.
pause
exit /b 1
