@echo off
chcp 65001 >nul
title sayoko-ui-launcher
echo ============================================
echo  サヨ子とお話し — 起動中
echo ============================================
echo [1/3] g24 でサーバを起動し、ポート転送を張ります...
start "sayoko-server (閉じると終了)" ssh -L 8998:localhost:8998 g24 "bash ~/sayoko-fullduplex/ui/server/serve_ui.sh"

echo [2/3] モデル読込を待っています (60秒ほどかかります)...
:wait
timeout /t 5 /nobreak >nul
powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing http://localhost:8998/healthz -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if errorlevel 1 goto wait

echo [3/3] ブラウザを開きます (Chrome 推奨)
rem OPENAI_KEY を .env から読み、URLフラグメント (#k=) でブラウザにだけ渡す。
rem フラグメントは HTTP リクエストに含まれないため、キーがサーバへ送られることはない。
set "KEY="
for %%F in ("%~dp0.env" "%~dp0..\.env" "%USERPROFILE%\Desktop\otamesi\Va-chan\.env" "%USERPROFILE%\Desktop\Va-chan\.env") do (
  if not defined KEY if exist %%F (
    for /f "usebackq tokens=1,* delims==" %%a in (%%F) do (
      if /i "%%a"=="OPENAI_KEY" set "KEY=%%b"
      if /i "%%a"=="OPENAI_API_KEY" set "KEY=%%b"
    )
  )
)
if defined KEY (
  start "" "http://localhost:8998/#k=%KEY%"
) else (
  echo   ※ .env に OPENAI_KEY が見つかりません。GPT-live は画面でキーを入力すると有効になります。
  start "" "http://localhost:8998/"
)
echo.
echo 起動しました。終了するときは "sayoko-server" のウィンドウを閉じてください。
timeout /t 8 >nul
