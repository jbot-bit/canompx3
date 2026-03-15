@echo off
REM Ralph Wiggum Overnight Audit — double-click or run from cmd
REM Uses the real Seven Sins auditor (ralph_headless.sh)
REM
REM Usage:
REM   run-ralph.bat          (default: 5 iterations)
REM   run-ralph.bat 10       (10 iterations)
REM   run-ralph.bat 3        (quick 3-iteration run)

set ITERS=%1
if "%ITERS%"=="" set ITERS=5

cd /d "C:\Users\joshd\canompx3"
echo Ralph Headless — %ITERS% iterations
"C:\Program Files\Git\bin\bash.exe" -l -c "cd /c/Users/joshd/canompx3 && export PATH=\"$HOME/.local/bin:$PATH\" && unset CLAUDECODE && bash scripts/tools/ralph_headless.sh %ITERS% 2>&1 | tee ralph-output.log"
pause
