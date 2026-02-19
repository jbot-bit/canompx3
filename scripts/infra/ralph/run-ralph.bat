@echo off
REM Ralph Wiggum Overnight Audit — double-click to run
cd /d "C:\Users\joshd\canompx3"
"C:\Program Files\Git\bin\bash.exe" -l -c "cd /c/Users/joshd/canompx3 && export PATH=\"$HOME/.local/bin:$PATH\" && unset CLAUDECODE && bash scripts/infra/ralph.sh 15 2>&1 | tee ralph-output.log"
pause
