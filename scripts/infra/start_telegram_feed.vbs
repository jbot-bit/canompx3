' Silent launcher for telegram_feed.py — no console window
' Placed in Windows Startup folder to auto-run on login

Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw C:\Users\joshd\canompx3\scripts\infra\telegram_feed.py", 0, False
