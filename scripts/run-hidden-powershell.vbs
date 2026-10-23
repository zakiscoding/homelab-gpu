' Run a .ps1 with no visible window (Task Scheduler / logon).
Option Explicit
Dim sh, ps1, cmd
If WScript.Arguments.Count < 1 Then
  WScript.Echo "Usage: wscript run-hidden-powershell.vbs C:\path\to\script.ps1"
  WScript.Quit 1
End If
ps1 = WScript.Arguments(0)
cmd = "powershell.exe -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File """ & ps1 & """"
Set sh = CreateObject("Wscript.Shell")
' 0 = hidden window, True = wait for exit (needed for scheduled tasks)
sh.Run cmd, 0, True
