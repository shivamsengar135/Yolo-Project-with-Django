Option Explicit

Dim shell, fso, scriptDir, pythonwPath, appPath, cmd
Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonwPath = scriptDir & "\.venv\Scripts\pythonw.exe"
appPath = scriptDir & "\src\app_gui.py"

If Not fso.FileExists(pythonwPath) Then
    MsgBox "Virtual environment not found." & vbCrLf & _
           "Please run setup first:" & vbCrLf & _
           "python -m venv .venv" & vbCrLf & _
           ".\.venv\Scripts\python.exe -m pip install -r requirements.txt", _
           vbExclamation, "Object Detector"
    WScript.Quit 1
End If

shell.CurrentDirectory = scriptDir
cmd = """" & pythonwPath & """ """ & appPath & """"
shell.Run cmd, 0, False
