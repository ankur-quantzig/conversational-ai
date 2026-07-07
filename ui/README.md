# UI

React/Vite frontend for Insight Copilot.

Run locally from the repo root:

```powershell
$env:Path = "$PWD\tools\node-v24.18.0-win-x64;$env:Path"
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
.\tools\node-v24.18.0-win-x64\npm.cmd --prefix ui run dev -- --host 127.0.0.1
```

Build:

```powershell
$env:Path = "$PWD\tools\node-v24.18.0-win-x64;$env:Path"
.\tools\node-v24.18.0-win-x64\npm.cmd --prefix ui run build
```
