# package.ps1 - Create submission zip for Windows PowerShell
# Usage: powershell -ExecutionPolicy Bypass -File scripts/package.ps1 [GROUP_NUMBER] [TITLE]

param(
    [string]$Group = "GroupXX",
    [string]$Title = "Farmer_Advisory_Chatbot_Multilingual_NLP"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir

Push-Location $ProjectDir
try {
    python scripts/package.py $Group $Title
} finally {
    Pop-Location
}

