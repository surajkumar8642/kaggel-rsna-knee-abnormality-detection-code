param(
    [string]$Submission = "outputs/submission.csv",
    [string]$Competition = "rsna-knee-abnormality-detection",
    [string]$Message = "baseline submission"
)

function Resolve-Python {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:APPDATA\..\..\AppData\Local\Programs\Python\Python311\python.exe",
        "C:\Users\suraj2\AppData\Local\Programs\Python\Python311\python.exe",
        "python",
        "python3"
    )

    foreach ($p in $candidates) {
        if ($p -match '[\\/]' -and (Test-Path $p)) {
            try { & $p --version 2>$null | Out-Null; return $p } catch {}
        }
        if (Get-Command $p -ErrorAction SilentlyContinue) { return $p }
    }
    return $null
}

$python = Resolve-Python
if (-not $python) {
    throw "Python was not found. Install Python or add a valid python executable to PATH."
}
& $python src\submission_workflow.py submit --submission $Submission --competition $Competition --message $Message
