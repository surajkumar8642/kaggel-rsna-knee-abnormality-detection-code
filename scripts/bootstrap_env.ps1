param(
    [string]$PythonPath = "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"
)

function Resolve-Python {
    $candidates = @(
        $PythonPath,
        "$env:APPDATA\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "C:\Users\suraj2\AppData\Local\Programs\Python\Python311\python.exe",
        "python",
        "py"
    )

    foreach ($p in $candidates) {
        if ($p -match '[\\/]') {
            if (Test-Path $p) {
                return $p
            }
        } elseif (Get-Command $p -ErrorAction SilentlyContinue) {
            return $p
        }
    }
    return $null
}

$python = Resolve-Python
if (-not $python) {
    throw "Python not found. Install Python 3.11+ and retry."
}

Write-Output "Using Python: $python"
& $python --version

Write-Output "Installing requirements..."
& $python -m pip install -r requirements.txt

Write-Output "Installing extra tooling (kaggle, pydicom, matplotlib)..."
& $python -m pip install kaggle pydicom matplotlib

$kagglePath = "$env:USERPROFILE\.kaggle\kaggle.json"
if (Test-Path $kagglePath) {
    Write-Output "Kaggle auth file found: $kagglePath"
    Write-Output "If token is valid, test with: python -m kaggle competitions submissions rsna-knee-abnormality-detection"
} else {
    Write-Output "Kaggle auth file not found: $kagglePath"
    Write-Output "Go to https://www.kaggle.com/account > Create New API Token > save kaggle.json to %USERPROFILE%\\.kaggle\\kaggle.json"
}
