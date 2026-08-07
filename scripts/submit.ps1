param(
    [string]$Submission = "outputs/submission.csv",
    [string]$Competition = "rsna-knee-abnormality-detection",
    [string]$Message = "baseline submission"
)

$python = if (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} else {
    "python3"
}

& $python src\submission_workflow.py submit --submission $Submission --competition $Competition --message $Message
