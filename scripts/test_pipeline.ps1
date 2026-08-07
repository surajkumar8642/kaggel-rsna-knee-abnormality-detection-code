param(
    [string]$Sample = "data/sample_submission_example.csv",
    [string]$Submission = "outputs/submission.csv"
)

 $python = if (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} else {
    "python3"
}

& $python src\submission_workflow.py build --sample-submission $Sample --submission $Submission --method zero
& $python src\submission_workflow.py validate --sample-submission $Sample --submission $Submission
