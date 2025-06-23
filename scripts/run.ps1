# run.ps1

$resultsDir = "results"
if (-Not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}

# Fixed log file name
$logFile = "$PSScriptRoot\log.txt"
Clear-Content $logFile -ErrorAction SilentlyContinue

function Run-DatasetEvaluations {
    param(
        [string]$datasetName,
        [string[]]$dirSuffixes
    )

    # Sort suffixes alphabetically
    $dirSuffixes = $dirSuffixes | Sort-Object

    foreach ($suffix in $dirSuffixes) {
        $savePath = Join-Path $resultsDir ($datasetName + "_" + $suffix)
        Write-Host "=== Running: Dataset=$datasetName, SaveDir=$savePath ==="
        Add-Content $logFile "`n=== Running: Dataset=$datasetName, SaveDir=$savePath ==="

        $cmd = @(
            "python -m TrajectoryNet.eval",
            "--dataset", $datasetName.ToUpper(),
            "--embedding_name", "PCA",
            "--leaveout_timepoint", "1",
            "--save", $savePath
        ) -join " "

        # Execute and log to file using Tee-Object
        & cmd /c $cmd 2>&1 | Tee-Object -FilePath $logFile -Append
    }
}

# Alphabetical order: CIRCLE5, CYCLE, TREE
$circle5Suffixes = @("base", "baseD", "baseDV", "base_V")
$cycleSuffixes   = @("base", "baseD", "baseDV", "base_V")
$treeSuffixes    = @("base", "baseD", "baseDV", "base_V")

# Run-DatasetEvaluations -datasetName "CIRCLE5" -dirSuffixes $circle5Suffixes
# Run-DatasetEvaluations -datasetName "CYCLE"   -dirSuffixes $cycleSuffixes
Run-DatasetEvaluations -datasetName "TREE"    -dirSuffixes $treeSuffixes

Write-Host "✅ All evaluations complete. Full log written to $logFile"
