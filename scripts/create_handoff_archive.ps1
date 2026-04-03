param(
    [string]$ArchivePath = "splendor_ground0_handoff.zip",
    [string[]]$IncludeDataDirs = @("data/corpus_greedy_random_001")
)

$ErrorActionPreference = "Stop"

$stagingDir = Join-Path $PWD ".handoff_staging"
if (Test-Path $stagingDir) {
    Remove-Item -Recurse -Force $stagingDir
}
New-Item -ItemType Directory -Path $stagingDir | Out-Null

$pathsToCopy = @(
    "pyproject.toml",
    "README.md",
    ".gitignore",
    "run_gui.py",
    "run_replay_corpus.py",
    "run_supervised.py",
    "configs",
    "docs",
    "scripts",
    "src",
    "tests"
) + $IncludeDataDirs

foreach ($path in $pathsToCopy) {
    if (-not (Test-Path $path)) {
        continue
    }

    $destination = Join-Path $stagingDir $path
    $parent = Split-Path -Parent $destination
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }

    if ((Get-Item $path) -is [System.IO.DirectoryInfo]) {
        Copy-Item -Recurse -Force $path $destination
    } else {
        Copy-Item -Force $path $destination
    }
}

if (Test-Path $ArchivePath) {
    Remove-Item -Force $ArchivePath
}

Compress-Archive -Path (Join-Path $stagingDir "*") -DestinationPath $ArchivePath
Remove-Item -Recurse -Force $stagingDir

Write-Host "[handoff] wrote archive: $ArchivePath"
