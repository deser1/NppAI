# Run from a checkout of the model-repair release on an interactive Windows x64
# desktop with Docker Desktop's Linux engine, Python3.12 and gh installed.
$ErrorActionPreference = 'Stop'
$tag = 'model-factorial-v1'
$root = Join-Path $env:TEMP ('nppai-validated-' + [guid]::NewGuid())
New-Item -ItemType Directory $root | Out-Null
Push-Location $PSScriptRoot\..
try {
    gh release download $tag --repo deser1/NppAI --dir $root
    if ($LASTEXITCODE) { throw 'Model release download failed' }
    gh release download v0.5.0-rc.2 --repo deser1/NppAI --pattern NppAI-x64.zip --dir $root
    if ($LASTEXITCODE) { throw 'Plugin download failed' }
    # Release hashes are checked against the published checksum file.
    foreach ($line in Get-Content "$root/SHA256SUMS-model.txt") {
        if ($line -notmatch '^([0-9a-f]{64})  ([A-Za-z0-9_.-]+)$') { throw 'Invalid checksum entry' }
        if ((Get-FileHash (Join-Path $root $Matches[2]) -Algorithm SHA256).Hash.ToLower() -ne $Matches[1]) { throw 'Release checksum mismatch' }
    }
    if ((Get-FileHash "$root/NppAI-x64.zip" -Algorithm SHA256).Hash.ToLower() -ne 'faac23fa483cf9a80da4406ebc8a21a7ec55d27f7da8972e9980b3b1a2d09848') { throw 'Plugin checksum mismatch' }
    Invoke-WebRequest 'https://github.com/notepad-plus-plus/notepad-plus-plus/releases/download/v8.9.8.1/npp.8.9.8.1.portable.x64.zip' -OutFile "$root/editor.zip"
    if ((Get-FileHash "$root/editor.zip" -Algorithm SHA256).Hash.ToLower() -ne 'beddf5548e75c97930d1575ba1c7d113e6e2b33d0a200310e734ca08e225f414') { throw 'Editor checksum mismatch' }
    py -3.12 -m venv "$root/venv"
    $python = "$root/venv/Scripts/python.exe"
    & $python -m pip install -r requirements-inference.txt 'pillow>=10,<14'
    if ($LASTEXITCODE) { throw 'Dependency install failed' }
    $env:NPPAI_DOCKER_HOST = 'npipe:////./pipe/docker_engine'
    docker --host npipe:////./pipe/docker_engine pull python:3.12-slim
    if ($LASTEXITCODE) { throw 'Docker Desktop Linux image unavailable' }
    $image = docker --host npipe:////./pipe/docker_engine image inspect python:3.12-slim --format '{{.Id}}'
    if ($LASTEXITCODE) { throw 'Docker image identity unavailable' }
    & $python scripts/portable_notepad_smoke.py --editor-zip "$root/editor.zip" --package "$root/NppAI-x64.zip" --model "$root/NppAI-factorial.nppai" --service-zip "$root/NppAI-factorial-service.zip" --output "$root/validated-evidence" --prompt-bytes 256 --python-functions --plain-task --exercise-generation --record-gif --validation-image $image
    if ($LASTEXITCODE) { throw 'Validated GUI Apply/Undo failed; retain evidence' }
    Write-Host "Actual validated GUI evidence: $root/validated-evidence"
} finally { Pop-Location }
