param(
  [string]$OutDir = "dist"
)

$ErrorActionPreference = 'Stop'

$manifestPath = Join-Path "extension" "manifest.json"
if (!(Test-Path $manifestPath)) { throw "manifest.json not found in extension/" }

$manifestJson = Get-Content $manifestPath -Raw | ConvertFrom-Json
$name = ($manifestJson.name -replace '\s+', '-' -replace '[^A-Za-z0-9\-]', '')
if (-not $name) { $name = "authentiscan" }
$version = $manifestJson.version
$timestamp = Get-Date -Format "yyyyMMddHHmmss"

New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$zipName = "$name-$version-$timestamp.zip"
$zipPath = Join-Path $OutDir $zipName
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }

Compress-Archive -Path (Join-Path "extension" "*") -DestinationPath $zipPath -Force

$hash = Get-FileHash -Path $zipPath -Algorithm SHA256
$size = (Get-Item $zipPath).Length

Write-Output "Package: $zipPath"
Write-Output "Version: $version"
Write-Output ("Size: {0:N0} bytes" -f $size)
Write-Output "SHA256: $($hash.Hash)"
