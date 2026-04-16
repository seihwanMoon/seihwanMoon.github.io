param(
    [string]$VaultPath = 'G:\My Drive\Bobsidian\github.vault',
    [string]$ExportPath = 'G:\My Drive\Bobsidian\seihwanMoon.github.io'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$generatorPath = Join-Path $PSScriptRoot 'publish-vault-site.mjs'
if (-not (Test-Path -LiteralPath $generatorPath)) {
    throw "Generator script not found: $generatorPath"
}

& node $generatorPath --vault-path $VaultPath --export-path $ExportPath
if ($LASTEXITCODE -ne 0) {
    throw "Site generation failed with exit code $LASTEXITCODE"
}
