# Download using correct file URL (not folder URL)
$url = "https://drive.google.com/file/d/1eWOMwjcRstn1N1RHxFCsB5lYOL1HONkY/view?usp=sharing"
Invoke-WebRequest -Uri $url -OutFile models.zip

# Verify download completed
if ((Get-Item models.zip).Length -lt 1MB) {
    Write-Error "Download failed - file too small"
    exit 1
}

# Extract
New-Item -ItemType Directory -Path saved_models -Force
Expand-Archive -Path models.zip -DestinationPath saved_models -Force

# Verify extraction
Get-ChildItem -Recurse saved_models            