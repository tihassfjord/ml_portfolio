# setup_all_projects.ps1 - Setup script for all ML projects

Write-Host "🚀 tihassfjord's ML Portfolio Setup Script" -ForegroundColor Green
Write-Host "Setting up all projects with virtual environments..." -ForegroundColor Yellow

# Check if UV is available, fall back to standard venv if not
$useUV = $false
try {
    $null = Get-Command uv -ErrorAction Stop
    $useUV = $true
    Write-Host "✅ UV detected - using UV for faster setup" -ForegroundColor Green
} catch {
    Write-Host "⚠️  UV not found - using standard Python venv" -ForegroundColor Yellow
}

$beginner_projects = @(
    "eda-portfolio-tihassfjord",
    "iris-flower-classifier-tihassfjord", 
    "titanic-survival-tihassfjord",
    "housing-price-predictor-tihassfjord",
    "customer-churn-tihassfjord"
)

$intermediate_projects = @(
    "linear-regression-tihassfjord",
    "image-classification-tihassfjord",
    "sentiment-analysis-tihassfjord", 
    "stock-price-predictor-tihassfjord",
    "recommendation-system-tihassfjord"
)

$advanced_projects = @(
    "nn-from-scratch-tihassfjord",
    "face-recognition-tihassfjord",
    "automl-pipeline-tihassfjord",
    "char-lm-tihassfjord",
    "ab-test-framework-tihassfjord",
    "image-gen-tihassfjord",
    "rl-game-ai-tihassfjord",
    "multilingual-nlp-tihassfjord",
    "fraud-detection-tihassfjord",
    "custom-automl-tihassfjord",
    "mlops-pipeline-tihassfjord",
    "distributed-ml-tihassfjord"
)

$all_projects = $beginner_projects + $intermediate_projects + $advanced_projects

Write-Host ""
Write-Host "📊 Beginner Projects: $($beginner_projects.Count)" -ForegroundColor Cyan
Write-Host "🚀 Intermediate Projects: $($intermediate_projects.Count)" -ForegroundColor Cyan
Write-Host "🏆 Advanced Projects: $($advanced_projects.Count)" -ForegroundColor Cyan
Write-Host "📁 Total Projects: $($all_projects.Count)" -ForegroundColor Green

$successCount = 0
$totalCount = $all_projects.Count

foreach ($project in $all_projects) {
    Write-Host ""
    Write-Host "📁 Setting up $project..." -ForegroundColor Cyan
    
    if (Test-Path $project) {
        try {
            Set-Location $project
            
            # Create virtual environment
            Write-Host "  Creating virtual environment..." -ForegroundColor Gray
            if ($useUV) {
                uv venv --quiet 2>$null
            } else {
                python -m venv .venv 2>$null
            }
            
            # Install requirements if file exists
            if (Test-Path "requirements.txt") {
                Write-Host "  Installing requirements..." -ForegroundColor Gray
                if ($useUV) {
                    uv pip install -r requirements.txt --quiet 2>$null
                } else {
                    .\.venv\Scripts\python.exe -m pip install --upgrade pip --quiet 2>$null
                    .\.venv\Scripts\python.exe -m pip install -r requirements.txt --quiet 2>$null
                }
            } else {
                Write-Host "  No requirements.txt found - skipping package installation" -ForegroundColor Yellow
            }
            
            Write-Host "  ✅ $project setup complete!" -ForegroundColor Green
            $successCount++
        } catch {
            Write-Host "  ❌ Error setting up $project : $($_.Exception.Message)" -ForegroundColor Red
        } finally {
            Set-Location ..
        }
    } else {
        Write-Host "  ❌ Project directory $project not found!" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "🎉 Setup complete! ($successCount/$totalCount projects successful)" -ForegroundColor Green
Write-Host ""
Write-Host "To activate a project environment:" -ForegroundColor Yellow
Write-Host "  cd project-name" -ForegroundColor Gray
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host ""
Write-Host "To run projects:" -ForegroundColor Yellow
Write-Host "  jupyter notebook  # For EDA projects" -ForegroundColor Gray
Write-Host "  python script.py  # For standalone scripts" -ForegroundColor Gray