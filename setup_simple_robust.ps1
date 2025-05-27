# Simple and robust setup script
Write-Host "🚀 Simple & Robust ML Portfolio Setup" -ForegroundColor Green

# List of all projects
$projects = @(
    "eda-portfolio-tihassfjord",
    "iris-flower-classifier-tihassfjord", 
    "titanic-survival-tihassfjord",
    "housing-price-predictor-tihassfjord",
    "customer-churn-tihassfjord",
    "linear-regression-tihassfjord",
    "image-classification-tihassfjord",
    "sentiment-analysis-tihassfjord", 
    "stock-price-predictor-tihassfjord",
    "recommendation-system-tihassfjord",
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

$successCount = 0

foreach ($project in $projects) {
    Write-Host "`n📁 Setting up $project..." -ForegroundColor Cyan
    
    if (Test-Path $project) {
        Set-Location $project
        
        try {
            # Create virtual environment with standard Python venv
            Write-Host "  Creating virtual environment..." -ForegroundColor Gray
            python -m venv .venv
            
            # Install requirements if file exists
            if (Test-Path "requirements.txt") {
                Write-Host "  Installing requirements..." -ForegroundColor Gray
                .\.venv\Scripts\python.exe -m pip install --upgrade pip
                .\.venv\Scripts\python.exe -m pip install -r requirements.txt
            }
            
            Write-Host "  ✅ $project complete!" -ForegroundColor Green
            $successCount++
        } catch {
            Write-Host "  ❌ Error: $($_.Exception.Message)" -ForegroundColor Red
        }
        
        Set-Location ..
    } else {
        Write-Host "  ❌ Project directory not found!" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Setup complete! ($successCount/$($projects.Count) projects successful)" -ForegroundColor Green
Write-Host "`nTo activate a project:" -ForegroundColor Yellow
Write-Host "  cd project-name" -ForegroundColor Gray
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
