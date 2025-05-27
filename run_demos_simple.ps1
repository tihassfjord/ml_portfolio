# Simple demo runner - no fancy features, just works
Write-Host "🎯 Simple Demo Runner" -ForegroundColor Green

# Project -> Script mapping
$demos = @{
    "iris-flower-classifier-tihassfjord" = "iris_classifier_tihassfjord.py"
    "titanic-survival-tihassfjord" = "titanic_predict_tihassfjord.py"
    "housing-price-predictor-tihassfjord" = "housing_price_tihassfjord.py"
    "customer-churn-tihassfjord" = "churn_predictor_tihassfjord.py"
    "linear-regression-tihassfjord" = "linear_regression_tihassfjord.py"
    "nn-from-scratch-tihassfjord" = "neural_network_tihassfjord.py"
}

foreach ($project in $demos.Keys) {
    Write-Host "`n🔥 Testing $project..." -ForegroundColor Cyan
    
    if (Test-Path $project) {
        Set-Location $project
        
        $scriptName = $demos[$project]
        if (Test-Path $scriptName) {
            # Try venv first, fallback to system python
            if (Test-Path ".venv\Scripts\python.exe") {
                Write-Host "  Using venv..." -ForegroundColor Gray
                & .\.venv\Scripts\python.exe $scriptName
            } else {
                Write-Host "  Using system python..." -ForegroundColor Yellow
                python $scriptName
            }
        } else {
            Write-Host "  ❌ Script $scriptName not found!" -ForegroundColor Red
        }
        
        Set-Location ..
        
        # Wait for user input
        Write-Host "`n  Press Enter to continue..." -ForegroundColor Gray
        Read-Host
    } else {
        Write-Host "  ❌ Project $project not found!" -ForegroundColor Red
    }
}

Write-Host "`n🎉 Demo run complete!" -ForegroundColor Green
