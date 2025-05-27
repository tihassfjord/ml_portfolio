Write-Host "🚀 Simple Setup Script" -ForegroundColor Green

$projects = "eda-portfolio-tihassfjord", "iris-flower-classifier-tihassfjord", "titanic-survival-tihassfjord", "housing-price-predictor-tihassfjord", "customer-churn-tihassfjord", "linear-regression-tihassfjord", "image-classification-tihassfjord", "sentiment-analysis-tihassfjord", "stock-price-predictor-tihassfjord", "recommendation-system-tihassfjord", "nn-from-scratch-tihassfjord", "face-recognition-tihassfjord", "automl-pipeline-tihassfjord", "char-lm-tihassfjord", "ab-test-framework-tihassfjord", "image-gen-tihassfjord", "rl-game-ai-tihassfjord", "multilingual-nlp-tihassfjord", "fraud-detection-tihassfjord", "custom-automl-tihassfjord", "mlops-pipeline-tihassfjord", "distributed-ml-tihassfjord"

foreach ($project in $projects) {
    Write-Host "Setting up $project..." -ForegroundColor Cyan
    if (Test-Path $project) {
        Set-Location $project
        uv venv --quiet
        uv pip install -r requirements.txt --quiet
        Write-Host "✅ $project complete!" -ForegroundColor Green
        Set-Location ..
    } else {
        Write-Host "❌ $project not found!" -ForegroundColor Red
    }
}
Write-Host "🎉 Setup complete!" -ForegroundColor Green