#!/usr/bin/env pwsh
# setup_all_projects.ps1 - Setup script for all ML projects using UV

Write-Host "🚀 tihassfjord's ML Portfolio Setup Script" -ForegroundColor Green
Write-Host "Setting up all projects with UV..." -ForegroundColor Yellow

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

Write-Host "`n📊 Beginner Projects: $($beginner_projects.Count)" -ForegroundColor Cyan
Write-Host "🚀 Intermediate Projects: $($intermediate_projects.Count)" -ForegroundColor Cyan
Write-Host "🏆 Advanced Projects: $($advanced_projects.Count)" -ForegroundColor Cyan
Write-Host "📁 Total Projects: $($all_projects.Count)" -ForegroundColor Green

foreach ($project in $all_projects) {
    Write-Host "`n📁 Setting up $project..." -ForegroundColor Cyan
    
    if (Test-Path $project) {
        Set-Location $project
        
        # Create virtual environment with UV
        Write-Host "  Creating virtual environment..." -ForegroundColor Gray
        uv venv --quiet
        
        # Install requirements
        Write-Host "  Installing requirements..." -ForegroundColor Gray
        uv pip install -r requirements.txt --quiet
        
        Write-Host "  ✅ $project setup complete!" -ForegroundColor Green
        Set-Location ..
    } else {
        Write-Host "  ❌ Project directory $project not found!" -ForegroundColor Red
    }
}

Write-Host "`n🎉 All projects setup complete!" -ForegroundColor Green
Write-Host "`nTo activate a project environment:" -ForegroundColor Yellow
Write-Host "  cd project-name" -ForegroundColor Gray
Write-Host "  .venv\Scripts\activate" -ForegroundColor Gray
Write-Host "`nTo run projects:" -ForegroundColor Yellow
Write-Host "  jupyter notebook  # For EDA projects" -ForegroundColor Gray
Write-Host "  python script.py  # For standalone scripts" -ForegroundColor Gray
