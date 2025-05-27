#!/usr/bin/env pwsh
# run_demos.ps1 - Quick demo runner for all projects

Write-Host "🎯 tihassfjord's ML Portfolio Demo Runner" -ForegroundColor Green

$beginner_demos = @{
    "iris-flower-classifier-tihassfjord" = "iris_classifier_tihassfjord.py"
    "titanic-survival-tihassfjord" = "titanic_predict_tihassfjord.py"
    "housing-price-predictor-tihassfjord" = "housing_price_tihassfjord.py"
    "customer-churn-tihassfjord" = "churn_predictor_tihassfjord.py"
}

$intermediate_demos = @{
    "linear-regression-tihassfjord" = "linear_regression_tihassfjord.py"
    "image-classification-tihassfjord" = "image_classification_tihassfjord.py"
    "sentiment-analysis-tihassfjord" = "sentiment_analysis_tihassfjord.py"
    "stock-price-predictor-tihassfjord" = "stock_price_predictor_tihassfjord.py"
    "recommendation-system-tihassfjord" = "movie_recommendation_system_tihassfjord.py"
}

$advanced_demos = @{
    "nn-from-scratch-tihassfjord" = "neural_network_tihassfjord.py"
    "face-recognition-tihassfjord" = "face_recognition_tihassfjord.py"
    "automl-pipeline-tihassfjord" = "automl_pipeline_tihassfjord.py"
    "char-lm-tihassfjord" = "char_language_model_tihassfjord.py"
    "ab-test-framework-tihassfjord" = "ab_testing_tihassfjord.py"
    "image-gen-tihassfjord" = "image_generation_tihassfjord.py"
    "rl-game-ai-tihassfjord" = "rl_game_ai_tihassfjord.py"
    "multilingual-nlp-tihassfjord" = "multilingual_nlp_tihassfjord.py"
    "fraud-detection-tihassfjord" = "fraud_detection_tihassfjord.py"
    "custom-automl-tihassfjord" = "custom_automl_tihassfjord.py"
    "mlops-pipeline-tihassfjord" = "mlops_pipeline_tihassfjord.py"
    "distributed-ml-tihassfjord" = "distributed_training_tihassfjord.py"
}

Write-Host "`n📊 Available Demos:" -ForegroundColor Yellow
Write-Host "Beginner Projects: $($beginner_demos.Count)" -ForegroundColor Cyan
Write-Host "Intermediate Projects: $($intermediate_demos.Count)" -ForegroundColor Cyan
Write-Host "Advanced Projects: $($advanced_demos.Count)" -ForegroundColor Cyan

$all_demos = $beginner_demos + $intermediate_demos + $advanced_demos

foreach ($project in $all_demos.Keys) {
    Write-Host "`n🔥 Running $project demo..." -ForegroundColor Cyan
    
    if (Test-Path $project) {
        try {
            Set-Location $project
            
            # Check if the Python script exists
            $scriptName = $all_demos[$project]
            if (-not (Test-Path $scriptName)) {
                Write-Host "  ⚠️  Script $scriptName not found in $project" -ForegroundColor Yellow
                Set-Location ..
                continue
            }
            
            # Check if virtual environment exists and use it
            if (Test-Path ".venv\Scripts\python.exe") {
                Write-Host "  Using virtual environment..." -ForegroundColor Gray
                & .\.venv\Scripts\python.exe $scriptName
            } elseif (Test-Path ".venv\bin\python") {
                # For Unix/Linux systems
                Write-Host "  Using virtual environment..." -ForegroundColor Gray
                & ./.venv/bin/python $scriptName
            } else {
                Write-Host "  ⚠️  Virtual environment not found. Using system Python..." -ForegroundColor Yellow
                Write-Host "  ⚠️  Run setup_all_projects.ps1 first for better results!" -ForegroundColor Yellow
                python $scriptName
            }
            
            Set-Location ..
        } catch {
            Write-Host "  ❌ Error running $project : $($_.Exception.Message)" -ForegroundColor Red
            Set-Location ..
        }
    } else {
        Write-Host "  ❌ Project $project not found!" -ForegroundColor Red
    }
    
    Write-Host "`n  Press any key to continue to next demo..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

Write-Host "`n🎉 All demos completed!" -ForegroundColor Green
Write-Host "`nFor EDA projects, run:" -ForegroundColor Yellow
Write-Host "  cd eda-portfolio-tihassfjord" -ForegroundColor Gray
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor Gray
Write-Host "  jupyter notebook" -ForegroundColor Gray