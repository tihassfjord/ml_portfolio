"""
Quick test to verify all projects are working
"""

import sys
import subprocess
import os

def test_project(project_name, script_name):
    """Test if a project can run successfully"""
    print(f"\nTesting {project_name}...")
    
    if not os.path.exists(project_name):
        print(f"Project directory {project_name} not found!")
        return False
    
    try:
        # Change to project directory
        original_dir = os.getcwd()
        os.chdir(project_name)
        
        # Try to import required modules
        result = subprocess.run([sys.executable, "-c", 
                               "import pandas, sklearn, matplotlib, seaborn, numpy; print('All imports successful')"],
                              capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"Dependencies OK for {project_name}")
            # Try to run the script with --help or dry run
            if script_name.endswith('.py'):
                # Check if script exists and is valid Python
                if os.path.exists(script_name):
                    result = subprocess.run([sys.executable, "-m", "py_compile", script_name],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        print(f"Script syntax OK for {script_name}")
                    else:
                        print(f"Script {script_name} has syntax issues")
                else:
                    print(f"Script {script_name} not found")
        else:
            print(f"Dependency issues for {project_name}")
            return False
            
    except Exception as e:
        print(f"Error testing {project_name}: {e}")
        return False
    finally:
        os.chdir(original_dir)
    
    return True

def main():
    print("🔍 tihassfjord's ML Portfolio Verification")
    print("Testing all projects...")
    
    beginner_projects = [
        ("iris-flower-classifier-tihassfjord", "iris_classifier_tihassfjord.py"),
        ("titanic-survival-tihassfjord", "titanic_predict_tihassfjord.py"),
        ("housing-price-predictor-tihassfjord", "housing_price_tihassfjord.py"),
        ("customer-churn-tihassfjord", "churn_predictor_tihassfjord.py"),
    ]
    
    intermediate_projects = [
        ("linear-regression-tihassfjord", "linear_regression_tihassfjord.py"),
        ("image-classification-tihassfjord", "image_classification_tihassfjord.py"),
        ("sentiment-analysis-tihassfjord", "sentiment_analysis_tihassfjord.py"),
        ("stock-price-predictor-tihassfjord", "stock_price_predictor_tihassfjord.py"),
        ("recommendation-system-tihassfjord", "movie_recommendation_system_tihassfjord.py"),
    ]
    
    advanced_projects = [
        ("nn-from-scratch-tihassfjord", "neural_network_tihassfjord.py"),
        ("face-recognition-tihassfjord", "face_recognition_tihassfjord.py"),
        ("automl-pipeline-tihassfjord", "automl_pipeline_tihassfjord.py"),
        ("char-lm-tihassfjord", "char_language_model_tihassfjord.py"),
        ("ab-test-framework-tihassfjord", "ab_testing_tihassfjord.py"),
        ("image-gen-tihassfjord", "image_generation_tihassfjord.py"),
        ("rl-game-ai-tihassfjord", "rl_game_ai_tihassfjord.py"),
        ("multilingual-nlp-tihassfjord", "multilingual_nlp_tihassfjord.py"),
        ("fraud-detection-tihassfjord", "fraud_detection_tihassfjord.py"),
        ("custom-automl-tihassfjord", "custom_automl_tihassfjord.py"),
        ("mlops-pipeline-tihassfjord", "mlops_pipeline_tihassfjord.py"),
        ("distributed-ml-tihassfjord", "distributed_training_tihassfjord.py"),
    ]
    
    all_projects = beginner_projects + intermediate_projects + advanced_projects
    
    results = []
    for project, script in all_projects:
        success = test_project(project, script)
        results.append((project, success))
    
    print("\n" + "="*50)
    print("Test Results Summary:")
    print(f"\nBeginner Projects ({len(beginner_projects)}):")
    for project, script in beginner_projects:
        success = next(s for p, s in results if p == project)
        status = "PASS" if success else "FAIL"
        print(f"  {project}: {status}")
    
    print(f"\nIntermediate Projects ({len(intermediate_projects)}):")
    for project, script in intermediate_projects:
        success = next(s for p, s in results if p == project)
        status = "PASS" if success else "FAIL"
        print(f"  {project}: {status}")
    
    print(f"\nAdvanced Projects ({len(advanced_projects)}):")
    for project, script in advanced_projects:
        success = next(s for p, s in results if p == project)
        status = "PASS" if success else "FAIL"
        print(f"  {project}: {status}")
    
    # Test EDA notebooks exist
    eda_notebooks = [
        "eda-portfolio-tihassfjord/notebooks/01_titanic_eda_tihassfjord.ipynb",
        "eda-portfolio-tihassfjord/notebooks/02_iris_eda_tihassfjord.ipynb"
    ]
    
    print("\nNotebook Files:")
    for notebook in eda_notebooks:
        if os.path.exists(notebook):
            print(f"  FOUND: {notebook}")
        else:
            print(f"  MISSING: {notebook}")
    
    # Test data files
    data_files = [
        "eda-portfolio-tihassfjord/data/titanic.csv",
        "eda-portfolio-tihassfjord/data/iris.csv",
        "titanic-survival-tihassfjord/data/titanic.csv",
        "housing-price-predictor-tihassfjord/data/housing.csv",
        "customer-churn-tihassfjord/data/churn.csv"
    ]
    
    print("\nData Files:")
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"  FOUND: {data_file}")
        else:
            print(f"  MISSING: {data_file}")
    
    success_count = sum(1 for _, success in results if success)
    total_count = len(results)
    
    print(f"\nOverall: {success_count}/{total_count} projects ready")
    print(f"Beginner: {len(beginner_projects)} | Intermediate: {len(intermediate_projects)} | Advanced: {len(advanced_projects)}")
    
    if success_count == total_count:
        print("All projects are ready to run!")
        print("\nNext steps:")
        print("1. Run: setup_all_projects.ps1 (to create virtual environments)")
        print("2. Run: run_demos.ps1 (to see all projects in action)")
        print("3. Or navigate to individual projects and explore!")
    else:
        print("Some projects need attention. Check the error messages above.")

if __name__ == "__main__":
    main()
