# A/B Testing Framework (highlight) — tihassfjord

## Goal
Design and analyze A/B tests with statistical significance testing and comprehensive reporting.

## Dataset
- Synthetic experiment data or real A/B test results
- Configurable sample sizes and effect sizes

## Requirements
- Python 3.8+
- numpy
- scipy
- matplotlib
- seaborn
- pandas

## How to Run
```bash
# Run with synthetic data
python ab_test_framework_tihassfjord.py

# Run with custom data file
python ab_test_framework_tihassfjord.py data/experiment_data.csv
```

## Example Output
```
A/B Testing Framework by tihassfjord
====================================

Experiment: Landing Page Conversion
Control (A): 1000 samples, 110 conversions (11.0%)
Treatment (B): 1000 samples, 130 conversions (13.0%)

Statistical Analysis:
P-value: 0.042
Effect Size: 2.0%
95% Confidence Interval: [0.1%, 3.9%]
Result: Statistically Significant ✓

Recommendation: Implement Treatment B
```

## Project Structure
```
ab-test-framework-tihassfjord/
│
├── ab_test_framework_tihassfjord.py  # Main testing framework
├── data/                             # Experiment data
│   └── sample_experiment.csv        # Sample A/B test data
├── reports/                         # Generated reports
├── requirements.txt                 # Dependencies
└── README.md                       # This file
```

## Key Features
- Statistical significance testing
- Effect size calculation
- Confidence intervals
- Power analysis
- Multiple comparison correction
- Visualization and reporting
- Experiment design recommendations
- Sample size calculator

## Learning Outcomes
- A/B testing methodology
- Statistical hypothesis testing
- Effect size and practical significance
- Experimental design principles
- Statistical power analysis
- Data-driven decision making

---
*Project by tihassfjord - Advanced ML Portfolio*
