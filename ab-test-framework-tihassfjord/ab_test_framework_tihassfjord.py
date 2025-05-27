"""
A/B Testing Framework by tihassfjord
Comprehensive toolkit for designing, running, and analyzing A/B tests
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ABTestFramework:
    """Complete A/B testing framework with statistical analysis"""
    
    def __init__(self, alpha=0.05, power=0.8):
        self.alpha = alpha  # Significance level
        self.power = power  # Statistical power
        self.results = {}
        
        # Create directories
        Path("data").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)
    
    def load_data(self, file_path=None):
        """Load experiment data"""
        if file_path is None:
            # Create sample data
            self._create_sample_data()
            file_path = "data/sample_experiment.csv"
        
        print(f"Loading experiment data: {file_path}")
        self.data = pd.read_csv(file_path)
        print(f"Data shape: {self.data.shape}")
        print(f"Columns: {list(self.data.columns)}")
        
        return self.data
    
    def _create_sample_data(self):
        """Create sample A/B test data"""
        np.random.seed(42)
        
        # Experiment parameters
        n_control = 1000
        n_treatment = 1000
        control_rate = 0.11  # 11% conversion rate
        treatment_rate = 0.13  # 13% conversion rate (2% lift)
        
        # Generate data
        control_conversions = np.random.binomial(1, control_rate, n_control)
        treatment_conversions = np.random.binomial(1, treatment_rate, n_treatment)
        
        # Create DataFrame
        data = []
        
        # Control group
        for i, conversion in enumerate(control_conversions):
            data.append({
                'user_id': f'ctrl_{i}',
                'group': 'control',
                'converted': conversion,
                'revenue': np.random.normal(25, 10) if conversion else 0,
                'session_duration': np.random.exponential(120),
                'page_views': np.random.poisson(3)
            })
        
        # Treatment group
        for i, conversion in enumerate(treatment_conversions):
            data.append({
                'user_id': f'treat_{i}',
                'group': 'treatment',
                'converted': conversion,
                'revenue': np.random.normal(27, 10) if conversion else 0,
                'session_duration': np.random.exponential(140),
                'page_views': np.random.poisson(3.5)
            })
        
        df = pd.DataFrame(data)
        df.to_csv("data/sample_experiment.csv", index=False)
        print("Created sample experiment data: data/sample_experiment.csv")
    
    def analyze_conversion_rate(self, metric='converted'):
        """Analyze conversion rate differences"""
        print(f"\nAnalyzing {metric} rates...")
        
        # Group by experiment group
        summary = self.data.groupby('group')[metric].agg(['count', 'sum', 'mean']).round(4)
        summary['rate'] = summary['mean']
        summary.columns = ['total', 'conversions', 'rate', 'rate_display']
        
        print("\nConversion Summary:")
        print(summary)
        
        # Extract data for statistical tests
        control_data = self.data[self.data['group'] == 'control'][metric]
        treatment_data = self.data[self.data['group'] == 'treatment'][metric]
        
        control_conversions = control_data.sum()
        control_total = len(control_data)
        treatment_conversions = treatment_data.sum()
        treatment_total = len(treatment_data)
        
        # Create contingency table
        contingency_table = np.array([
            [control_conversions, control_total - control_conversions],
            [treatment_conversions, treatment_total - treatment_conversions]
        ])
        
        # Chi-square test
        chi2, p_value_chi2, dof, expected = chi2_contingency(contingency_table)
        
        # Fisher's exact test (more accurate for small samples)
        odds_ratio, p_value_fisher = fisher_exact(contingency_table)
        
        # Effect size (Cohen's h for proportions)
        p1 = control_conversions / control_total
        p2 = treatment_conversions / treatment_total
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Confidence interval for difference in proportions
        p_pooled = (control_conversions + treatment_conversions) / (control_total + treatment_total)
        se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))
        diff = p2 - p1
        margin_error = stats.norm.ppf(1 - self.alpha/2) * se_diff
        ci_lower = diff - margin_error
        ci_upper = diff + margin_error
        
        # Store results
        self.results[metric] = {
            'control_rate': p1,
            'treatment_rate': p2,
            'difference': diff,
            'lift_percent': (diff / p1) * 100 if p1 > 0 else 0,
            'chi2_statistic': chi2,
            'p_value_chi2': p_value_chi2,
            'p_value_fisher': p_value_fisher,
            'effect_size': effect_size,
            'confidence_interval': (ci_lower, ci_upper),
            'significant': p_value_fisher < self.alpha
        }
        
        return self.results[metric]
    
    def analyze_continuous_metric(self, metric='revenue'):
        """Analyze continuous metrics (e.g., revenue, session duration)"""
        print(f"\nAnalyzing {metric}...")
        
        control_data = self.data[self.data['group'] == 'control'][metric]
        treatment_data = self.data[self.data['group'] == 'treatment'][metric]
        
        # Summary statistics
        summary = self.data.groupby('group')[metric].describe()
        print(f"\n{metric} Summary:")
        print(summary)
        
        # Statistical tests
        # T-test (assuming normal distribution)
        t_stat, p_value_ttest = stats.ttest_ind(control_data, treatment_data)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, p_value_mann = stats.mannwhitneyu(control_data, treatment_data, alternative='two-sided')
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * control_data.var() + 
                             (len(treatment_data) - 1) * treatment_data.var()) / 
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = (treatment_data.mean() - control_data.mean()) / pooled_std
        
        # Confidence interval for difference in means
        diff_mean = treatment_data.mean() - control_data.mean()
        se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - self.alpha/2, df)
        margin_error = t_critical * se_diff
        ci_lower = diff_mean - margin_error
        ci_upper = diff_mean + margin_error
        
        # Store results
        self.results[metric] = {
            'control_mean': control_data.mean(),
            'treatment_mean': treatment_data.mean(),
            'difference': diff_mean,
            'percent_change': (diff_mean / control_data.mean()) * 100 if control_data.mean() != 0 else 0,
            't_statistic': t_stat,
            'p_value_ttest': p_value_ttest,
            'p_value_mann_whitney': p_value_mann,
            'effect_size_cohens_d': cohens_d,
            'confidence_interval': (ci_lower, ci_upper),
            'significant': p_value_ttest < self.alpha
        }
        
        return self.results[metric]
    
    def calculate_sample_size(self, baseline_rate, minimum_detectable_effect, alpha=None, power=None):
        """Calculate required sample size for A/B test"""
        if alpha is None:
            alpha = self.alpha
        if power is None:
            power = self.power
        
        # Effect size calculation
        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect
        
        # Cohen's h
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Sample size calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        n = 2 * ((z_alpha + z_beta) / h) ** 2
        
        return int(np.ceil(n))
    
    def run_power_analysis(self):
        """Run power analysis for the current experiment"""
        print("\nPower Analysis:")
        print("-" * 30)
        
        # Get conversion data
        if 'converted' in self.results:
            result = self.results['converted']
            observed_effect = abs(result['difference'])
            
            # Calculate achieved power
            n_per_group = len(self.data[self.data['group'] == 'control'])
            h = abs(result['effect_size'])
            
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_power = h * np.sqrt(n_per_group / 2) - z_alpha
            achieved_power = stats.norm.cdf(z_power)
            
            print(f"Observed effect size: {observed_effect:.4f}")
            print(f"Sample size per group: {n_per_group}")
            print(f"Achieved power: {achieved_power:.3f}")
            
            # Minimum detectable effect for current sample size
            min_detectable = (z_alpha + stats.norm.ppf(self.power)) * np.sqrt(2 / n_per_group)
            min_detectable_rate = 2 * np.sin(min_detectable / 2) ** 2
            
            print(f"Minimum detectable effect (current n): {min_detectable_rate:.4f}")
            
            # Required sample size for observed effect
            required_n = self.calculate_sample_size(
                baseline_rate=result['control_rate'],
                minimum_detectable_effect=observed_effect
            )
            print(f"Required n for observed effect: {required_n} per group")
    
    def visualize_results(self):
        """Create visualizations of A/B test results"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('A/B Test Results (tihassfjord)', fontsize=16, fontweight='bold')
        
        # 1. Conversion rates
        if 'converted' in self.results:
            ax1 = axes[0, 0]
            groups = ['Control', 'Treatment']
            rates = [self.results['converted']['control_rate'], 
                    self.results['converted']['treatment_rate']]
            bars = ax1.bar(groups, rates, color=['skyblue', 'lightcoral'])
            ax1.set_title('Conversion Rates')
            ax1.set_ylabel('Conversion Rate')
            
            # Add value labels on bars
            for bar, rate in zip(bars, rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                        f'{rate:.3f}', ha='center', va='bottom')
        
        # 2. Revenue comparison
        if 'revenue' in self.results:
            ax2 = axes[0, 1]
            control_revenue = self.data[self.data['group'] == 'control']['revenue']
            treatment_revenue = self.data[self.data['group'] == 'treatment']['revenue']
            
            ax2.boxplot([control_revenue, treatment_revenue], labels=['Control', 'Treatment'])
            ax2.set_title('Revenue Distribution')
            ax2.set_ylabel('Revenue')
        
        # 3. Statistical significance
        ax3 = axes[1, 0]
        metrics = list(self.results.keys())
        p_values = [self.results[m]['p_value_fisher'] if 'p_value_fisher' in self.results[m] 
                   else self.results[m]['p_value_ttest'] for m in metrics]
        
        colors = ['red' if p < self.alpha else 'green' for p in p_values]
        bars = ax3.bar(metrics, p_values, color=colors, alpha=0.7)
        ax3.axhline(y=self.alpha, color='red', linestyle='--', label=f'α = {self.alpha}')
        ax3.set_title('P-values by Metric')
        ax3.set_ylabel('P-value')
        ax3.legend()
        
        # Add significance indicators
        for bar, p_val in zip(bars, p_values):
            significance = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    significance, ha='center', va='bottom')
        
        # 4. Effect sizes
        ax4 = axes[1, 1]
        effect_sizes = []
        effect_labels = []
        
        for metric in metrics:
            if 'effect_size' in self.results[metric]:
                effect_sizes.append(abs(self.results[metric]['effect_size']))
                effect_labels.append(f"{metric}\n(Cohen's h)")
            elif 'effect_size_cohens_d' in self.results[metric]:
                effect_sizes.append(abs(self.results[metric]['effect_size_cohens_d']))
                effect_labels.append(f"{metric}\n(Cohen's d)")
        
        if effect_sizes:
            bars = ax4.bar(range(len(effect_sizes)), effect_sizes, color='orange', alpha=0.7)
            ax4.set_xticks(range(len(effect_sizes)))
            ax4.set_xticklabels(effect_labels)
            ax4.set_title('Effect Sizes')
            ax4.set_ylabel('Effect Size')
            
            # Add effect size interpretation lines
            ax4.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Small (0.2)')
            ax4.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium (0.5)')
            ax4.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='Large (0.8)')
            ax4.legend()
        
        plt.tight_layout()
        plt.savefig('reports/ab_test_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate comprehensive A/B test report"""
        print("\n" + "="*60)
        print("A/B TEST REPORT (tihassfjord)")
        print("="*60)
        
        for metric, result in self.results.items():
            print(f"\n{metric.upper()} ANALYSIS:")
            print("-" * 40)
            
            if 'control_rate' in result:  # Binary metric
                print(f"Control: {result['control_rate']:.3f} ({result['control_rate']*100:.1f}%)")
                print(f"Treatment: {result['treatment_rate']:.3f} ({result['treatment_rate']*100:.1f}%)")
                print(f"Difference: {result['difference']:.4f} ({result['lift_percent']:.1f}% lift)")
                print(f"P-value: {result['p_value_fisher']:.4f}")
                
            else:  # Continuous metric
                print(f"Control: {result['control_mean']:.2f}")
                print(f"Treatment: {result['treatment_mean']:.2f}")
                print(f"Difference: {result['difference']:.2f} ({result['percent_change']:.1f}% change)")
                print(f"P-value: {result['p_value_ttest']:.4f}")
            
            print(f"95% CI: [{result['confidence_interval'][0]:.4f}, {result['confidence_interval'][1]:.4f}]")
            print(f"Result: {'✓ Significant' if result['significant'] else '✗ Not Significant'}")
        
        # Overall recommendation
        print(f"\n{'='*60}")
        print("RECOMMENDATION:")
        
        significant_improvements = [m for m, r in self.results.items() if r['significant'] and 
                                  (r.get('difference', 0) > 0 or r.get('lift_percent', 0) > 0)]
        
        if significant_improvements:
            print("✓ IMPLEMENT TREATMENT")
            print(f"  Significant improvements in: {', '.join(significant_improvements)}")
        else:
            print("✗ DO NOT IMPLEMENT")
            print("  No statistically significant improvements detected")
        
        print("="*60)
    
    def run_complete_analysis(self, file_path=None):
        """Run complete A/B test analysis"""
        print("A/B Testing Framework by tihassfjord")
        print("="*40)
        
        try:
            # Load data
            self.load_data(file_path)
            
            # Analyze metrics
            self.analyze_conversion_rate('converted')
            
            if 'revenue' in self.data.columns:
                self.analyze_continuous_metric('revenue')
            
            if 'session_duration' in self.data.columns:
                self.analyze_continuous_metric('session_duration')
            
            # Power analysis
            self.run_power_analysis()
            
            # Generate visualizations and report
            self.visualize_results()
            self.generate_report()
            
            print(f"\n🎉 A/B test analysis complete!")
            
        except Exception as e:
            print(f"Error in A/B test analysis: {e}")
            raise

def main():
    """Main function"""
    # Get data file from command line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Create and run A/B test framework
    ab_test = ABTestFramework(alpha=0.05, power=0.8)
    ab_test.run_complete_analysis(file_path)

if __name__ == "__main__":
    main()
