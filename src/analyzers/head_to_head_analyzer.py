"""
Analyzer for comparing fields head-to-head to identify correlations between them.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter, defaultdict
from scipy.stats import chi2_contingency
from analyzers.base import register_analyzer, Analyzer

@register_analyzer
class HeadToHeadAnalyzer(Analyzer):
    """Analyzer that compares pairs of fields to identify correlations."""
    
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        return "Head-to-Head Field Comparison Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Analyzes correlations between pairs of fields in the dataset documentation"
    
    def analyze(self, data, output_dir='outputs'):
        """
        Analyze correlations between pairs of fields in the dataset documentation.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Get analyzer-specific output directory
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        
        output_file = os.path.join(analyzer_dir, 'head_to_head_analysis.txt')
        
        # Load field mappings and answer mappings
        answer_mapping = self._load_json('data/answer_mapping.json')
        
        # Format: (field1, field2, question1, question2, simplify_field1, simplify_field2, custom_title)
        field_pairs = [
            ('Overlap Synthesis', 'Synthesis Type', 
             'What type of overlap synthesis was used?', 'Was the synthesis type described?', 
             False, True, 'Overlap Synthesis vs Synthesis Type'),
            
            ('Human Labels', 'OG Labels', 
             'Did humans label the dataset?', 'Were the used labels original?', 
             False, False, 'Human Labels vs OG Labels'),
            
            ('Compensation', 'Training', 
             'Was the compensation method stated?', 'Was annotator training documented?', 
             False, False, 'Compensation vs Training'),
        ]
        
        # Process each field pair
        results = {}
        plot_files = []
        
        for pair in field_pairs:
            field1, field2, question1, question2, simplify1, simplify2, custom_title = pair
            
            # Skip pairs where one or both fields don't exist in the datasets
            if not self._fields_exist_in_data(data, field1, field2):
                print(f"Skipping pair ({field1}, {field2}) - one or both fields not found in datasets.")
                continue
                
            # Process the field pair
            pair_data = self._process_field_pair(data, field1, field2, 
                                                simplify1, simplify2,
                                                answer_mapping)
            
            if pair_data:
                # Create visualization
                plot_file = self._create_comparison_plot(pair_data, field1, field2, 
                                                        question1, question2, custom_title,
                                                        output_dir)
                plot_files.append(plot_file)
                
                # Store results
                results[(field1, field2, question1, question2, custom_title)] = pair_data
        
        # Write analysis results
        self._write_analysis(output_file, results, plot_files)
        
        return output_file
    
    def _load_json(self, file_path):
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def _fields_exist_in_data(self, data, field1, field2):
        """Check if both fields exist in at least one dataset."""
        for dataset_info in data.values():
            if field1 in dataset_info and field2 in dataset_info:
                return True
        return False
    
    def _process_field_pair(self, data, field1, field2, simplify1, simplify2, answer_mapping):
        """Process a pair of fields to prepare for correlation analysis."""
        # Collect all datasets that have both fields
        valid_datasets = []
        field1_values = []
        field2_values = []
        
        for dataset_name, dataset_info in data.items():
            if field1 in dataset_info and field2 in dataset_info:
                value1 = dataset_info[field1]
                value2 = dataset_info[field2]
                
                # Standardize answers if needed
                if simplify1:
                    value1 = self._standardize_answer(value1, answer_mapping, field1)
                if simplify2:
                    value2 = self._standardize_answer(value2, answer_mapping, field2)
                
                # Add to lists
                valid_datasets.append(dataset_name)
                field1_values.append(value1)
                field2_values.append(value2)
        
        if not valid_datasets:
            return None
            
        # Create a DataFrame for easier processing
        df = pd.DataFrame({
            'dataset': valid_datasets,
            'field1': field1_values,
            'field2': field2_values
        })
        
        # Get contingency table
        contingency = pd.crosstab(df['field1'], df['field2'])
        
        # Calculate chi-square test
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        # Calculate Cramer's V (normalized measure of association)
        n = contingency.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
        
        return {
            'df': df,
            'contingency': contingency,
            'chi2': chi2,
            'p_value': p,
            'dof': dof,
            'cramers_v': cramers_v
        }
    
    def _standardize_answer(self, value, answer_mapping, field):
        """
        Standardize an answer using simplified rules for non-dropdown fields.
        
        Args:
            value: The field value to standardize
            answer_mapping: Dictionary of answer mappings
            field: The field name
        
        Returns:
            Standardized answer category
        """
        if value is None:
            return "No"
        
        value_str = str(value).strip()
        if not value_str or value_str.lower() == 'nan':
            return "No"
        
        # Check for negative patterns
        negative_patterns = [
            "no information", "unknown", 
            "no details", "none", "not reported", "not specified"
        ]

        na_patterns = ["not applicable", 'n/a']

        if any(pattern in value_str.lower() for pattern in na_patterns):
            return "Not applicable"
        
        if any(pattern in value_str.lower() for pattern in negative_patterns):
            return "No"
            
        return "Yes"
    
    def _create_comparison_plot(self, pair_data, field1, field2, question1, question2, custom_title, output_dir):
        """Create a visualization comparing two fields."""
        # Get the contingency table
        contingency = pair_data['contingency']
        
        # Set up figure with appropriate size
        plt.figure(figsize=(14, 10))
        plt.style.use('ggplot')
        
        # Create a visually appealing color palette
        colors = sns.color_palette("viridis", len(contingency.columns))
        
        # Calculate percentages for each row
        contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
        
        # Plot stacked bars
        ax = contingency.plot(kind='bar', stacked=True, figsize=(14, 8), 
                             colormap='viridis', width=0.7)
        
        # Enhance the plot
        ax.set_xlabel(question1, fontsize=14, fontweight='bold')
        ax.set_ylabel("Count", fontsize=14, fontweight='bold')
        plt.title(custom_title, fontsize=16, fontweight='bold')
        
        # Add a descriptive subtitle
        plt.figtext(0.5, 0.01, f"Cramer's V: {pair_data['cramers_v']:.3f} | p-value: {pair_data['p_value']:.4f}", 
                  fontsize=12, ha='center')
        
        # Add value labels on bars
        for i, (idx, row) in enumerate(contingency.iterrows()):
            cumulative = 0
            for j, val in enumerate(row):
                if val > 0:  # Only add labels for non-zero values
                    percentage = contingency_pct.iloc[i, j]
                    plt.text(i, cumulative + val/2, f"{val}\n({percentage:.1f}%)", 
                           ha='center', va='center', fontsize=10, fontweight='bold',
                           color='white' if j in [0, 1, 4] else 'black')
                cumulative += val
        
        # Add totals at the top of each bar
        for i, total in enumerate(contingency.sum(axis=1)):
            plt.text(i, total + 1, f'Total: {total}', ha='center', va='bottom', 
                   fontsize=11, fontweight='bold')
        
        # Enhance the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], title=question2, 
                title_fontsize=12, fontsize=11, bbox_to_anchor=(1.05, 1),
                loc='upper left', borderaxespad=0.)
        
        plt.tight_layout(pad=2.0, rect=[0, 0.03, 1, 0.97])
        
        # Save the figure to the analyzer's output directory
        filename = f"{field1.lower().replace(' ', '_')}_vs_{field2.lower().replace(' ', '_')}.png"
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        plot_file = os.path.join(analyzer_dir, filename)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    def _write_analysis(self, output_file, results, plot_files):
        """Write analysis results to file."""
        with open(output_file, 'w') as f:
            f.write("=== Head-to-Head Field Comparison Analysis ===\n\n")
            
            if not results:
                f.write("No valid field pairs found for analysis.\n")
                return output_file
            
            f.write(f"Number of field pairs analyzed: {len(results)}\n\n")
            
            # Write analysis for each field pair
            for (field1, field2, question1, question2, custom_title), pair_data in results.items():
                f.write(f"Comparison: '{question1}' vs '{question2}'\n")
                f.write("-" * 60 + "\n\n")
                
                # Statistical analysis
                f.write("Statistical Analysis:\n")
                f.write(f"- Chi-square value: {pair_data['chi2']:.4f}\n")
                f.write(f"- Degrees of freedom: {pair_data['dof']}\n")
                f.write(f"- p-value: {pair_data['p_value']:.4f}")
                
                # Add interpretation of p-value
                if pair_data['p_value'] < 0.001:
                    f.write(" (Highly significant association)\n")
                elif pair_data['p_value'] < 0.01:
                    f.write(" (Very significant association)\n")
                elif pair_data['p_value'] < 0.05:
                    f.write(" (Significant association)\n")
                else:
                    f.write(" (No significant association)\n")
                    
                f.write(f"- Cramer's V: {pair_data['cramers_v']:.4f}")
                
                # Add interpretation of Cramer's V
                if pair_data['cramers_v'] < 0.1:
                    f.write(" (Negligible association)\n")
                elif pair_data['cramers_v'] < 0.3:
                    f.write(" (Weak association)\n")
                elif pair_data['cramers_v'] < 0.5:
                    f.write(" (Moderate association)\n")
                else:
                    f.write(" (Strong association)\n")
                
                # Contingency table
                f.write("\nContingency Table:\n")
                f.write(str(pair_data['contingency']))
                f.write("\n\n")
                
                # Percentage breakdown by rows
                f.write("Percentage Table (by row):\n")
                
                contingency = pair_data['contingency']
                row_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
                
                f.write(str(row_pct.round(1)) + " %\n\n")
                
                # Description of key findings
                f.write("Key Findings:\n")
                
                # Find the highest percentage for each row
                for idx, row in row_pct.iterrows():
                    max_col = row.idxmax()
                    max_pct = row.max()
                    f.write(f"- When '{field1}' is '{idx}', '{field2}' is most commonly '{max_col}' ({max_pct:.1f}%)\n")
                
                f.write("\n")
                
                # Write total datasets with each combination
                total_datasets = pair_data['df'].shape[0]
                f.write(f"Total datasets with both fields: {total_datasets}\n\n")
                
                f.write("=" * 60 + "\n\n")
            
            # Summary of visualizations
            f.write("\nVisualizations Generated:\n")
            for plot_file in plot_files:
                f.write(f"- {os.path.basename(plot_file)}\n")
            
        return output_file
