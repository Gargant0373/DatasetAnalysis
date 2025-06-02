import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from analyzers.base import register_analyzer

@register_analyzer
class CitationMissingCorrelationAnalyzer:
    """Analyzer that correlates citation counts with missing information percentage."""
    
    @staticmethod
    def get_name():
        return "citation_vs_missing"
    
    @staticmethod
    def get_description():
        return "Correlates dataset citation counts with percentage of missing information"
    
    @staticmethod
    def analyze(data, output_dir='outputs'):
        """
        Analyze correlation between citation counts and missing information.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Create analyzer-specific output directory
        analyzer_dir = os.path.join(output_dir, CitationMissingCorrelationAnalyzer.get_name())
        if not os.path.exists(analyzer_dir):
            os.makedirs(analyzer_dir)
        
        output_file = os.path.join(analyzer_dir, 'citation_vs_missing_analysis.txt')
        plot_file = os.path.join(analyzer_dir, 'citation_vs_missing_plot.png')
        
        # Standard fields that aren't considered missing info fields
        standard_fields = {'name', 'period', 'citation_sum', 'status', 
                          'DS DOI', 'DS Citation', 'Done for?', 'Empty', 
                          'Outcome', 'Link to Data', 'Notes', 'Additional Links'}
        
        dataset_missing = {}
        dataset_citations = {}
        
        # Calculate missing information percentage for each dataset
        for dataset_name, dataset_info in data.items():
            # Get citation count
            citation_count = 0
            if 'citation_sum' in dataset_info and dataset_info['citation_sum']:
                try:
                    citation_count = int(dataset_info['citation_sum'])
                except (ValueError, TypeError):
                    # If citation_sum is not a valid integer, use 0
                    pass
            
            dataset_citations[dataset_name] = citation_count
            
            # Count total fields and missing fields
            total_fields = 0
            missing_fields = 0
            
            for field, value in dataset_info.items():
                if field not in standard_fields:
                    total_fields += 1
                    # Check if field is missing
                    is_missing = False
                    if value is None or value == "" or value == "N/A" or "No information" in str(value) or "Not applicable" in str(value):
                        is_missing = True
                    
                    if is_missing:
                        missing_fields += 1
            
            # Calculate missing percentage
            missing_percentage = 0
            if total_fields > 0:
                missing_percentage = (missing_fields / total_fields) * 100
            
            dataset_missing[dataset_name] = missing_percentage
        
        # Prepare data for correlation analysis
        datasets = list(dataset_missing.keys())
        missing_percentages = [dataset_missing[d] for d in datasets]
        citation_counts = [dataset_citations[d] for d in datasets]
        
        # Calculate correlation coefficients
        pearson_r, pearson_p = stats.pearsonr(citation_counts, missing_percentages)
        spearman_r, spearman_p = stats.spearmanr(citation_counts, missing_percentages)
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(citation_counts, missing_percentages, alpha=0.7)
        
        # Add trend line
        z = np.polyfit(citation_counts, missing_percentages, 1)
        p = np.poly1d(z)
        plt.plot(sorted(citation_counts), p(sorted(citation_counts)), "r--", alpha=0.7)
        
        # Add labels for each point
        for i, dataset in enumerate(datasets):
            plt.annotate(dataset, 
                         (citation_counts[i], missing_percentages[i]),
                         textcoords="offset points",
                         xytext=(0,5), 
                         ha='center',
                         fontsize=8)
        
        plt.xlabel('Citation Count')
        plt.ylabel('Missing Information (%)')
        plt.title('Correlation between Citation Count and Missing Information')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Write results to output file
        with open(output_file, 'w') as f:
            f.write("=== Citation vs Missing Information Analysis ===\n\n")
            
            f.write(f"Pearson correlation coefficient: {pearson_r:.4f} (p-value: {pearson_p:.4f})\n")
            f.write(f"Spearman rank correlation: {spearman_r:.4f} (p-value: {spearman_p:.4f})\n\n")
            
            if pearson_p < 0.05:
                if pearson_r > 0:
                    f.write("There is a statistically significant POSITIVE correlation between citation count and missing information.\n")
                    f.write("This suggests that more highly cited datasets tend to have MORE missing information.\n\n")
                else:
                    f.write("There is a statistically significant NEGATIVE correlation between citation count and missing information.\n")
                    f.write("This suggests that more highly cited datasets tend to have LESS missing information.\n\n")
            else:
                f.write("There is NO statistically significant correlation between citation count and missing information.\n\n")
            
            f.write("Top 5 Datasets with Highest Citation Counts:\n")
            f.write("-" * 50 + "\n")
            for dataset, count in sorted(dataset_citations.items(), key=lambda x: x[1], reverse=True)[:5]:
                f.write(f"{dataset}: {count} citations, {dataset_missing[dataset]:.2f}% missing information\n")
            
            f.write("\nTop 5 Datasets with Lowest Missing Information:\n")
            f.write("-" * 50 + "\n")
            for dataset, missing in sorted(dataset_missing.items(), key=lambda x: x[1])[:5]:
                f.write(f"{dataset}: {missing:.2f}% missing information, {dataset_citations[dataset]} citations\n")
            
            f.write("\nFull Dataset Information:\n")
            f.write("-" * 50 + "\n")
            f.write("Dataset, Citation Count, Missing Information (%)\n")
            for dataset in datasets:
                f.write(f"{dataset}, {dataset_citations[dataset]}, {dataset_missing[dataset]:.2f}%\n")
            
            f.write(f"\nA scatter plot has been saved to: {plot_file}\n")
        
        return output_file