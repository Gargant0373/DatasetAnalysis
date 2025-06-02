"""
Analyzer for creating scatter plots of dataset citation counts for specific venues.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from collections import defaultdict
from analyzers.base import register_analyzer, Analyzer

@register_analyzer
class PaperCitationAnalyzer(Analyzer):
    """Analyzer that creates scatter plots of dataset citation counts for papers."""
    
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        return "Paper Citation Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Creates scatter plots showing citation counts papers"
    
    def analyze(self, data, output_dir='outputs', venue="TPAMI"):
        """
        Analyze citation counts for datasets from a specific venue and create a scatter plot.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            venue: The venue to filter by (default: TPAMI)
            
        Returns:
            Path to the output file
        """
        # Get analyzer-specific output directory
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        
        venue_slug = venue.lower().replace(' ', '_')
        output_file = os.path.join(analyzer_dir, f'{venue_slug}_citation_analysis.txt')
        plot_file = os.path.join(analyzer_dir, f'{venue_slug}_citation_plot.png')
        
        # Extract dataset names and citation counts
        dataset_names = []
        citation_counts = []
        years = defaultdict(list)
        
        for dataset_name, dataset_info in data.items():
            # Check if the dataset is from the specified venue
            if 'Done for?' in dataset_info and dataset_info['Done for?'] == venue:
                if 'DS Citation' in dataset_info and dataset_info['DS Citation']:
                    try:
                        citation_count = int(dataset_info['DS Citation'])
                        dataset_names.append(dataset_name)
                        citation_counts.append(citation_count)
                        
                        # Try to extract year from period information
                        if 'period' in dataset_info and dataset_info['period']:
                            period = str(dataset_info['period'])
                            # Try to extract a year from the period string
                            for word in period.split():
                                if word.isdigit() and len(word) == 4:  # Check if it's a 4-digit year
                                    years[word].append((dataset_name, citation_count))
                                    break
                    except (ValueError, TypeError):
                        # Skip datasets with invalid citation counts
                        continue
        
        if not dataset_names:
            # No datasets found for this venue
            with open(output_file, 'w') as f:
                f.write(f"=== Citation Count Analysis for {venue} ===\n\n")
                f.write(f"No datasets found for venue: {venue}\n")
            return output_file
        
        # Sort datasets by citation count for the analysis
        sorted_indices = np.argsort(citation_counts)[::-1]  # Sort in descending order
        sorted_dataset_names = [dataset_names[i] for i in sorted_indices]
        sorted_citation_counts = [citation_counts[i] for i in sorted_indices]
        
        # Create a simple, compact scatter plot
        plt.figure(figsize=(8, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Generate x-coordinates (just indices)
        x = np.arange(len(citation_counts))
        
        # Create scatter plot with logarithmic scale
        plt.scatter(x, citation_counts, 
                   s=80,  # marker size
                   alpha=0.7,  # transparency
                   c=citation_counts,  # color by citation count
                   cmap='viridis',  # colormap
                   edgecolors='white',  # white edge for better visibility
                   linewidths=0.5)  # thin white edge
        
        # Use logarithmic scale on y-axis for better visualization of the distribution
        plt.yscale('log')
        
        # Hide x-ticks since they don't represent anything meaningful
        plt.xticks([])
        
        # Remove borders and add minimal grid lines for readability
        plt.box(True)
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add labels and title with enhanced formatting
        plt.xlabel(f'Datasets ({len(citation_counts)} total)', fontsize=12)
        plt.ylabel('Citation Count (log scale)', fontsize=12)
        plt.title(f'Citation Distribution of {venue} Dataset Papers', 
                 fontsize=14, pad=10)
        
        # Add mean and median lines
        mean_citations = np.mean(citation_counts)
        median_citations = np.median(citation_counts)
        
        plt.axhline(y=mean_citations, color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_citations:.1f}')
        plt.axhline(y=median_citations, color='g', linestyle='--', alpha=0.7,
                   label=f'Median: {median_citations:.1f}')
        
        # Add legend
        plt.legend(loc='lower right')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Write analysis to output file
        with open(output_file, 'w') as f:
            f.write(f"=== Citation Count Analysis for {venue} ===\n\n")
            
            # Basic statistics
            f.write("Citation Count Statistics:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Total {venue} datasets analyzed: {len(citation_counts)}\n")
            f.write(f"Mean citation count: {mean_citations:.2f}\n")
            f.write(f"Median citation count: {median_citations:.2f}\n")
            f.write(f"Standard deviation: {np.std(citation_counts):.2f}\n")
            f.write(f"Minimum: {min(citation_counts)}\n")
            f.write(f"Maximum: {max(citation_counts)}\n\n")
            
            # List all datasets
            f.write(f"All {venue} Datasets by Citation Count:\n")
            f.write("-" * 50 + "\n")
            for i in range(len(sorted_dataset_names)):
                f.write(f"{i+1}. {sorted_dataset_names[i]}: {sorted_citation_counts[i]} citations\n")
            
            f.write("\n")
            
            # Identify potential influential papers
            # Consider a paper influential if it has more than 2x the median citations
            influence_threshold = 2 * median_citations
            influential_papers = [(name, count) for name, count in 
                                 zip(sorted_dataset_names, sorted_citation_counts) 
                                 if count > influence_threshold]
            
            if influential_papers:
                f.write("Potentially Influential Dataset Papers:\n")
                f.write("-" * 50 + "\n")
                f.write("(Citation count > 2x median)\n")
                for i, (name, count) in enumerate(influential_papers):
                    f.write(f"{i+1}. {name}: {count} citations\n")
                
                f.write("\n")
                
            # Identify statistical outliers using z-score method
            citation_array = np.array(citation_counts)
            z_scores = np.abs(stats.zscore(citation_array))
            outlier_threshold = 3.0  # Common threshold for identifying outliers
            outlier_indices = np.where(z_scores > outlier_threshold)[0]
            
            if len(outlier_indices) > 0:
                f.write("Statistical Outliers (Z-score > 3.0):\n")
                f.write("-" * 50 + "\n")
                f.write("These datasets have citation counts that deviate significantly from the normal distribution:\n\n")
                
                for idx in outlier_indices:
                    name = dataset_names[idx]
                    count = citation_counts[idx]
                    z_score = z_scores[idx]
                    f.write(f"- {name}: {count} citations (Z-score: {z_score:.2f})\n")
                
                f.write("\n")
            
            # Add references to the generated visualizations
            f.write("\nVisualizations Generated:\n")
            f.write(f"- {os.path.basename(plot_file)}: Scatter plot of {venue} citation counts\n")
        
        return output_file
