"""
Analyzer for creating scatter plots of dataset citation counts for specific venues.
Analyzes both dataset publication paper citations and dataset usage citations.
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
        return "Analyzes both dataset publication paper citations (Citation Sum) and dataset usage citations (DS Citation)"
    
    def analyze(self, data, output_dir='outputs', venue="TPAMI"):
        """
        Analyze citation counts for datasets from a specific venue and create scatter plots.
        Analyzes both:
        - Citation Sum: Citations of the original dataset publication paper
        - DS Citation: Cumulative citations from papers that reference/use the dataset
        
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
        plot_file_publication = os.path.join(analyzer_dir, f'{venue_slug}_publication_citation_plot.png')
        plot_file_usage = os.path.join(analyzer_dir, f'{venue_slug}_usage_citation_plot.png')
        
        # Extract dataset names and both types of citation counts
        # Use dictionaries to track unique datasets and their highest citation counts
        unique_datasets = {}  # dataset_name -> {pub_citations, usage_citations, full_name}
        years = defaultdict(list)
        
        for dataset_name, dataset_info in data.items():
            # Extract the base dataset name (part before the comma)
            base_dataset_name = dataset_name.split(',')[0].strip()
            
            # Get Citation Sum (publication paper citations)
            publication_citations = None
            if 'citation_sum' in dataset_info and dataset_info['citation_sum']:
                try:
                    publication_citations = int(dataset_info['citation_sum'])
                except (ValueError, TypeError):
                    publication_citations = None
            
            # Get DS Citation (dataset usage citations)
            usage_citations = None
            if 'DS Citation' in dataset_info and dataset_info['DS Citation']:
                try:
                    usage_citations = int(dataset_info['DS Citation'])
                except (ValueError, TypeError):
                    usage_citations = None
            
            # Include dataset if we have at least one type of citation data
            if publication_citations is not None or usage_citations is not None:
                # If this is the first time we see this dataset, or if we have better data
                if base_dataset_name not in unique_datasets:
                    unique_datasets[base_dataset_name] = {
                        'pub_citations': publication_citations if publication_citations is not None else 0,
                        'usage_citations': usage_citations if usage_citations is not None else 0,
                        'full_name': dataset_name
                    }
                else:
                    # Keep the highest citation counts for each type
                    current = unique_datasets[base_dataset_name]
                    if publication_citations is not None and publication_citations > current['pub_citations']:
                        current['pub_citations'] = publication_citations
                    if usage_citations is not None and usage_citations > current['usage_citations']:
                        current['usage_citations'] = usage_citations
                
                # Try to extract year from period information
                if 'period' in dataset_info and dataset_info['period']:
                    period = str(dataset_info['period'])
                    # Try to extract a year from the period string
                    for word in period.split():
                        if word.isdigit() and len(word) == 4:  # Check if it's a 4-digit year
                            years[word].append((base_dataset_name, publication_citations, usage_citations))
                            break
          # Convert unique datasets back to lists
        dataset_names = list(unique_datasets.keys())
        publication_citation_counts = [unique_datasets[name]['pub_citations'] for name in dataset_names]
        usage_citation_counts = [unique_datasets[name]['usage_citations'] for name in dataset_names]
        
        if not dataset_names:
            # No datasets found for this venue
            with open(output_file, 'w') as f:
                f.write(f"=== Citation Count Analysis for {venue} ===\n\n")
                f.write(f"No datasets found for venue: {venue}\n")
            return output_file
        
        # Create analysis and plots for both citation types
        self._create_citation_analysis(
            dataset_names, publication_citation_counts, usage_citation_counts,
            output_file, plot_file_publication, plot_file_usage, venue, analyzer_dir, venue_slug
        )
        return output_file
    
    def _create_citation_analysis(self, dataset_names, publication_citations, usage_citations, 
                                 output_file, plot_file_publication, plot_file_usage, venue, analyzer_dir, venue_slug):
        """Create comprehensive citation analysis for both citation types."""
          # Filter out zero values for meaningful statistics
        valid_pub_citations = [c for c in publication_citations if c > 0]
        valid_usage_citations = [c for c in usage_citations if c > 0]
        
        # Create plots for both citation types
        if valid_pub_citations:
            self._create_scatter_plot(
                dataset_names, publication_citations, plot_file_publication,
                f"Dataset Publication Paper Citations - {venue}",
                "Citation Count of Original Dataset Papers (log scale)"
            )
        
        if valid_usage_citations:
            self._create_scatter_plot(
                dataset_names, usage_citations, plot_file_usage,
                f"Dataset Usage Citations - {venue}", 
                "Citations from Papers Using Dataset (log scale)"
            )
        
        # Create combined comparison plot if we have both types of data
        if valid_pub_citations and valid_usage_citations:
            plot_file_combined = os.path.join(analyzer_dir, f'{venue_slug}_combined_citation_plot.png')
            self._create_combined_plot(
                dataset_names, publication_citations, usage_citations, plot_file_combined, venue            )
        
        # Write comprehensive analysis to output file
        with open(output_file, 'w') as f:
            f.write(f"=== Comprehensive Citation Analysis for {venue} ===\n\n")
            f.write("This analysis covers two types of citations:\n")
            f.write("1. Publication Paper Citations (Citation Sum): Citations of the original dataset publication paper\n")
            f.write("2. Dataset Usage Citations (DS Citation): Cumulative citations from papers that reference/use the dataset\n\n")
            f.write(f"DATASET DEDUPLICATION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total unique dataset papers analyzed: {len(dataset_names)}\n")
            f.write("Note: Datasets appearing across multiple time periods have been deduplicated.\n")
            f.write("For each dataset, the highest citation count across all periods is used.\n\n")
            
            # Publication Paper Citations Analysis
            if valid_pub_citations:
                f.write("=" * 60 + "\n")
                f.write("PUBLICATION PAPER CITATIONS ANALYSIS\n")
                f.write("=" * 60 + "\n")
                f.write("(Citations of the original dataset publication papers)\n\n")
                
                self._write_citation_statistics(f, dataset_names, publication_citations, "publication paper")
                
            # Dataset Usage Citations Analysis  
            if valid_usage_citations:
                f.write("\n" + "=" * 60 + "\n")
                f.write("DATASET USAGE CITATIONS ANALYSIS\n")
                f.write("=" * 60 + "\n")
                f.write("(Citations from papers that reference/use the dataset)\n\n")
                
                self._write_citation_statistics(f, dataset_names, usage_citations, "dataset usage")
            
            # Comparative Analysis
            if valid_pub_citations and valid_usage_citations:
                f.write("\n" + "=" * 60 + "\n")
                f.write("COMPARATIVE ANALYSIS\n")
                f.write("=" * 60 + "\n")
                
                # Calculate correlations
                non_zero_pairs = [(pub, usage) for pub, usage in zip(publication_citations, usage_citations) 
                                 if pub > 0 and usage > 0]
                
                if len(non_zero_pairs) > 1:
                    pub_vals, usage_vals = zip(*non_zero_pairs)
                    correlation = np.corrcoef(pub_vals, usage_vals)[0, 1]
                    f.write(f"Correlation between publication citations and usage citations: {correlation:.3f}\n")
                    if correlation > 0.7:
                        f.write("-> Strong positive correlation: Highly cited dataset papers tend to be used more\n")
                    elif correlation > 0.3:
                        f.write("-> Moderate positive correlation: Some relationship between paper impact and dataset usage\n")
                    else:
                        f.write("-> Weak correlation: Dataset usage may be independent of original paper citations\n")
                    f.write("\nDatasets with notable citation patterns:\n")
                f.write("-" * 50 + "\n")
                for i, name in enumerate(dataset_names):
                    pub_cit = publication_citations[i]
                    usage_cit = usage_citations[i]
                    if pub_cit > 0 and usage_cit > 0:
                        ratio = usage_cit / pub_cit if pub_cit > 0 else 0
                        f.write(f"{name}:\n")
                        f.write(f"  Publication citations: {pub_cit:,}\n")
                        f.write(f"  Usage citations: {usage_cit:,}\n")
                        f.write(f"  Usage/Publication ratio: {ratio:.2f}\n\n")
            
            # Add references to visualizations
            f.write("\nVisualizations Generated:\n")
            f.write("-" * 30 + "\n")
            if valid_pub_citations:
                f.write(f"- {os.path.basename(plot_file_publication)}: Publication paper citation distribution\n")
            if valid_usage_citations:
                f.write(f"- {os.path.basename(plot_file_usage)}: Dataset usage citation distribution\n")
            if valid_pub_citations and valid_usage_citations:
                plot_file_combined = os.path.join(analyzer_dir, f'{venue_slug}_combined_citation_plot.png')
                f.write(f"- {os.path.basename(plot_file_combined)}: Combined comparison of both citation types\n")
    
    def _create_scatter_plot(self, dataset_names, citation_counts, plot_file, title, ylabel):
        """Create a scatter plot for citation counts."""
        valid_citations = [(name, count) for name, count in zip(dataset_names, citation_counts) if count > 0]
        
        if not valid_citations:
            return
            
        names, counts = zip(*valid_citations)
        
        # Sort datasets by citation count for the analysis
        sorted_indices = np.argsort(counts)[::-1]  # Sort in descending order
        sorted_counts = [counts[i] for i in sorted_indices]
        
        # Create a simple, compact scatter plot
        plt.figure(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Generate x-coordinates (just indices)
        x = np.arange(len(sorted_counts))
        
        # Create scatter plot with logarithmic scale
        plt.scatter(x, sorted_counts, 
                   s=80,  # marker size
                   alpha=0.7,  # transparency
                   c=sorted_counts,  # color by citation count
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
        plt.xlabel(f'Datasets ({len(sorted_counts)} total)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14, pad=10)
        
        # Add mean and median lines
        mean_citations = np.mean(sorted_counts)
        median_citations = np.median(sorted_counts)
        
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
    
    def _write_citation_statistics(self, f, dataset_names, citation_counts, citation_type):
        """Write statistical analysis for a specific citation type."""
        # Filter valid citations
        valid_data = [(name, count) for name, count in zip(dataset_names, citation_counts) if count > 0]
        
        if not valid_data:
            f.write(f"No valid {citation_type} citation data found.\n")
            return
            
        names, counts = zip(*valid_data)
        
        # Sort datasets by citation count
        sorted_indices = np.argsort(counts)[::-1]
        sorted_names = [names[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]        # Basic statistics
        f.write(f"{citation_type.title()} Citation Statistics:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total unique datasets with {citation_type} citations: {len(counts)}\n")
        f.write(f"Mean citation count: {np.mean(counts):.2f}\n")
        f.write(f"Median citation count: {np.median(counts):.2f}\n")
        f.write(f"Standard deviation: {np.std(counts):.2f}\n")
        f.write(f"Minimum: {min(counts):,}\n")
        f.write(f"Maximum: {max(counts):,}\n\n")
        
        # List top datasets
        f.write(f"Top {citation_type.title()} Citations:\n")
        f.write("-" * 50 + "\n")
        for i in range(min(10, len(sorted_names))):  # Show top 10
            f.write(f"{i+1}. {sorted_names[i]}: {sorted_counts[i]:,} citations\n")
        
        f.write("\n")
        
        # Identify potential influential papers
        median_citations = np.median(counts)
        influence_threshold = 2 * median_citations
        influential_papers = [(name, count) for name, count in zip(sorted_names, sorted_counts) 
                             if count > influence_threshold]
        
        if influential_papers:
            f.write(f"High-Impact {citation_type.title()} Citations:\n")
            f.write("-" * 50 + "\n")
            f.write(f"(Citation count > 2x median = {influence_threshold:.1f})\n")
            for i, (name, count) in enumerate(influential_papers):
                f.write(f"{i+1}. {name}: {count:,} citations\n")
            
            f.write("\n")
            
    def _create_combined_plot(self, dataset_names, publication_citations, usage_citations, plot_file, venue):
        """Create a combined plot showing both publication and usage citations."""
        # Filter to datasets that have both types of citations
        combined_data = []
        for i, name in enumerate(dataset_names):
            pub_cit = publication_citations[i]
            usage_cit = usage_citations[i]
            if pub_cit > 0 and usage_cit > 0:
                combined_data.append((name, pub_cit, usage_cit))
        
        if not combined_data:
            return
        
        # Sort by total citations (publication + usage)
        combined_data.sort(key=lambda x: x[1] + x[2], reverse=True)
        
        names, pub_counts, usage_counts = zip(*combined_data)
        
        # Create figure for bar chart only
        plt.figure(figsize=(14, 8))
        
        x_pos = np.arange(len(names))
        
        # Side-by-side bars
        width = 0.35
        plt.bar(x_pos - width/2, pub_counts, width, label='Publication Citations', 
               alpha=0.8, color='skyblue')
        plt.bar(x_pos + width/2, usage_counts, width, label='Usage Citations', 
               alpha=0.8, color='lightcoral')
        
        plt.yscale('log')
        plt.xlabel('Datasets')
        plt.ylabel('Citation Count (log scale)')
        plt.title(f'Publication vs Usage Citations - {venue}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(x_pos, names, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
