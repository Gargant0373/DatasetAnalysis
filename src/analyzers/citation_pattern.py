import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from analyzers.base import register_analyzer, Analyzer

@register_analyzer
class CitationPatternAnalyzer(Analyzer):
    """Analyzer for visualizing citation patterns over time."""
    
    def get_name(self) -> str:
        """Get the name of the analyzer."""
        return "Citation Pattern Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Analyzes citation patterns over years and creates scatter plots"
    
    def analyze(self, data, output_dir='outputs', venue="CVPR"):
        """
        Create scatter plots showing citation patterns over years.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            venue: The venue to analyze (default: CVPR)
            
        Returns:
            Path to the output file
        """
        # Get analyzer-specific output directory
        analyzer_dir = self.get_analyzer_output_dir(output_dir)

        # Load all paper data
        papers_data = self._load_paper_data()
        if papers_data is None or papers_data.empty:
            return analyzer_dir
            
        # Filter by venue if specified
        if venue:
            papers_data = papers_data[papers_data['Source title'] == venue]
            
        if papers_data.empty:
            print(f"No papers found for venue: {venue}")
            return analyzer_dir
            
        # Create plots
        output_file = os.path.join(analyzer_dir, f"{venue.lower()}_citation_pattern.png")
        self._create_citation_scatter_plot(papers_data, output_file, venue)
        
        # Create plots by period
        if 'Period' in papers_data.columns:
            for period in papers_data['Period'].unique():
                period_data = papers_data[papers_data['Period'] == period]
                if not period_data.empty:
                    period_output_file = os.path.join(analyzer_dir, f"{venue.lower()}_period_{period}_citation_pattern.png")
                    self._create_citation_scatter_plot(period_data, period_output_file, f"{venue} - Period {period}")
        
        return analyzer_dir
    
    def _load_paper_data(self):
        """Load paper data from CSV."""
        csv_path = os.path.join('data', 'tab1.csv')
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return None
            
        try:
            # Read the CSV file
            df = pd.read_csv(csv_path)
            
            # Assuming the column names in your CSV match these
            # If they don't, adjust accordingly
            if 'Year' in df.columns and 'Cited by' in df.columns:
                # Convert 'Cited by' and 'Year' columns to numeric
                df['Cited by'] = pd.to_numeric(df['Cited by'], errors='coerce')
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                
                # Drop rows with NaN values in important columns
                df = df.dropna(subset=['Year', 'Cited by'])
                
                return df
            else:
                print(f"Required columns not found in CSV file")
                return None
        except Exception as e:
            print(f"Error loading paper data: {e}")
            return None
    
    def _create_citation_scatter_plot(self, df, output_path, title):
        """Create a scatter plot of citations per year."""
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(
            df['Year'], 
            df['Cited by'],
            alpha=0.7,
            s=50  # Size of dots
        )
        
        # Add trend line
        x = df['Year'].values
        y = df['Cited by'].values
        if len(x) > 1:
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            trend_x = np.array([min(x), max(x)])
            trend_y = slope * trend_x + intercept
            plt.plot(trend_x, trend_y, 'r--', linewidth=2, alpha=0.7,
                    label=f'Trend (R²={r_value**2:.2f})')
        
        # Set axis labels and title
        plt.xlabel('Publication Year')
        plt.ylabel('Citation Count')
        plt.title(f'Citation Pattern - {title}')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend
        plt.legend()
        
        # Ensure years are formatted as integers on x-axis
        plt.xticks(sorted(df['Year'].unique()))
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
