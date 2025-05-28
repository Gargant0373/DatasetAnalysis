"""
Analyzer for counting datasets by time period.
"""
import os
from typing import Dict, Any, List
from collections import Counter

from src.analyzers.base import Analyzer, register_analyzer


@register_analyzer
class TimeperiodAnalyzer(Analyzer):
    """
    Analyzer that counts datasets by time period.
    """
    
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        return "Time Period Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Analyzes the distribution of datasets across different time periods."
    
    def analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str:
        """
        Analyze the dataset time periods.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, 'timeperiod_analysis.txt')
        
        # Count datasets by time period
        time_periods = Counter()
        
        for dataset_name, dataset_info in data.items():
            if 'period' in dataset_info and dataset_info['period']:
                time_periods[dataset_info['period']] += 1
        
        # Write results to output file
        with open(output_file, 'w') as f:
            f.write("=== Time Period Analysis ===\n\n")
            
            f.write("Datasets by Time Period:\n")
            f.write("-" * 50 + "\n")
            
            for period, count in sorted(time_periods.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(data)) * 100
                f.write(f"{period}: {count} datasets ({percentage:.2f}%)\n")
        
        return output_file
