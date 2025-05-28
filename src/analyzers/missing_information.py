"""
Analyzer for detecting missing information in dataset fields.
"""
import os
from typing import Dict, Any, List

from src.analyzers.base import Analyzer, register_analyzer


@register_analyzer
class MissingInformationAnalyzer(Analyzer):
    """
    Analyzer that detects missing information in dataset fields.
    """
    
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        return "Missing Information Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Analyzes datasets for missing information patterns and reports on fields and datasets with high missing information rates."
    
    def analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str:
        """
        Analyze the dataset for missing information.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, 'missing_information_analysis.txt')
        
        # Define patterns for missing information
        missing_patterns = ["N/A", "No information", "Not applicable", "Unknown", "Unsure", "nan"]
        
        # Count occurrences of missing information for each field
        field_missing_counts = {}
        dataset_missing_counts = {}
        
        for dataset_name, dataset_info in data.items():
            dataset_missing_count = 0
            
            for field, value in dataset_info.items():
                if field not in field_missing_counts:
                    field_missing_counts[field] = 0
                
                # Check if the value contains any missing information pattern
                if value is None or (isinstance(value, str) and any(pattern.lower() in str(value).lower() for pattern in missing_patterns)):
                    field_missing_counts[field] += 1
                    dataset_missing_count += 1
            
            dataset_missing_counts[dataset_name] = dataset_missing_count
        
        # Write results to output file
        with open(output_file, 'w') as f:
            f.write("=== Missing Information Analysis ===\n\n")
            
            # Summary of missing information by field
            f.write("Missing Information by Field:\n")
            f.write("-" * 50 + "\n")
            for field, count in sorted(field_missing_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(data)) * 100
                f.write(f"{field}: {count} ({percentage:.2f}%)\n")
            
            f.write("\n")
            
            # Summary of missing information by dataset
            f.write("Missing Information by Dataset:\n")
            f.write("-" * 50 + "\n")
            for dataset, count in sorted(dataset_missing_counts.items(), key=lambda x: x[1], reverse=True):
                total_fields = len(data[dataset])
                percentage = (count / total_fields) * 100
                f.write(f"{dataset}: {count}/{total_fields} fields ({percentage:.2f}%)\n")
        
        return output_file
