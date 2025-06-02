"""
Analyzer for detecting missing information in dataset fields.
"""
import os
import re
from typing import Dict, Any, List
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np

from analyzers.base import Analyzer, register_analyzer

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
        # Get analyzer-specific output directory
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        
        output_file = os.path.join(analyzer_dir, 'missing_information_analysis.txt')
        
        # Define patterns for missing information
        missing_patterns = ["N/A", "No information", "Not applicable", "Unknown", "Unsure", "nan"]
        
        # Count occurrences of missing information for each field
        field_missing_counts = {}
        dataset_missing_counts = {}
        
        # Period-related analysis
        periods = set()
        datasets_by_period = defaultdict(list)
        missing_by_period = defaultdict(int)
        total_fields_by_period = defaultdict(int)
        field_missing_by_period = defaultdict(lambda: defaultdict(int))
        
        # First pass: identify all periods and datasets by period
        for dataset_name, dataset_info in data.items():
            period = dataset_info.get('period', 'Unknown')
            
            # Handle empty periods
            if not period or isinstance(period, float) or str(period).strip() == '':
                period = 'Unknown'
                
            periods.add(period)
            datasets_by_period[period].append(dataset_name)
        
        # Second pass: analyze missing information
        for dataset_name, dataset_info in data.items():
            dataset_missing_count = 0
            period = dataset_info.get('period', 'Unknown')
            
            # Handle empty periods
            if not period or isinstance(period, float) or str(period).strip() == '':
                period = 'Unknown'
            
            # Count total fields for this period (excluding 'period' field itself)
            dataset_field_count = len(dataset_info) - 1  # Subtract 1 for the period field
            total_fields_by_period[period] += dataset_field_count
            
            for field, value in dataset_info.items():
                # Skip counting period field itself in missing analysis
                if field == 'period':
                    continue
                    
                if field not in field_missing_counts:
                    field_missing_counts[field] = 0
                
                # Check if the value contains any missing information pattern
                is_missing = value is None or (isinstance(value, str) and any(pattern.lower() in str(value).lower() for pattern in missing_patterns))
                
                if is_missing:
                    field_missing_counts[field] += 1
                    dataset_missing_count += 1
                    field_missing_by_period[period][field] += 1
                    missing_by_period[period] += 1
            
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
                total_fields = len(data[dataset]) - 1  # Subtract 1 for period field
                percentage = (count / total_fields) * 100 if total_fields > 0 else 0
                f.write(f"{dataset}: {count}/{total_fields} fields ({percentage:.2f}%)\n")
            
            f.write("\n")
            
            # Summary of missing information by period
            f.write("Missing Information by Period:\n")
            f.write("-" * 50 + "\n")
            
            # Sort periods by missing percentage for better readability
            sorted_periods = sorted(
                periods, 
                key=lambda p: (missing_by_period[p] / total_fields_by_period[p] * 100) if total_fields_by_period[p] > 0 else 0,
                reverse=True
            )
            
            for period in sorted_periods:
                dataset_count = len(datasets_by_period[period])
                missing_count = missing_by_period[period]
                total_count = total_fields_by_period[period]
                
                if total_count > 0:
                    percentage = (missing_count / total_count) * 100
                    f.write(f"{period}: {missing_count}/{total_count} fields missing ({percentage:.2f}%) across {dataset_count} datasets\n")
                    
                    # For each period, list the top missing fields
                    f.write("  Top missing fields in this period:\n")
                    
                    # Sort fields by missing count in this period
                    period_fields = sorted(
                        field_missing_by_period[period].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]  # Show top 5 missing fields
                    
                    for field, count in period_fields:
                        if count > 0:
                            field_percentage = (count / dataset_count) * 100
                            f.write(f"  - {field}: missing in {count}/{dataset_count} datasets ({field_percentage:.2f}%)\n")
                    
                    f.write("\n")
        
        # Create visualizations
        self._create_visualizations(field_missing_counts, missing_by_period, field_missing_by_period, 
                                  datasets_by_period, total_fields_by_period, periods, output_dir)
        
        return output_file
    
    def _create_visualizations(self, field_missing_counts, missing_by_period, field_missing_by_period,
                              datasets_by_period, total_fields_by_period, periods, output_dir):
        """Create visualizations for missing information analysis."""
        # 1. Create bar chart for top missing fields overall
        self._plot_top_missing_fields(field_missing_counts, output_dir)
        
        # 2. Create bar chart for missing information by period
        self._plot_missing_by_period(missing_by_period, total_fields_by_period, periods, output_dir)
        
        # 3. Create bar chart for top missing fields by period
        self._plot_top_missing_by_period(field_missing_by_period, datasets_by_period, periods, output_dir)
        
    def _plot_top_missing_fields(self, field_missing_counts, output_dir):
        """Plot top missing fields across all datasets."""
        plt.figure(figsize=(12, 8))
        
        # Get top 10 missing fields
        top_fields = sorted(field_missing_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        fields = [field for field, _ in top_fields]
        counts = [count for _, count in top_fields]
        
        # Create the bar chart
        bars = plt.bar(fields, counts, color='skyblue')
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        # Customize the chart
        plt.title('Top 10 Missing Fields Across All Datasets', fontsize=14)
        plt.xlabel('Field Name', fontsize=12)
        plt.ylabel('Number of Datasets Missing Field', fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        
        # Save the figure
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        output_path = os.path.join(analyzer_dir, 'missing_fields_overall.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_missing_by_period(self, missing_by_period, total_fields_by_period, periods, output_dir):
        """Plot missing information percentage by period."""
        plt.figure(figsize=(10, 6))
        
        # Calculate missing percentages for each period
        sorted_periods = sorted(periods)
        period_labels = [str(p) for p in sorted_periods]
        percentages = []
        
        for period in sorted_periods:
            if total_fields_by_period[period] > 0:
                percentage = (missing_by_period[period] / total_fields_by_period[period]) * 100
                percentages.append(percentage)
            else:
                percentages.append(0)
        
        # Create the bar chart with period labels
        bars = plt.bar(range(len(sorted_periods)), percentages, color='lightcoral')
        
        # Add value labels on top of each bar
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            # Add period labels in the bar itself
            plt.text(bar.get_x() + bar.get_width()/2., height/2,
                    period_labels[i], ha='center', va='center', 
                    fontsize=10, color='white', fontweight='bold')
        
        # Customize the chart
        plt.title('Missing Information Percentage by Period', fontsize=14)
        plt.ylabel('Missing Information Percentage', fontsize=12)
        plt.ylim(0, max(percentages) * 1.15)  # Add some space for labels
        plt.xticks([])  # Remove x-axis ticks
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the figure
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        output_path = os.path.join(analyzer_dir, 'missing_by_period.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_top_missing_by_period(self, field_missing_by_period, datasets_by_period, periods, output_dir):
        """Plot top missing fields for each period."""
        # Get top 3 periods with most datasets
        top_periods = sorted(periods, key=lambda p: len(datasets_by_period[p]), reverse=True)[:3]
        
        # For each of the top 3 periods, get the top 5 missing fields
        period_top_fields = {}
        all_top_fields = set()
        
        for period in top_periods:
            # Get top 5 missing fields for this period
            if period in field_missing_by_period:
                top_fields = sorted(field_missing_by_period[period].items(), 
                                   key=lambda x: x[1], reverse=True)[:5]
                period_top_fields[period] = top_fields
                
                # Add to set of all top fields
                for field, _ in top_fields:
                    all_top_fields.add(field)
        
        # Create figure
        plt.figure(figsize=(14, 8))
        
        # Set up bar chart
        x = np.arange(len(all_top_fields))
        width = 0.25  # width of bars
        
        # Sort fields alphabetically for consistent ordering
        all_top_fields = sorted(all_top_fields)
        
        # Create bars for each period
        for i, period in enumerate(top_periods):
            # Get percentages for each field in this period
            percentages = []
            
            for field in all_top_fields:
                dataset_count = len(datasets_by_period[period])
                field_count = field_missing_by_period[period].get(field, 0)
                
                if dataset_count > 0:
                    percentage = (field_count / dataset_count) * 100
                else:
                    percentage = 0
                
                percentages.append(percentage)
            
            # Plot the bars for this period
            offset = width * (i - 1)
            bars = plt.bar(x + offset, percentages, width, 
                          label=f'Period {period}', 
                          alpha=0.7)
        
        # Customize the chart
        plt.xlabel('Field Name', fontsize=12)
        plt.ylabel('Percentage of Datasets Missing Field', fontsize=12)
        plt.title('Top Missing Fields by Period (Top 3 Periods)', fontsize=14)
        plt.xticks(x, all_top_fields, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        output_path = os.path.join(analyzer_dir, 'missing_fields_by_period.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
