import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from typing import Dict, Any, List, Tuple

from analyzers.base import Analyzer, register_analyzer

@register_analyzer
class FieldImpactAnalyzer(Analyzer):
    """
    Analyzer that examines the impact of providing information for each field
    on the overall completeness of datasets.
    """
    
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        return "Field Impact Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Analyzes how providing information for specific fields correlates with overall dataset completeness"
    
    def analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str:
        """
        Analyze each field's impact on overall dataset completeness.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Get analyzer-specific output directory
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        output_file = os.path.join(analyzer_dir, 'field_impact_analysis.txt')
        
        # Load answer mappings from JSON files
        answer_mapping = self._load_json('data/answer_mapping.json')
        
        # Fields to skip in the analysis (metadata fields)
        skip_fields = ['period', 'Done for?', 'Additional Links', 'DS Citation', 
                      'DS DOI', 'name', 'citation_sum', 'status', 'Available',
                      'Notes', 'Unnamed: 33']
        
        # First, identify all possible fields across all datasets
        all_possible_fields = set()
        for dataset_info in data.values():
            for field in dataset_info.keys():
                if field not in skip_fields:
                    all_possible_fields.add(field)
        
        print(f"Total fields identified for analysis: {len(all_possible_fields)}")
        print(f"Fields being analyzed: {sorted(all_possible_fields)}")
        
        # Calculate missing information percentage for each dataset
        dataset_info_status = {}
        
        for dataset_name, dataset_info in data.items():
            total_fields = 0
            missing_fields = 0
            field_status = {}
            
            for field in all_possible_fields:
                value = dataset_info.get(field, "")
                if field in skip_fields:
                    continue
                
                # Standardize the answer using the same approach as DocumentationCompletenessAnalyzer
                standardized = self._standardize_answer(value, answer_mapping, field)
                field_status[field] = standardized
                
                # Count towards missing percentage calculation
                total_fields += 1
                if standardized in ["No", "No information"]:
                    missing_fields += 1
            
            # Calculate missing percentage
            missing_percentage = (missing_fields / total_fields * 100) if total_fields > 0 else 0
            
            # Store results for this dataset
            dataset_info_status[dataset_name] = {
                "field_status": field_status,
                "missing_percentage": missing_percentage,
                "total_fields": total_fields,
                "missing_fields": missing_fields
            }
        
        # Now analyze field by field
        field_analysis_results = []
        
        for field in sorted(all_possible_fields):
            # Group datasets by their standardized status for this field
            datasets_by_status = defaultdict(list)
            
            for dataset_name, info in dataset_info_status.items():
                status = info["field_status"].get(field, "No information")
                datasets_by_status[status].append((dataset_name, info))
            
            # Calculate average missing percentage for datasets in each status category
            status_stats = {}
            
            for status, datasets in datasets_by_status.items():
                if not datasets:
                    continue
                    
                # Calculate average missing percentage for this status
                avg_missing = sum(info["missing_percentage"] for _, info in datasets) / len(datasets)
                
                # Calculate standard deviation
                if len(datasets) > 1:
                    std_dev = np.std([info["missing_percentage"] for _, info in datasets])
                else:
                    std_dev = 0
                    
                status_stats[status] = {
                    "count": len(datasets),
                    "avg_missing": avg_missing,
                    "std_dev": std_dev,
                    "datasets": datasets
                }
                
            # Calculate the impact of having "Yes" information for this field
            has_impact = False
            impact_value = 0
            
            if "Yes" in status_stats and "No information" in status_stats:
                yes_missing = status_stats["Yes"]["avg_missing"]
                no_info_missing = status_stats["No information"]["avg_missing"]
                impact_value = no_info_missing - yes_missing
                has_impact = True
                
            # Store analysis results for this field
            field_analysis_results.append({
                "field": field,
                "status_stats": status_stats,
                "has_impact": has_impact,
                "impact_value": impact_value
            })
                
        # Sort results by impact value (positive impact first)
        field_analysis_results.sort(key=lambda x: -x["impact_value"] if x["has_impact"] else float('-inf'))
        
        # Write analysis results
        with open(output_file, 'w') as f:
            f.write("=== Field Information Impact Analysis ===\n\n")
            f.write("This analysis examines how providing information for specific fields\n")
            f.write("correlates with the overall completeness of datasets.\n\n")
            
            f.write(f"Total fields identified: {len(all_possible_fields)}\n")
            analyzable_fields = [r for r in field_analysis_results if r["has_impact"]]
            f.write(f"Fields with sufficient variation for analysis: {len(analyzable_fields)}\n\n")
            
            f.write("For each field, we compare datasets that provide information ('Yes')\n")
            f.write("with datasets that don't ('No information'), measuring the average\n")
            f.write("percentage of missing information across all fields.\n\n")
            
            # Top Fields with Positive Impact
            f.write("Fields with the Strongest Impact on Overall Completeness:\n")
            f.write("-------------------------------------------------------\n")
            positive_impact_fields = [r for r in field_analysis_results if r["has_impact"] and r["impact_value"] > 0]
            
            for result in positive_impact_fields:
                field = result["field"]
                impact = result["impact_value"]
                yes_stats = result["status_stats"].get("Yes", {"count": 0, "avg_missing": 0})
                no_info_stats = result["status_stats"].get("No information", {"count": 0, "avg_missing": 0})
                
                f.write(f"{field}:\n")
                f.write(f"  Impact: POSITIVE ({impact:.2f}% difference)\n")
                f.write(f"  Datasets with '{field}' information ({yes_stats['count']}): {yes_stats['avg_missing']:.2f}% missing information\n")
                f.write(f"  Datasets without '{field}' information ({no_info_stats['count']}): {no_info_stats['avg_missing']:.2f}% missing information\n")
                
                # Show category breakdown
                for status, stats in result["status_stats"].items():
                    f.write(f"    {status}: {stats['count']} datasets, {stats['avg_missing']:.2f}% avg missing\n")
                f.write("\n")
            
            # Neutral/Negative Impact Fields
            f.write("\n\nFields with NEUTRAL or NEGATIVE Impact:\n")
            f.write("---------------------------------------\n")
            other_fields = [r for r in field_analysis_results if not r["has_impact"] or r["impact_value"] <= 0]
            
            for result in other_fields:
                field = result["field"]
                if not result["has_impact"]:
                    f.write(f"\n{field}: Insufficient data for comparison\n")
                    
                    # List available statuses
                    for status, stats in result["status_stats"].items():
                        f.write(f"  {status}: {stats['count']} datasets, {stats['avg_missing']:.2f}% avg missing\n")
                else:
                    impact = result["impact_value"]
                    yes_stats = result["status_stats"].get("Yes", {"count": 0, "avg_missing": 0})
                    no_info_stats = result["status_stats"].get("No information", {"count": 0, "avg_missing": 0})
                    
                    f.write(f"\n{field}:\n")
                    f.write(f"  Impact: NEGATIVE ({abs(impact):.2f}% difference)\n")
                    f.write(f"  Datasets with '{field}' information ({yes_stats['count']}): {yes_stats['avg_missing']:.2f}% missing overall\n")
                    f.write(f"  Datasets without '{field}' information ({no_info_stats['count']}): {no_info_stats['avg_missing']:.2f}% missing overall\n")
                    
                    # Show category breakdown  
                    for status, stats in result["status_stats"].items():
                        f.write(f"    {status}: {stats['count']} datasets, {stats['avg_missing']:.2f}% avg missing\n")
        
        # Create visualizations
        self._create_visualizations(field_analysis_results, analyzer_dir)
        
        return output_file
        
    def _standardize_answer(self, value, answer_mapping, field):
        """
        Standardize an answer using the answer mapping or simplified rules for non-dropdown fields.
        
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
            
        # Handle non-dropdown fields with simpler binary logic
        non_dropdown_fields = [
            "Labeller Population Rationale",
            "Label Source",
            "Total Labellers",
            "Annotators per item",
            "Label Threshold",
            "Item Population",
            "Item Population Rationale",
            "Item source",
            "a Priori Sample Size",
            "Item Sample Size Rationale",
            "a Priori Annotation Schema",
            "Annotation Schema Rationale"
        ]

        no_information_fields = [
            "Human Labels",
            "OG Labels",
            "Overlap",
            "Formal Instructions",
            "Discussion",
            "a Priori Sample Size",
            "a Priori Annotation Schema"
        ]

        if field in no_information_fields and value_str == "No information":
            return "No information"

        if field in non_dropdown_fields:
            # Check for negative patterns
            negative_patterns = [
                "no information", "unknown", 
                "no details", "none", "not reported", "not specified", "unsure"
            ]

            na_patterns = ["not applicable", 'n/a']

            if any(pattern in value_str.lower() for pattern in na_patterns):
                return "Not applicable"
            
            if any(pattern in value_str.lower() for pattern in negative_patterns):
                return "No"
            return "Yes"
            
        # For dropdown fields, use the answer mapping
        # Try to find exact match in answer mapping
        if value_str in answer_mapping:
            return answer_mapping[value_str]
        
        # Try case-insensitive match
        for answer_key, std_value in answer_mapping.items():
            if value_str.lower() == answer_key.lower():
                return std_value
        
        # Default categorization based on content
        if value_str.lower() in ['n/a', 'not applicable']:
            return "Not applicable"
        
        # If no mapping found and value is present, consider it as "Yes"
        return "Yes"
    
    def _load_json(self, file_path):
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def _create_visualizations(self, results, output_dir):
        """Create visualizations for the field impact analysis."""
        # Filter for fields that have both 'Yes' and 'No information' statuses
        comparable_fields = [r for r in results if r["has_impact"]]
        
        if not comparable_fields:
            return
            
        # 1. Bar chart of field impact values
        plt.figure(figsize=(12, max(8, len(comparable_fields) * 0.4)))
        
        fields = [r["field"] for r in comparable_fields]
        impact_values = [r["impact_value"] for r in comparable_fields]
        
        # Color bars based on positive/negative impact
        colors = ['green' if val > 0 else 'red' for val in impact_values]
        
        y_pos = np.arange(len(fields))
        plt.barh(y_pos, impact_values, color=colors)
        plt.yticks(y_pos, fields)
        plt.xlabel('Impact on Missing Information %')
        plt.title('Impact of Providing Field Information on Dataset Completeness')
        
        # Add zero line
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(impact_values):
            plt.text(v + 0.5 if v >= 0 else v - 3, i, f"{v:.1f}%", va='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'field_impact_chart.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Side-by-side comparison chart for top 5 positive impact fields
        positive_fields = [r for r in comparable_fields if r["impact_value"] > 0]
        top_positive = sorted(positive_fields, key=lambda x: x["impact_value"], reverse=True)[:5]
        
        if top_positive:
            plt.figure(figsize=(12, 8))
            
            field_names = [r["field"] for r in top_positive]
            yes_values = [r["status_stats"]["Yes"]["avg_missing"] for r in top_positive]
            no_info_values = [r["status_stats"]["No information"]["avg_missing"] for r in top_positive]
            
            x = np.arange(len(field_names))
            width = 0.35
            
            plt.bar(x - width/2, yes_values, width, label='With Information', color='green', alpha=0.7)
            plt.bar(x + width/2, no_info_values, width, label='Without Information', color='red', alpha=0.7)
            
            plt.ylabel('Missing Information %')
            plt.title('Top 5 Fields: Impact on Dataset Completeness')
            plt.xticks(x, field_names, rotation=45, ha='right')
            plt.legend()
            
            # Add value labels
            for i, v in enumerate(yes_values):
                plt.text(i - width/2, v + 1, f"{v:.1f}%", ha='center')
            
            for i, v in enumerate(no_info_values):
                plt.text(i + width/2, v + 1, f"{v:.1f}%", ha='center')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'top_fields_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Scatter plot showing relationship between field presence and completeness
        plt.figure(figsize=(12, 8))
        
        x_values = []  # Percentage of datasets with information for each field
        y_values = []  # Impact value
        sizes = []     # Size based on number of datasets
        labels = []    # Field names
        
        for result in comparable_fields:
            yes_count = result["status_stats"].get("Yes", {"count": 0})["count"]
            no_info_count = result["status_stats"].get("No information", {"count": 0})["count"]
            total = yes_count + no_info_count
            
            if total > 0:
                presence_pct = (yes_count / total) * 100
                x_values.append(presence_pct)
                y_values.append(result["impact_value"])
                sizes.append(total * 10)  # Scale dot sizes
                labels.append(result["field"])
        
        scatter = plt.scatter(x_values, y_values, s=sizes, alpha=0.6, c=y_values, cmap='RdYlGn')
        
        # Add labels for each point
        for i, label in enumerate(labels):
            plt.annotate(label, (x_values[i], y_values[i]), fontsize=8,
                        xytext=(5, 5), textcoords='offset points')
        
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.colorbar(scatter, label='Impact Value')
        
        plt.xlabel('% of Datasets with Field Information')
        plt.ylabel('Impact on Missing Information %')
        plt.title('Relationship Between Field Presence and Completeness Impact')
        
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'field_presence_impact_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()