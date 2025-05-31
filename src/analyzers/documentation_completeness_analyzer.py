"""
Analyzer for visualizing dataset documentation completeness with stacked bar charts.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
from analyzers.base import register_analyzer, Analyzer

@register_analyzer
class DocumentationCompletenessAnalyzer(Analyzer):
    """Analyzer that creates stacked bar charts showing documentation completeness."""
    
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        return "Documentation Completeness Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Creates stacked bar charts showing documentation completeness by field"
    
    def analyze(self, data, output_dir='outputs'):
        """
        Analyze the dataset documentation completeness and create stacked bar charts.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, 'documentation_completeness_analysis.txt')
        
        # Load field mappings and answer mappings
        field_mapping = self._load_json('data/field_mapping.json')
        answer_mapping = self._load_json('data/answer_mapping.json')
        
        # Process the dataset to get standardized answers
        field_answers, standardized_answers = self._process_fields_and_answers(data, field_mapping, answer_mapping)
        
        # Create the visualization
        plot_file = self._create_stacked_bar_chart(standardized_answers, field_mapping, output_dir)
        
        # Write analysis results
        self._write_analysis(output_file, field_answers, standardized_answers, field_mapping)
        
        return output_file
    
    def _load_json(self, file_path):
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def _process_fields_and_answers(self, data, field_mapping, answer_mapping):
        """Process dataset fields and standardize answers."""
        field_answers = defaultdict(list)
        standardized_answers = defaultdict(Counter)
        
        # Additional field mappings (non-dropdown fields)
        # TODO synthesis type should be considered only where there is overlap synthesis ?
        additional_field_mappings = {
            "Labeller Population Rationale": "Rationale behind choosing those labellers?",
            "Label Source": "Provided the label source?",
            "Labeller Population Rationale": "Reason provided for the labeller population?",
            "Total Labellers": "Provided the total number of labellers?",
            "Annotators per item": "Specified how many labellers annotated each item?",
            "Label Threshold": "Provided the label threshold?",
            # "Synthesis Type": "When overlap was done, did they mention how it was aggregated?",
            "Item Population": "Population of the items described?",
            "Item Population Rationale": "Reason given for choosing this item population?",
            "Item source": "Item source given?",
            "a Priori Sample Size": "Sample size chosen before data collection?",
            "Item Sample Size Rationale": "Rationale given for the item size?",
            "a Priori Annotation Schema": "Annotation schema created beforehand?",
            "Annotation Schema Rationale": "Reason given for annotation schema?"
        }
        
        # Update field mapping with additional mappings
        for field, question in additional_field_mappings.items():
            if field not in field_mapping:
                field_mapping[field] = question
                
        # Process each field in each dataset
        for dataset_name, dataset_info in data.items():
            for field, value in dataset_info.items():
                # Only process fields that have mappings
                if field in field_mapping:
                    field_answers[field].append(value)
                    
                    # Standardize the answer
                    standardized = self._standardize_answer(value, answer_mapping, field)
                    standardized_answers[field][standardized] += 1
        
        return field_answers, standardized_answers
    
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
        # For these fields: "No information" or "Not applicable" -> "No", anything else -> "Yes"
        # TODO Convert empty column
        non_dropdown_fields = [
            "Labeller Population Rationale",
            "Label Source",
            "Labeller Population Rationale",
            "Total Labellers",
            "Annotators per item",
            "Label Threshold",
            # "Synthesis Type",
            "Item Population",
            "Item Population Rationale",
            "Item source",
            "a Priori Sample Size",
            "Item Sample Size Rationale",
            "a Priori Annotation Schema",
            "Annotation Schema Rationale"
        ]
        
        if field in non_dropdown_fields:
            # Check for negative patterns
            negative_patterns = [
                "no information", "not applicable", "n/a", "unknown", 
                "no details", "none", "not reported", "not specified"
            ]
            
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
    
    def _create_stacked_bar_chart(self, standardized_answers, field_mapping, output_dir):
        """Create a stacked bar chart showing documentation completeness."""
        # Set up the figure with appropriate size and style
        plt.figure(figsize=(16, 12))
        plt.style.use('ggplot')
        
        # Create a more visually appealing color scheme
        colors = {
            'Yes': '#2ca02c',          # Green
            'Partially': '#1f77b4',    # Blue
            'No': '#ff7f0e',           # Orange/Amber
            'Not applicable': '#d3d3d3'  # Light Gray
        }
        
        # Setup for hatching (stripes) - more distinctive patterns
        hatches = {
            'Yes': '///',
            'Partially': '\\\\',
            'No': 'xxx',
            'Not applicable': '.'
        }
        
        # Standard categories in display order
        categories = ['Yes', 'Partially', 'No', 'Not applicable']
        
        # Get fields and sort them based on Yes count (descending)
        fields = sorted(standardized_answers.keys(), 
                      key=lambda x: standardized_answers[x].get('Yes', 0), 
                      reverse=True)
        
        # Map fields to questions for display
        questions = [field_mapping.get(field, field) for field in fields]
        
        # Create y-positions for the bars with more spacing
        y_pos = np.arange(len(fields)) * 1.2
        
        # Prepare data for stacking
        data_for_plotting = []
        left_positions = np.zeros(len(fields))
        
        # Calculate total count for each field for percentage labels
        total_counts = []
        for field in fields:
            total = sum(standardized_answers[field].values())
            total_counts.append(total)
        
        # For each category, create a bar segment
        for category in categories:
            category_counts = [standardized_answers[field].get(category, 0) for field in fields]
            data_for_plotting.append((category, category_counts))
        
        # Plot each category as a stacked segment
        ax = plt.gca()
        bars = []
        for category, counts in data_for_plotting:
            bar = plt.barh(y_pos, counts, left=left_positions, color=colors[category], 
                   hatch=hatches[category], edgecolor='white', linewidth=0.5,
                   label=category if category in colors else category, height=0.8)
            bars.append(bar)
            
            # Add percentage labels inside bars (for significant segments)
            for i, (count, total) in enumerate(zip(counts, total_counts)):
                if count > 0 and (count/total) > 0.05:  # Only add label if segment is >5% of total
                    percentage = (count / total) * 100
                    # Position the text in the middle of each segment
                    text_x = left_positions[i] + count/2
                    plt.text(text_x, y_pos[i], f'{percentage:.0f}%', 
                            ha='center', va='center', fontsize=9, 
                            color='white' if category in ['No', 'No information / Unsure'] else 'black',
                            fontweight='bold')
                    
            left_positions = left_positions + counts
        
        # Enhance the labels and title with better typography
        plt.yticks(y_pos, questions, fontsize=11, fontweight='bold')
        plt.xlabel('Count', fontsize=12, fontweight='bold')
        plt.title('Summary Results for Dataset Documentation', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Add subtitle
        plt.figtext(0.5, 0.93, 'Presence of key information across analyzed datasets', 
                   fontsize=12, ha='center', fontweight='regular', style='italic')
        
        # Add legend at the bottom right with better styling
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = plt.legend(by_label.values(), by_label.keys(), 
                          loc='lower right', fontsize=11, framealpha=0.9,
                          title='Documentation Status', title_fontsize=12)
        legend.get_frame().set_linewidth(1.0)
        
        # Enhance grid and background
        plt.grid(axis='x', linestyle='--', alpha=0.4)
        ax.set_facecolor('#f5f5f5')  # Light gray background
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        plt.tight_layout(pad=2.0)
        
        # Save the figure
        plot_file = os.path.join(output_dir, 'documentation_completeness.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    def _write_analysis(self, output_file, field_answers, standardized_answers, field_mapping):
        """Write analysis results to file."""
        with open(output_file, 'w') as f:
            f.write("=== Documentation Completeness Analysis ===\n\n")
            
            # Summary by field
            f.write("Documentation Completeness by Field:\n")
            f.write("-" * 50 + "\n")
            
            for field, answers in standardized_answers.items():
                question = field_mapping.get(field, field)
                total = sum(answers.values())
                
                f.write(f"{question}:\n")
                for category, count in sorted(answers.items(), key=lambda x: x[1], reverse=True):
                    percentage = (count / total) * 100 if total > 0 else 0
                    f.write(f"  - {category}: {count} ({percentage:.2f}%)\n")
                f.write("\n")
            
            # Overall statistics
            f.write("Overall Documentation Statistics:\n")
            f.write("-" * 50 + "\n")
            
            total_fields = sum(len(answers) for answers in field_answers.values())
            f.write(f"Total fields analyzed: {len(field_answers)}\n")
            f.write(f"Total data points: {total_fields}\n")
            
            # Count overall categories
            all_categories = Counter()
            for field, categories in standardized_answers.items():
                for category, count in categories.items():
                    all_categories[category] += count
            
            for category, count in sorted(all_categories.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_fields) * 100 if total_fields > 0 else 0
                f.write(f"- {category}: {count} ({percentage:.2f}%)\n")
            
        return output_file
