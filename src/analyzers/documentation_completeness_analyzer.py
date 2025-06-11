"""
Analyzer for visualizing dataset documentation completeness with stacked bar charts.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
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
        # Get analyzer-specific output directory
        analyzer_dir = self.get_analyzer_output_dir(output_dir)

        output_file_txt = os.path.join(analyzer_dir, 'documentation_completeness_analysis.txt')
        output_file_csv = os.path.join(analyzer_dir, 'documentation_completeness_analysis.csv')
        
        # Load field mappings and answer mappings
        field_mapping = self._load_json('data/field_mapping.json')
        answer_mapping = self._load_json('data/answer_mapping.json')
        
        # Process the dataset to get standardized answers
        field_answers_15, standardized_answers_15 = self._process_fields_and_answers(data, field_mapping, answer_mapping, 15)
        field_answers_5, standardized_answers_5 = self._process_fields_and_answers(data, field_mapping, answer_mapping, 5)
        field_answers_2, standardized_answers_2 = self._process_fields_and_answers(data, field_mapping, answer_mapping, 2)
        field_answers_overall, standardized_answers_overall = self._process_fields_and_answers(data, field_mapping, answer_mapping, 0)        # Create the visualization
        plot_file = self._create_stacked_bar_chart(standardized_answers_15, field_mapping, output_dir, period=15)
        self._create_stacked_bar_chart(standardized_answers_5, field_mapping, output_dir, period=5)
        self._create_stacked_bar_chart(standardized_answers_2, field_mapping, output_dir, period=2)
        self._create_stacked_bar_chart(standardized_answers_overall, field_mapping, output_dir, period=0)

        # Write comprehensive analysis results for all periods
        self._write_analysis(output_file_txt,
                            {
                                15: (field_answers_15, standardized_answers_15),
                                5: (field_answers_5, standardized_answers_5),
                                2: (field_answers_2, standardized_answers_2),
                                0: (field_answers_overall, standardized_answers_overall)
                            }, 
                            field_mapping)
        # Write analysis results into a csv format
        self._write_analysis_csv(output_file_csv, standardized_answers_overall, field_mapping)

        return output_file_txt
    
    def _load_json(self, file_path):
        """Load and parse a JSON file."""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {file_path}: {e}")
            return {}
    
    def _process_fields_and_answers(self, data, field_mapping, answer_mapping, period=0):
        """Process dataset fields and standardize answers."""
        field_answers = defaultdict(list)
        standardized_answers = defaultdict(Counter)
        
        # Additional field mappings (non-dropdown fields)
        # TODO synthesis type should be considered only where there is overlap synthesis ?
        additional_field_mappings = {
            "Label Source": "Was the label source provided?",
            "Labeller Population Rationale": "Was there a reason provided for the labeller population?",
            "Total Labellers": "Was the total number of labellers provided?",
            "Annotators per item": "Was the number of labellers per item specified?",
            "Label Threshold": "Was the label threshold provided?",
            "Synthesis Type": "Was the synthesis type described?",
            "Item Population": "Was the population of the items described?",
            "Item Population Rationale": "Was a reason given for choosing this item population?",
            "Item source": "Was the item source given?",
            "a Priori Sample Size": "Was the sample size chosen before data collection?",
            "Item Sample Size Rationale": "Was a rationale given for the sample size?",
            "a Priori Annotation Schema": "Was the annotation schema created beforehand?",
            "Annotation Schema Rationale": "Was a reason given for annotation schema?"
        }
        
        # Update field mapping with additional mappings
        for field, question in additional_field_mappings.items():
            if field not in field_mapping:
                field_mapping[field] = question

        # Deduplicate if the overall period is used
        if period == 0:
            filtered_data = {}
            seen = set()

            for full_name, info in data.items():
                base_name = full_name.split(',')[0]
                if base_name not in seen:
                    filtered_data[full_name] = info
                    seen.add(base_name)

            data = filtered_data

        # Process each field in each dataset
        for dataset_name, dataset_info in data.items():
            if dataset_info.get('period') == period or period == 0:
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

        no_information_fields = [
            "Human Labels",
            "OG Labels",
            "Overlap",
            "Formal Instructions",
            "Discussion",
            "a Priori Sample Size",
            "a Priori Annotation Schema"
        ]

        if field in no_information_fields and value == "No information":
            return "No information"

        if field in non_dropdown_fields:
            # Check for negative patterns
            negative_patterns = [
                "no information", "unknown", 
                "no details", "none", "not reported", "not specified", "unsure"
            ]

            na_patterns = [ "not applicable", 'n/a']

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
    
    def _create_stacked_bar_chart(self, standardized_answers, field_mapping, output_dir, sort=False, period=0):
        """Create a stacked bar chart showing documentation completeness."""
        # Set up the figure with appropriate size and style
        plt.figure(figsize=(16, 12))
        plt.style.use('ggplot')
        
        # Create a more visually appealing color scheme
        colors = {
            'Yes': '#009E73',            # Teal-green (distinct from red/green confusion)
            'Partially': '#0072B2',      # Blue (safe for most types of colorblindness)
            'No information': '#444444',
            'No': '#D55E00',             # Orange (distinct from teal and blue)
            'Not applicable': '#999999'  # Medium gray (neutral, readable)
        }

        
        # Setup for hatching (stripes) - more distinctive patterns
        # hatches = {
        #     'Yes': '///',
        #     'Partially': '\\\\',
        #     'No': 'xxx',
        #     'Not applicable': '.'
        # }
        
        # Standard categories in display order
        categories = ['Yes', 'Partially', 'No information', 'No', 'Not applicable']
        
        fields = list(reversed(standardized_answers.keys()))
        
        if sort:
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
                   edgecolor='white', linewidth=0.5,
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
                            color='white' if category in ['No', 'No information'] else 'black',
                            fontweight='bold')
                    
            left_positions = left_positions + counts

        if period == 0:
            title_text = "Overall"
        else:
            title_text = f"{period} Year Period"
        
        # Enhance the labels and title with better typography
        plt.yticks(y_pos, questions, fontsize=11, fontweight='bold')
        plt.xlabel('Count', fontsize=12, fontweight='bold')
        plt.title(f'Summary Results for Dataset Documentation, {title_text}',
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
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        plot_file = os.path.join(analyzer_dir, f'documentation_completeness_{period}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    def _write_analysis(self, output_file, period_data, field_mapping):
        """
        Write comprehensive analysis results to file for all time periods.
        
        Args:
            output_file: Path to output analysis file
            period_data: Dictionary mapping periods to (field_answers, standardized_answers) tuples
            field_mapping: Dictionary mapping field names to display names
        """
        with open(output_file, 'w') as f:
            f.write("=== DOCUMENTATION COMPLETENESS ANALYSIS ===\n\n")
            f.write("This analysis examines the completeness of dataset documentation across different time periods.\n")
            f.write("The analysis is broken down by 15-year, 5-year, and 2-year periods, as well as overall statistics.\n\n")
            
            # Write executive summary
            f.write("EXECUTIVE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # Calculate overall completeness percentages for each period
            period_names = {15: "15-Year Period", 5: "5-Year Period", 2: "2-Year Period", 0: "Overall"}
            period_completeness = {}
            
            for period, (field_answers, standardized_answers) in period_data.items():
                total_fields = sum(len(answers) for answers in field_answers.values())
                all_categories = Counter()
                
                for field, categories in standardized_answers.items():
                    for category, count in categories.items():
                        all_categories[category] += count
                
                yes_count = all_categories.get('Yes', 0)
                partial_count = all_categories.get('Partially', 0)
                yes_percentage = (yes_count / total_fields) * 100 if total_fields > 0 else 0
                partial_percentage = (partial_count / total_fields) * 100 if total_fields > 0 else 0
                combined_percentage = yes_percentage + partial_percentage
                
                period_completeness[period] = {
                    'total_fields': total_fields,
                    'yes_percentage': yes_percentage,
                    'partial_percentage': partial_percentage,
                    'combined_percentage': combined_percentage,
                    'categories': all_categories
                }
                
                # Count datasets
                dataset_count = 0
                if period == 15:
                    for field, answers in field_answers.items():
                        dataset_count = len(answers)
                        break
                
                # Write summary for this period
                f.write(f"{period_names[period]}:\n")
                if period != 0:
                    f.write(f"- Datasets analyzed: {dataset_count}\n")
                f.write(f"- Documentation completeness: {combined_percentage:.2f}%\n")
                f.write(f"  • Complete information: {yes_percentage:.2f}%\n")
                f.write(f"  • Partial information: {partial_percentage:.2f}%\n")
                f.write(f"- Most documented fields: {self._get_top_fields(standardized_answers, 3)}\n")
                f.write(f"- Least documented fields: {self._get_bottom_fields(standardized_answers, 3)}\n\n")
            
            # Trend analysis
            if all(period in period_completeness for period in [15, 5, 2]):
                f.write("Documentation Trends Over Time:\n")
                trend_15_to_5 = period_completeness[5]['combined_percentage'] - period_completeness[15]['combined_percentage']
                trend_5_to_2 = period_completeness[2]['combined_percentage'] - period_completeness[5]['combined_percentage']
                
                f.write(f"- From 15-year to 5-year period: {'+' if trend_15_to_5 > 0 else ''}{trend_15_to_5:.2f}% change\n")
                f.write(f"- From 5-year to 2-year period: {'+' if trend_5_to_2 > 0 else ''}{trend_5_to_2:.2f}% change\n")
                
                if trend_15_to_5 > 0 and trend_5_to_2 > 0:
                    f.write("- Overall trend: Documentation completeness is consistently improving over time\n\n")
                elif trend_15_to_5 < 0 and trend_5_to_2 < 0:
                    f.write("- Overall trend: Documentation completeness is declining over time\n\n")
                else:
                    f.write("- Overall trend: Documentation completeness shows mixed trends over time\n\n")
            
            # Add visualizations reference
            f.write("Visualizations:\n")
            f.write("- documentation_completeness_15.png: Stacked bar chart for 15-year period\n")
            f.write("- documentation_completeness_5.png: Stacked bar chart for 5-year period\n")
            f.write("- documentation_completeness_2.png: Stacked bar chart for 2-year period\n")
            f.write("- documentation_completeness_0.png: Stacked bar chart for overall analysis\n\n")
            
            # Detailed Analysis by Period
            for period, (field_answers, standardized_answers) in period_data.items():
                f.write("\n" + "=" * 80 + "\n")
                f.write(f"DETAILED ANALYSIS: {period_names[period].upper()}\n")
                f.write("=" * 80 + "\n\n")
                
                # Calculate dataset count for this period
                dataset_count = 0
                for field, answers in field_answers.items():
                    if len(answers) > dataset_count:
                        dataset_count = len(answers)
                
                if period != 0:
                    f.write(f"Number of datasets analyzed: {dataset_count}\n\n")
                
                # Top and bottom documented fields
                f.write("Top 5 Most Documented Fields:\n")
                f.write("-" * 50 + "\n")
                top_fields = self._get_top_fields_with_percentages(standardized_answers, 5)
                for field, percentage in top_fields:
                    question = field_mapping.get(field, field)
                    f.write(f"- {question}: {percentage:.2f}% complete or partial documentation\n")
                f.write("\n")
                
                f.write("Bottom 5 Least Documented Fields:\n")
                f.write("-" * 50 + "\n")
                bottom_fields = self._get_bottom_fields_with_percentages(standardized_answers, 5)
                for field, percentage in bottom_fields:
                    question = field_mapping.get(field, field)
                    f.write(f"- {question}: {percentage:.2f}% complete or partial documentation\n")
                f.write("\n")
                
                # Field breakdown
                f.write("Documentation Completeness by Field:\n")
                f.write("-" * 50 + "\n")
                
                sorted_fields = sorted(
                    standardized_answers.keys(),
                    key=lambda x: (standardized_answers[x].get('Yes', 0) + standardized_answers[x].get('Partially', 0)) / 
                                 sum(standardized_answers[x].values()) if sum(standardized_answers[x].values()) > 0 else 0,
                    reverse=True
                )
                
                for field in sorted_fields:
                    question = field_mapping.get(field, field)
                    answers = standardized_answers[field]
                    total = sum(answers.values())
                    
                    f.write(f"{question}:\n")
                    
                    # Calculate percentages
                    yes_count = answers.get('Yes', 0)
                    partial_count = answers.get('Partially', 0)
                    no_count = answers.get('No', 0) + answers.get('No information', 0)
                    na_count = answers.get('Not applicable', 0)
                    
                    yes_pct = (yes_count / total) * 100 if total > 0 else 0
                    partial_pct = (partial_count / total) * 100 if total > 0 else 0
                    no_pct = (no_count / total) * 100 if total > 0 else 0
                    na_pct = (na_count / total) * 100 if total > 0 else 0
                    
                    f.write(f"  - Complete: {yes_count} ({yes_pct:.2f}%)\n")
                    f.write(f"  - Partial: {partial_count} ({partial_pct:.2f}%)\n")
                    f.write(f"  - Missing: {no_count} ({no_pct:.2f}%)\n")
                    f.write(f"  - Not applicable: {na_count} ({na_pct:.2f}%)\n")
                    f.write("\n")
                
                # Overall statistics for this period
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
                
                # Display in a specific order
                categories_order = ['Yes', 'Partially', 'No information', 'No', 'Not applicable']
                for category in categories_order:
                    if category in all_categories:
                        count = all_categories[category]
                        percentage = (count / total_fields) * 100 if total_fields > 0 else 0
                        f.write(f"- {category}: {count} ({percentage:.2f}%)\n")
            
            # Add recommendations section
            f.write("\n" + "=" * 80 + "\n")
            f.write("RECOMMENDATIONS FOR IMPROVING DATASET DOCUMENTATION\n")
            f.write("=" * 80 + "\n\n")
            
            # Get consistently problematic fields across all periods
            problematic_fields = self._get_consistently_problematic_fields(period_data)
            
            f.write("Based on the analysis, the following documentation aspects need the most improvement:\n\n")
            
            for field, avg_percentage in problematic_fields[:5]:
                question = field_mapping.get(field, field)
                f.write(f"1. {question} - Only {avg_percentage:.2f}% of datasets provide this information\n")
                f.write(f"   Recommendation: Make this field a required part of dataset documentation standards\n\n")
            
            f.write("General recommendations:\n")
            f.write("1. Develop and promote standardized documentation templates that include all critical fields\n")
            f.write("2. Create documentation checklists for dataset publishers\n")
            f.write("3. Incentivize complete documentation through recognition or citation benefits\n")
            f.write("4. Conduct regular audits of dataset documentation completeness in repositories\n")
        
        return output_file
    
    def _get_top_fields(self, standardized_answers, n=3):
        """Get the top n most documented fields."""
        field_completeness = {}
        
        for field, answers in standardized_answers.items():
            total = sum(answers.values())
            yes_count = answers.get('Yes', 0)
            partial_count = answers.get('Partially', 0)
            completeness = (yes_count + partial_count) / total if total > 0 else 0
            field_completeness[field] = completeness
        
        top_fields = sorted(field_completeness.items(), key=lambda x: x[1], reverse=True)[:n]
        return ", ".join(field for field, _ in top_fields)
    
    def _get_bottom_fields(self, standardized_answers, n=3):
        """Get the bottom n least documented fields."""
        field_completeness = {}
        
        for field, answers in standardized_answers.items():
            total = sum(answers.values())
            yes_count = answers.get('Yes', 0)
            partial_count = answers.get('Partially', 0)
            completeness = (yes_count + partial_count) / total if total > 0 else 0
            field_completeness[field] = completeness
        
        bottom_fields = sorted(field_completeness.items(), key=lambda x: x[1])[:n]
        return ", ".join(field for field, _ in bottom_fields)
    
    def _get_top_fields_with_percentages(self, standardized_answers, n=5):
        """Get the top n most documented fields with their percentages."""
        field_completeness = {}
        
        for field, answers in standardized_answers.items():
            total = sum(answers.values())
            yes_count = answers.get('Yes', 0)
            partial_count = answers.get('Partially', 0)
            completeness = ((yes_count + partial_count) / total) * 100 if total > 0 else 0
            field_completeness[field] = completeness
        
        top_fields = sorted(field_completeness.items(), key=lambda x: x[1], reverse=True)[:n]
        return top_fields
    
    def _get_bottom_fields_with_percentages(self, standardized_answers, n=5):
        """Get the bottom n least documented fields with their percentages."""
        field_completeness = {}
        
        for field, answers in standardized_answers.items():
            total = sum(answers.values())
            yes_count = answers.get('Yes', 0)
            partial_count = answers.get('Partially', 0)
            completeness = ((yes_count + partial_count) / total) * 100 if total > 0 else 0
            field_completeness[field] = completeness
        
        bottom_fields = sorted(field_completeness.items(), key=lambda x: x[1])[:n]
        return bottom_fields
    
    def _get_consistently_problematic_fields(self, period_data):
        """Identify fields that are consistently problematic across all time periods."""
        # Calculate average completeness for each field across all periods
        field_avg_completeness = defaultdict(list)
        
        for period, (_, standardized_answers) in period_data.items():
            for field, answers in standardized_answers.items():
                total = sum(answers.values())
                yes_count = answers.get('Yes', 0)
                partial_count = answers.get('Partially', 0)
                completeness = ((yes_count + partial_count) / total) * 100 if total > 0 else 0
                field_avg_completeness[field].append(completeness)
        
        # Calculate average completeness across periods
        field_avg = {}
        for field, percentages in field_avg_completeness.items():
            if len(percentages) == len(period_data):  # Only include fields present in all periods
                field_avg[field] = sum(percentages) / len(percentages)
        
        # Sort by average completeness (ascending)
        problematic_fields = sorted(field_avg.items(), key=lambda x: x[1])
        
        return problematic_fields

    def _write_analysis_csv(self, output_file, standardized_answers, field_mapping):
        """Write analysis results to CSV file."""
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write the CSV header
            writer.writerow([
                "Question",
                "Yes",
                "Partially",
                "No information",
                "No",
                "Partially",
                "Not applicable"
            ])

            # Standard categories to ensure consistent column ordering
            categories = ["Yes", "Partially", "No information", "No", "Not applicable"]

            for field, answers in standardized_answers.items():
                question = field_mapping.get(field, field)
                row = [question]
                # Add counts for each category, defaulting to 0 if missing
                for cat in categories:
                    row.append(answers.get(cat, 0))
                writer.writerow(row)

        return output_file
