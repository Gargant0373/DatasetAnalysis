import os
import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter, defaultdict
from typing import Dict, Any, List, Tuple

from analyzers.base import Analyzer, register_analyzer

@register_analyzer
class LabelSourceIRRAnalyzer(Analyzer):
    """Analyzer that extracts and categorizes label sources and IRR measures."""
    
    def get_name(self) -> str:
        """Get the name of the analyzer."""
        return "Label Source and IRR Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Analyzes label sources and IRR measures across datasets in detail"
    
    def analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str:
        """
        Extract and analyze label sources and IRR measures from datasets.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Get analyzer-specific output directory
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        output_file = os.path.join(analyzer_dir, 'label_source_irr_analysis.txt')
        
        # Convert data to DataFrame for easier analysis
        df = pd.DataFrame(data.values())
        
        # ===== LABEL SOURCE ANALYSIS =====
        # Extract all label sources with original text
        all_label_sources = []
        for dataset_name, info in data.items():
            label_source = info.get('Label Source', '')
            if label_source and str(label_source).lower() not in ['no information', 'nan', 'none', 'not applicable']:
                all_label_sources.append({
                    'dataset': dataset_name,
                    'label_source': label_source
                })
        
        # Define categories for grouping label sources
        label_source_categories = {
            'mturk': ['turk', 'amt', 'amazon mechanical'],
            'university': ['university', 'college', 'student', 'faculty', 'lab members'],
            'crowdsource_other': ['crowd', 'figure eight', 'prolific', 'clickworker'],
            'experts': ['expert', 'specialist', 'professional', 'trained annotator', 'researchers', 'inspectors'],
            'authors': ['author', 'creator'],
            'volunteers': ['volunteer', 'unpaid', 'community'],
            'existing_source': ['existing dataset', 'previous work', 'already labeled', 'dataset', 'original', 'web']
        }
        
        # Categorize each label source
        categorized_sources = []
        for item in all_label_sources:
            source_text = str(item['label_source']).lower()
            category = 'other'
            
            # Find matching category
            for cat_name, keywords in label_source_categories.items():
                if any(keyword in source_text for keyword in keywords):
                    category = cat_name
                    break
                    
            categorized_sources.append({
                'dataset': item['dataset'],
                'label_source': item['label_source'],
                'category': category
            })
        
        # Count categories
        category_counts = Counter([item['category'] for item in categorized_sources])
        
        # ===== IRR ANALYSIS =====
        # Extract metrics used for evaluation when IRR was calculated
        all_metrics = []
        for dataset_name, info in data.items():
            irr_status = info.get('IRR', '')
            metrics = info.get('Metric', '')
            
            # Only analyze metrics if IRR was calculated
            if str(irr_status).lower() in ['calculated', 'yes', 'measured', 'reported', 'computed']:
                if metrics and str(metrics).lower() not in ['no information', 'nan', 'none', 'not applicable']:
                    all_metrics.append({
                        'dataset': dataset_name,
                        'metric': metrics
                    })

        # Define categories for metrics
        metric_categories = {
            'kappa': ['kappa', 'κ', 'cohen', 'fleiss'],
            'agreement_percentage': ['percent', '%', 'agreement', 'accuracy'],
            'icc': ['icc', 'intraclass', 'correlation coefficient'],
            'krippendorff': ['krippendorff', 'alpha'],
            'f1': ['f1', 'f-score', 'f-measure'],
            'precision': ['precision', 'positive predictive value'],
            'recall': ['recall', 'sensitivity', 'true positive rate'],
            'correlation': ['correlation', 'pearson', 'spearman', 'kendall'],
            'mse': ['mse', 'mean squared error', 'rmse', 'root mean squared'],
            'mae': ['mae', 'mean absolute error'],
            'iou': ['iou', 'intersection over union', 'jaccard'],
            'pairwise': ['pairwise', 'inter-annotator', 'inter-rater'],
            'classifier_performance': ['classifier performance', 'classification accuracy', 'model accuracy'],
            'statistical_summary': ['standard deviation', 'gcc', 'lce', 'global consistency', 'local consistency']
        }

        # Categorize each metric
        categorized_metrics = []
        for item in all_metrics:
            metric_text = str(item['metric']).lower()
            categories = []
            
            # Find matching category - allow multiple categories per metric
            for cat_name, keywords in metric_categories.items():
                if any(keyword in metric_text for keyword in keywords):
                    categories.append(cat_name)
            
            # If no categories found, use 'other'
            if not categories:
                categories = ['other']
                
            # Extract numeric values if present
            numeric_values = {}
            for category in categories:
                # Look for numbers near category keywords
                for keyword in metric_categories.get(category, []):
                    if keyword in metric_text:
                        # Try to find a number near this keyword
                        number_match = re.search(r'{}[^0-9]*(\d+\.?\d*)'.format(keyword), metric_text)
                        if not number_match:
                            # Try looking before the keyword
                            number_match = re.search(r'(\d+\.?\d*)[^0-9]*{}'.format(keyword), metric_text)
                        
                        if number_match:
                            try:
                                value = float(number_match.group(1))
                                # If percentage is greater than 1 but not specified as percentage, assume it's a percentage
                                if category == 'agreement_percentage' and value > 1 and '%' not in metric_text:
                                    value /= 100
                                numeric_values[category] = value
                                break
                            except ValueError:
                                pass
            # If no specific number found for categories, try to find any number
            if not numeric_values and categories != ['other']:
                general_match = re.search(r'(\d+\.?\d*)', metric_text)
                if general_match:
                    try:
                        value = float(general_match.group(1))
                        # For simplicity, assign to first category
                        if value <= 1.0 or '%' in metric_text:  # Looks like a percentage or coefficient
                            numeric_values[categories[0]] = value
                    except ValueError:
                        pass
                        
            categorized_metrics.append({
                'dataset': item['dataset'],
                'metric': item['metric'],
                'categories': categories,
                'numeric_values': numeric_values
            })
        
        # Count metric categories (a dataset may use multiple metrics)
        metric_category_counts = Counter()
        for item in categorized_metrics:
            for category in item['categories']:
                metric_category_counts[category] += 1

        # Calculate average values by metric category
        metric_averages = defaultdict(list)
        for item in categorized_metrics:
            for category, value in item['numeric_values'].items():
                metric_averages[category].append(value)
                    
        metric_avg_values = {}
        for category, values in metric_averages.items():
            if values:
                metric_avg_values[category] = sum(values) / len(values)



        # ===== WRITE RESULTS =====
        with open(output_file, 'w') as f:
            f.write("=== Label Source and IRR Analysis ===\n\n")
            
            # Write Label Source Analysis
            f.write("LABEL SOURCE ANALYSIS\n")
            f.write("---------------------\n\n")
            
            f.write(f"Total datasets with specified label sources: {len(all_label_sources)}\n\n")
            
            f.write("Label Source Categories:\n")
            for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(categorized_sources)) * 100
                f.write(f"  {category}: {count} ({percentage:.1f}%)\n")
                
            f.write("\nDetailed Label Sources by Category:\n")
            for category in sorted(category_counts.keys()):
                f.write(f"\n{category.upper()}:\n")
                category_items = [item for item in categorized_sources if item['category'] == category]
                for item in category_items:
                    f.write(f"  - {item['dataset']}: {item['label_source']}\n")
            
            # Write IRR Analysis
            f.write("\n\nMETRIC ANALYSIS\n")
            f.write("---------------\n\n")

            f.write(f"Total datasets reporting metrics with IRR calculation: {len(all_metrics)}\n\n")

            f.write("Metric Categories:\n")
            for category, count in sorted(metric_category_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(categorized_metrics)) * 100
                avg_value = metric_avg_values.get(category)
                avg_text = f", avg value: {avg_value:.2f}" if avg_value is not None else ""
                f.write(f"  {category}: {count} ({percentage:.1f}%){avg_text}\n")
                
            f.write("\nDetailed Metrics by Category:\n")
            for category in sorted(metric_category_counts.keys()):
                f.write(f"\n{category.upper()}:\n")
                # Find all datasets that use this metric category
                category_items = [item for item in categorized_metrics if category in item['categories']]
                for item in category_items:
                    value_text = ""
                    if category in item['numeric_values']:
                        value_text = f" (value: {item['numeric_values'][category]:.2f})"
                    f.write(f"  - {item['dataset']}: {item['metric']}{value_text}\n")

            # Update summary section:
            top_metric_categories = sorted(metric_category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            f.write("\nTop metrics used with IRR:\n")
            for category, count in top_metric_categories:
                percentage = (count / len(categorized_metrics)) * 100
                f.write(f"  - {category}: {count} ({percentage:.1f}%)\n")
                
            if metric_avg_values:
                f.write("\nAverage metric values by category:\n")
                for category, avg in sorted(metric_avg_values.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"  - {category}: {avg:.2f}\n")
            
            # Write summary of findings
            f.write("\n\nSUMMARY OF FINDINGS\n")
            f.write("===================\n\n")
            
            # Label source summary
            top_source_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            f.write("Top label sources:\n")
            for category, count in top_source_categories:
                percentage = (count / len(categorized_sources)) * 100
                f.write(f"  - {category}: {count} ({percentage:.1f}%)\n")
                
            # # IRR summary
            # top_irr_categories = sorted(irr_category_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            # f.write("\nTop IRR measures:\n")
            # for category, count in top_irr_categories:
            #     percentage = (count / len(categorized_irr)) * 100
            #     f.write(f"  - {category}: {count} ({percentage:.1f}%)\n")
                
            # if irr_avg_values:
            #     f.write("\nAverage IRR values by category:\n")
            #     for category, avg in sorted(irr_avg_values.items(), key=lambda x: x[1], reverse=True):
            #         f.write(f"  - {category}: {avg:.2f}\n")
        
        # Create visualizations
        self._create_visualizations(categorized_sources, categorized_metrics, 
                           metric_avg_values, analyzer_dir)
        
        return output_file
    
    def _create_visualizations(self, 
                          categorized_sources: List[Dict], 
                          categorized_metrics: List[Dict],
                          metric_avg_values: Dict,
                          output_dir: str) -> None:
        """
        Create visualizations for the analysis results.
        
        Args:
            categorized_sources: List of dictionaries with categorized label sources
            categorized_metrics: List of dictionaries with categorized metrics
            metric_avg_values: Dictionary of average metric values by category
            output_dir: Directory to write output files
        """
        # [Keep the label source pie chart code as is]
        
        # 2. Metric categories bar chart
        if categorized_metrics:
            plt.figure(figsize=(12, 7))
            # A dataset can have multiple metrics, so count each occurrence
            metric_categories = []
            for item in categorized_metrics:
                metric_categories.extend(item['categories'])
            
            metric_counts = Counter(metric_categories)
            
            # Sort bars from tallest to shortest
            categories = []
            counts = []
            for category, count in sorted(metric_counts.items(), key=lambda x: x[1], reverse=True):
                categories.append(category)
                counts.append(count)
            
            plt.bar(categories, counts, color='skyblue')
            plt.xlabel('Metric Type')
            plt.ylabel('Number of Datasets')
            plt.title('Types of Metrics Used for IRR')
            plt.xticks(rotation=45, ha='right')
            
            # Add count labels on top of bars
            for i, v in enumerate(counts):
                plt.text(i, v + 0.1, str(v), ha='center')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metric_categories.png'), dpi=300)
            plt.close()
            
        # 3. Metric value distribution if we have numeric values
        metrics_with_values = []
        for item in categorized_metrics:
            for category, value in item['numeric_values'].items():
                metrics_with_values.append({
                    'dataset': item['dataset'],
                    'metric': item['metric'],
                    'category': category,
                    'value': value
                })
                
        if metrics_with_values and len(metrics_with_values) > 2:
            plt.figure(figsize=(12, 7))
            
            # Group by category
            categories = set(item['category'] for item in metrics_with_values)
            
            # Create boxplots for each category with sufficient data
            boxplot_data = []
            labels = []
            
            for category in categories:
                values = [item['value'] for item in metrics_with_values 
                        if item['category'] == category]
                if len(values) >= 2:  # Only include if we have at least 2 values
                    boxplot_data.append(values)
                    labels.append(category)
            
            if boxplot_data:
                plt.boxplot(boxplot_data, labels=labels)
                plt.ylabel('Metric Value')
                plt.title('Distribution of Metric Values by Type')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'metric_value_distribution.png'), dpi=300)
                plt.close()
        
        # 4. Average metric values by category
        if metric_avg_values:
            plt.figure(figsize=(12, 7))
            
            categories = []
            averages = []
            
            # Sort by average value (highest first)
            for category, avg in sorted(metric_avg_values.items(), key=lambda x: x[1], reverse=True):
                categories.append(category)
                averages.append(avg)
            
            bars = plt.bar(categories, averages, color='lightgreen')
            plt.xlabel('Metric Type')
            plt.ylabel('Average Value')
            plt.title('Average Metric Values by Type')
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom')
            
            plt.ylim(0, max(averages) * 1.2)
            plt.grid(axis='y', linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metric_average_values.png'), dpi=300)
            plt.close()
            
        # 5. Combined summary visualization
        plt.figure(figsize=(15, 10))
        
        # [Keep the label source pie chart code as is]
        
        # Metrics measures bar chart
        plt.subplot(2, 1, 2)
        if categorized_metrics:
            # A dataset can have multiple metrics, so count each occurrence
            metric_categories = []
            for item in categorized_metrics:
                metric_categories.extend(item['categories'])
            
            metric_counts = Counter(metric_categories)
            
            # Sort bars from tallest to shortest
            categories = []
            counts = []
            for category, count in sorted(metric_counts.items(), key=lambda x: x[1], reverse=True):
                categories.append(category)
                counts.append(count)
            
            plt.bar(categories, counts, color='skyblue')
            plt.xlabel('Metric Type')
            plt.ylabel('Number of Datasets')
            plt.title('Types of Metrics Used for IRR')
            plt.xticks(rotation=45, ha='right')
            
            # Add count labels on top of bars
            for i, v in enumerate(counts):
                plt.text(i, v + 0.1, str(v), ha='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'label_source_metric_summary.png'), dpi=300)
        plt.close()