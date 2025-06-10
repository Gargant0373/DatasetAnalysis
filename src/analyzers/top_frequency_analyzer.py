import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from analyzers.base import Analyzer, register_analyzer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

@register_analyzer
class TopDatasetFrequencyAnalyzer(Analyzer):
    def get_name(self) -> str:
        return "Top Dataset Frequency Table Generator"

    def get_description(self) -> str:
        return "Generates table-style images of top-20 dataset frequencies by period and overall."

    def _get_cosine_sim(self, dict1, dict2):
        # Union of keys
        keys = list(set(dict1.keys()).union(dict2.keys()))
        v1 = np.array([dict1.get(k, 0) for k in keys]).reshape(1, -1)
        v2 = np.array([dict2.get(k, 0) for k in keys]).reshape(1, -1)
        return cosine_similarity(v1, v2)[0][0]


    def analyze(self, data, output_dir: str = 'outputs') -> str:
        analyzer_dir = self.get_analyzer_output_dir(output_dir)

        # Load CSV files
        period_path = os.path.join('data', 'dataset_frequency_by_period.csv')
        overall_path = os.path.join('data', 'dataset_frequency_overall.csv')

        if not os.path.exists(period_path) or not os.path.exists(overall_path):
            return analyzer_dir

        period_df = pd.read_csv(period_path)
        overall_df = pd.read_csv(overall_path)

        # Generate table image for each period
        for period in sorted(period_df['Period'].unique()):
            subset = period_df[period_df['Period'] == period].sort_values(by='Count', ascending=False).head(20)[['Dataset', 'Count']]
            output_path = os.path.join(analyzer_dir, f"top20_period_{period}_table.png")
            self._plot_table(subset, f"Top 20 Datasets - Period {period}", output_path)

        # Generate table image for overall dataset
        top_overall = overall_df.sort_values(by='Count', ascending=False).head(20)[['Dataset', 'Count']]
        output_path = os.path.join(analyzer_dir, "top20_overall_table.png")
        self._plot_table(top_overall, "Top 20 Datasets - Overall", output_path)
        output_path = os.path.join(analyzer_dir, "overlap_report.txt")
        self._write_report(period_df, output_path)

        return analyzer_dir

    def _write_report(self, df, filepath):
        overlap_report = ""

        periods_to_compare = [2, 5, 15]
        top20_by_period = {}

        # Extract top-20 datasets for each period of interest
        for period in periods_to_compare:
            period_subset = df[df['Period'] == period]
            overlap_report += (
                f"Period {period}: {len(period_subset.index)} unique datasets\n"
            )
            top20_subset = period_subset.sort_values(by='Count', ascending=False).head(20)
            top20_by_period[period] = set(top20_subset['Dataset'].tolist())

        overlap_report += "\n"

        table_data = {
            '5 - 15': [],
            '2 - 5': [],
            '2 - 15': []
        }

        pairs = [ (2,5), (2,15), (5,15) ]
        for pair_str, (p1, p2) in zip(['2 - 5', '2 - 15', '5 - 15'], pairs):
            top1 = df[df['Period'] == p1].sort_values(by='Count', ascending=False).head(20)
            top2 = df[df['Period'] == p2].sort_values(by='Count', ascending=False).head(20)
            top_set1 = set(top1['Dataset'])
            top_set2 = set(top2['Dataset'])
            overlap_top = top_set1 & top_set2
            jaccard_top = len(overlap_top) / len(top_set1 | top_set2) if (top_set1 | top_set2) else 0.0
            # Build dicts for cosine
            d1 = dict(zip(top1['Dataset'], top1['Count']))
            d2 = dict(zip(top2['Dataset'], top2['Count']))
            cosine_top = self._get_cosine_sim(d1, d2)

            overlap_report += (
                f"Overlap between Period {p1} and Period {p2} in top 20:\n"
                f"Count: {len(overlap_top)}\n"
                f"Jaccard similarity: {jaccard_top:.3f}\n"
                f"Cosine similarity: {cosine_top:.3f}\n"
                f"Datasets: {sorted(list(overlap_top))}\n\n"
            )

            # General sets and counts
            all_s1 = df[df['Period'] == p1]
            all_s2 = df[df['Period'] == p2]
            all_set1 = set(all_s1['Dataset'])
            all_set2 = set(all_s2['Dataset'])
            overlap_all = all_set1 & all_set2
            jaccard_all = len(overlap_all) / len(all_set1 | all_set2) if (all_set1 | all_set2) else 0.0
            d1_all = dict(zip(all_s1['Dataset'], all_s1['Count']))
            d2_all = dict(zip(all_s2['Dataset'], all_s2['Count']))
            cosine_all = self._get_cosine_sim(d1_all, d2_all)

            overlap_report += (
                f"Overlap between Period {p1} and Period {p2} generally:\n"
                f"Count: {len(overlap_all)}\n"
                f"Jaccard similarity: {jaccard_all:.3f}\n"
                f"Cosine similarity: {cosine_all:.3f}\n"
                f"Datasets: {sorted(list(overlap_all))}\n\n"
            )

            table_data[pair_str] = [
                cosine_all,
                len(overlap_all),
                cosine_top,
                len(overlap_top)
            ]

        table_rows = ['overall cos', 'overall common', 'top20 cos', 'top20 common']
        self._plot_table_similarity(table_data, table_rows, filepath)

        # Save overlap report to a text file
        with open(filepath, 'w') as f:
            f.write(overlap_report)
        
    def _plot_table_similarity(self, df, table_rows, filepath):
        # Build DataFrame
        summary_df = pd.DataFrame(df, index=table_rows)

        # Plot as an image
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.axis('off')
        tbl = ax.table(
            cellText=summary_df.round(3).reset_index().values,
            colLabels=[''] + list(summary_df.columns),
            rowLabels=None,
            loc='center',
            cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        plt.tight_layout()

        table_img_path = filepath.replace('.txt', '_summary.png')
        plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_table(self, df, title, filepath):
        fig, ax = plt.subplots(figsize=(1.7, len(df) * 0.23))
        ax.axis('off')

        table_data = [df.columns.tolist()] + df.values.tolist()
        table = ax.table(cellText=table_data, colLabels=None, loc='center', cellLoc='left', colWidths=[1.0,0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        # table.scale(0.9, 1.0)

        #ax.set_title(titl, fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
