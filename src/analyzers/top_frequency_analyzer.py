import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from analyzers.base import Analyzer, register_analyzer

@register_analyzer
class TopDatasetFrequencyAnalyzer(Analyzer):
    def get_name(self) -> str:
        return "Top Dataset Frequency Table Generator"

    def get_description(self) -> str:
        return "Generates table-style images of top-20 dataset frequencies by period and overall."

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

        return analyzer_dir

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
