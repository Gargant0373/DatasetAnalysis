import os
import pandas as pd
import matplotlib.pyplot as plt
from analyzers.base import Analyzer, register_analyzer

@register_analyzer
class PaperDatasetStatsAnalyzer(Analyzer):
    def get_name(self) -> str:
        return "Paper-Dataset Statistics Table Generator"

    def get_description(self) -> str:
        return (
            "Computes paper-dataset statistics (average datasets per paper, "
            "unique/total datasets used, and their ratio) for a given venue, "
            "by period and overall. Outputs the result as a table image."
        )

    def analyze(self, data, output_dir: str = 'outputs') -> str:
        data_path = os.path.join('data', 'tab2.csv')
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        venue = 'ACL'

        if not os.path.exists(data_path):
            print(f"File {data_path} does not exist. Skipping analysis.")
            return analyzer_dir

        os.makedirs(analyzer_dir, exist_ok=True)
        
        analyzer_dir = self.get_analyzer_output_dir(output_dir)
        venue = 'ACL'
        os.makedirs(analyzer_dir, exist_ok=True)

        # Load CSV file
        df = pd.read_csv(data_path, sep=None, engine='python')

        # Filter by venue
        df = df[df['venue'] == venue]

        periods = [2, 5, 15]
        stats = {}
        # Function to compute stats for a given frame
        def compute_stats(frame):
            if frame.empty:
                return [0, 0, 0, 0]
            # 1. avg datasets per paper
            paper_to_datasets = frame.groupby('paper name')['dataset lowercase'].nunique()
            avg_datasets_per_paper = paper_to_datasets.mean()
            # 2. unique datasets used
            n_unique_datasets = frame['dataset lowercase'].nunique()
            # 3. total datasets used (rows)
            total_datasets_used = frame.shape[0]
            # 4. ratio
            ratio = n_unique_datasets / total_datasets_used if total_datasets_used else 0
            return [round(avg_datasets_per_paper, 3), n_unique_datasets, total_datasets_used, round(ratio, 3)]

        # Per period
        for period in periods:
            stats[str(period)] = compute_stats(df[df['period'] == period])
        # Overall
        stats['overall'] = compute_stats(df)

        # Build DataFrame for pretty table
        index = [
            "Avg datasets",
            "Unique datasets",
            "Total datasets",
            "Unique/Total"
        ]
        stats_df = pd.DataFrame(stats, index=index)

        # Plot as table image
        fig, ax = plt.subplots(figsize=(5, 2.2))
        ax.axis('off')

        def pretty_num(x):
            if isinstance(x, float) and x.is_integer():
                return str(int(x))
            return str(x)

        table_values = [[pretty_num(cell) for cell in row] for row in stats_df.values]

        tbl = ax.table(
            cellText=table_values,
            rowLabels=stats_df.index,
            colLabels=stats_df.columns,
            loc='center',
            cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.2, 1.2)
        plt.tight_layout()

        table_img_path = os.path.join(analyzer_dir, f"paper_dataset_stats_{venue}.png")
        plt.savefig(table_img_path, dpi=300, bbox_inches='tight')
        plt.close()


        return analyzer_dir
