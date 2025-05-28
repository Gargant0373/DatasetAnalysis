import pandas as pd
import os

class DataService():
    def __init__(self, datasets="datasets.csv", data_source="tab3.csv"):
        self.datasets = os.path.join('data', datasets)
        self.data_source = os.path.join('data', data_source)
        self.data = {}

    def fetch_data(self):
        """Load and process data from datasets.csv and tab3.csv"""
        if not os.path.exists(self.datasets) or not os.path.exists(self.data_source):
            raise FileNotFoundError(f"Data files not found at {self.datasets} or {self.data_source}")
        
        # Load and filter datasets.csv
        datasets_df = pd.read_csv(self.datasets)
        datasets_df = datasets_df[datasets_df['Done?'] != "Waiting"]
        
        # Load all data from tab3.csv
        data_source_df = pd.read_csv(self.data_source)
        
        # Print summary information
        print(f"Number of datasets in datasets.csv: {len(datasets_df)}")
        print(f"Number of datasets in tab3.csv: {len(data_source_df)}")
        
        self.process_datasets(datasets_df, data_source_df)
        
        return self.data
        
    def process_datasets(self, datasets_df, data_source_df):
        """Match datasets between datasets.csv and tab3.csv and extract detailed information"""
        # Create normalized versions of the dataset names for matching
        datasets_df['Dataset_normalized'] = datasets_df['Dataset'].str.lower().str.strip()
        data_source_df['DS_Descriptor_normalized'] = data_source_df['DS Descriptor'].str.lower().str.strip()
        
        for _, dataset_row in datasets_df.iterrows():
            dataset_name = dataset_row['Dataset']
            dataset_name_normalized = dataset_row['Dataset_normalized']
            
            # Add basic dataset information
            self.data[dataset_name] = {
                'name': dataset_name,
                'period': dataset_row['Period'],
                'citation_sum': dataset_row['Citation Sum'],
                'status': dataset_row['Done?']
            }
            
            matching_rows = data_source_df[data_source_df['DS_Descriptor_normalized'] == dataset_name_normalized]
            if not matching_rows.empty:
                source_row = matching_rows.iloc[0]
                self.data[dataset_name].update(self.process_data(source_row))
            
    def process_data(self, data):
        """Extract relevant fields from a row in tab3.csv"""
        processed_data = {}
        for column, value in data.items():
            if column != 'DS Descriptor' and not column.startswith('DS_Descriptor_'):
                processed_data[column] = value
        
        return processed_data