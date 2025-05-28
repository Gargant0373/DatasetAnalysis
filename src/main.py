import src.data_service as ds
import argparse
import csv
import os
from datetime import datetime
from src.analyzers.manager import AnalysisManager

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process and display dataset information')
    parser.add_argument('-c', '--csv', action='store_true', 
                        help='Generate a CSV file with all dataset information')
    parser.add_argument('-o', '--output', action='store_true',
                        help='Outputs data into the terminal')
    parser.add_argument('-a', '--analyze', action='store_true',
                        help='Analyze dataset information and output results to files')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Minimize terminal output, only show loading information')
    parser.add_argument('-l', '--list-analyzers', action='store_true',
                        help='List all available analyzers')
    parser.add_argument('-r', '--run-analyzer', 
                        help='Run a specific analyzer by name')
    return parser.parse_args()

def generate_csv(data, output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f'dataset_info.csv')
    
    all_keys = set()
    for dataset_info in data.values():
        all_keys.update(dataset_info.keys())
    
    if 'name' in all_keys:
        all_keys.remove('name')
    header = ['name'] + sorted(all_keys)
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        
        for dataset_name, dataset_info in data.items():
            if 'name' not in dataset_info:
                dataset_info['name'] = dataset_name
            writer.writerow(dataset_info)
    
    return output_file

def main():
    args = parse_arguments()
    data_service = ds.DataService()
    
    try:
        print("Fetching data from datasets.csv and tab3.csv...")
        data = data_service.fetch_data()
        
        detailed_datasets = 0
        basic_datasets = 0
        
        # Count datasets with detailed information
        for dataset_name, dataset_info in data.items():
            has_detailed_info = any(key for key in dataset_info.keys() if key not in ['name', 'period', 'citation_sum', 'status'])
            if has_detailed_info:
                detailed_datasets += 1
            else:
                basic_datasets += 1
        
        # Print summary information
        print(f"\nSummary:")
        print(f"Total datasets: {len(data)}")
        print(f"Datasets with detailed information: {detailed_datasets}")
        print(f"Datasets without detailed information: {basic_datasets}")

        # Print detailed information only if not in quiet mode and output is requested
        if args.output and not args.quiet:
            print("\nDataset Information:")
            print("-" * 70)
            
            for dataset_name, dataset_info in data.items():
                print(f"Dataset: {dataset_name}")
                
                # Check if this dataset has detailed info from tab3.csv
                has_detailed_info = any(key for key in dataset_info.keys() if key not in ['name', 'period', 'citation_sum', 'status'])
                
                if not has_detailed_info:
                    print("  [WARNING] No detailed information found in tab3.csv for this dataset")
                
                # Print dataset information
                for key, value in dataset_info.items():
                    if key != 'name':
                        print(f"  {key}: {value}")
                print("-" * 70)

        # Generate CSV file if requested
        if args.csv:
            output_file = generate_csv(data)
            print(f"\nCSV file generated: {output_file}")
        
        # Initialize the analysis manager
        analysis_manager = AnalysisManager(output_dir='outputs')
        
        # List all available analyzers if requested
        if args.list_analyzers:
            analyzers = analysis_manager.list_analyzers()
            print("\nAvailable Analyzers:")
            print("-" * 70)
            for analyzer in analyzers:
                print(f"Name: {analyzer['name']}")
                print(f"Display Name: {analyzer['display_name']}")
                print(f"Description: {analyzer['description']}")
                print("-" * 70)
        
        # Run a specific analyzer if requested
        if args.run_analyzer:
            try:
                output_file = analysis_manager.run_analyzer(args.run_analyzer, data)
                print(f"\nAnalyzer '{args.run_analyzer}' completed. Results saved to: {output_file}")
            except KeyError as e:
                print(f"Error: {e}")
                print("Use --list-analyzers to see available analyzers.")
        
        # Run all analyzers if requested
        if args.analyze:
            results = analysis_manager.run_all_analyzers(data)
            print("\nAnalysis completed. Results saved to:")
            for analyzer_name, output_file in results.items():
                print(f"- {analyzer_name}: {output_file}")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()