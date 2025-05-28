# Analyzer Framework Documentation

This document describes how to use the modular analyzer framework to create custom analyzers for the Dataset Curating application.

## Overview

The analyzer framework allows you to create custom analyzers that can analyze dataset information and generate reports. 
All analyzers are automatically discovered and registered, making it easy to add new functionality without modifying existing code.

## Basic Usage

To use existing analyzers, you can run the application with the following command-line arguments:

```bash
# List all available analyzers
python -m src.main --list-analyzers

# Run all available analyzers
python -m src.main --analyze

# Run a specific analyzer by name (e.g., MissingInformationAnalyzer)
python -m src.main --run-analyzer MissingInformationAnalyzer
```

## Creating a New Analyzer

To create a new analyzer, follow these steps:

1. Create a new Python file in the `src/analyzers` directory
2. Import the necessary components from the base analyzer module
3. Create a new analyzer class that inherits from the `Analyzer` base class
4. Implement the required methods
5. Register your analyzer using the `@register_analyzer` decorator

### Example

Here's a simple example of a new analyzer that counts datasets by time period:

```python
"""
Analyzer for counting datasets by time period.
"""
import os
from typing import Dict, Any, List
from collections import Counter

from src.analyzers.base import Analyzer, register_analyzer


@register_analyzer
class TimeperiodAnalyzer(Analyzer):
    """
    Analyzer that counts datasets by time period.
    """
    
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        return "Time Period Analyzer"
    
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        return "Analyzes the distribution of datasets across different time periods."
    
    def analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str:
        """
        Analyze the dataset time periods.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, 'timeperiod_analysis.txt')
        
        # Count datasets by time period
        time_periods = Counter()
        
        for dataset_name, dataset_info in data.items():
            if 'period' in dataset_info and dataset_info['period']:
                time_periods[dataset_info['period']] += 1
        
        # Write results to output file
        with open(output_file, 'w') as f:
            f.write("=== Time Period Analysis ===\n\n")
            
            f.write("Datasets by Time Period:\n")
            f.write("-" * 50 + "\n")
            
            for period, count in sorted(time_periods.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(data)) * 100
                f.write(f"{period}: {count} datasets ({percentage:.2f}%)\n")
        
        return output_file
```

### Required Methods

Each analyzer must implement the following methods:

1. `get_name(self) -> str`: Returns a human-readable name for the analyzer
2. `get_description(self) -> str`: Returns a description of what the analyzer does
3. `analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str`: Performs the analysis and returns the path to the output file

## Analyzer Data Format

The `data` parameter passed to the `analyze` method is a dictionary with the following structure:

```python
{
    'dataset_name1': {
        'name': 'dataset_name1',
        'period': '1990-2000',
        'citation_sum': 123,
        'status': 'Done',
        # Additional fields from tab3.csv
    },
    'dataset_name2': {
        # ...
    }
}
```

## Best Practices

1. **Give your analyzer a descriptive name**: The name should clearly indicate what the analyzer does.
2. **Write a clear description**: The description should explain what insights the analyzer provides.
3. **Handle missing data**: Your analyzer should handle cases where fields are missing or contain invalid data.
4. **Format output clearly**: Make your output file easy to read with clear sections and headings.
5. **Follow the existing pattern**: Look at the `MissingInformationAnalyzer` for an example of good analyzer design.

## Advanced Features

You can also create analyzers that generate different types of output files, such as CSV files, JSON files, or even HTML reports. Just make sure to return the path to the output file from the `analyze` method.
