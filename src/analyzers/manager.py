import os
from typing import Dict, Any, List, Optional

from analyzers.base import AnalyzerRegistry

class AnalysisManager:
    """
    Manager for running analysis on dataset information.
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize the analysis manager.
        
        Args:
            output_dir: Directory to write output files
        """
        self.output_dir = output_dir
        
        # Ensure the output directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Discover all available analyzers
        AnalyzerRegistry.discover_analyzers()
    
    def run_analyzer(self, analyzer_name: str, data: Dict[str, Dict[str, Any]]) -> str:
        """
        Run a specific analyzer.
        
        Args:
            analyzer_name: Name of the analyzer to run
            data: Dataset information to analyze
            
        Returns:
            Path to the output file
        """
        analyzer_class = AnalyzerRegistry.get_analyzer(analyzer_name)
        analyzer = analyzer_class()
        return analyzer.analyze(data, self.output_dir)
    
    def run_all_analyzers(self, data: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """
        Run all available analyzers.
        
        Args:
            data: Dataset information to analyze
            
        Returns:
            Dictionary mapping analyzer names to output file paths
        """
        results = {}
        
        for name, analyzer_class in AnalyzerRegistry.get_all_analyzers().items():
            analyzer = analyzer_class()
            output_file = analyzer.analyze(data, self.output_dir)
            results[name] = output_file
            
        return results
    
    def list_analyzers(self) -> List[Dict[str, str]]:
        """
        List all available analyzers.
        """
        analyzers = []
        
        for name, analyzer_class in AnalyzerRegistry.get_all_analyzers().items():
            analyzer = analyzer_class()
            analyzers.append({
                'name': name,
                'display_name': analyzer.get_name(),
                'description': analyzer.get_description()
            })
            
        return analyzers
