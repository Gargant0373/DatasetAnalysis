import os
import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Type


class Analyzer(ABC):
    """Base class for all analyzers."""
    
    @abstractmethod
    def analyze(self, data: Dict[str, Dict[str, Any]], output_dir: str = 'outputs') -> str:
        """
        Analyze the dataset and write results to a file.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the human-readable name of the analyzer."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Get a description of what the analyzer does."""
        pass


class AnalyzerRegistry:
    """Registry for all analyzer classes."""
    
    _analyzers: Dict[str, Type[Analyzer]] = {}
    
    @classmethod
    def register(cls, analyzer_class: Type[Analyzer]) -> Type[Analyzer]:
        """
        Register an analyzer class.
        
        Args:
            analyzer_class: The analyzer class to register
            
        Returns:
            The registered analyzer class
        """
        # Use the class name as the key
        key = analyzer_class.__name__
        cls._analyzers[key] = analyzer_class
        return analyzer_class
    
    @classmethod
    def get_all_analyzers(cls) -> Dict[str, Type[Analyzer]]:
        """
        Get all registered analyzers.
        
        Returns:
            Dictionary of analyzer class names to analyzer classes
        """
        return cls._analyzers
    
    @classmethod
    def get_analyzer(cls, name: str) -> Type[Analyzer]:
        """
        Get an analyzer by name.
        
        Args:
            name: The name of the analyzer class
            
        Returns:
            The analyzer class
        
        Raises:
            KeyError: If no analyzer with the given name is registered
        """
        if name not in cls._analyzers:
            raise KeyError(f"No analyzer named '{name}' is registered")
        return cls._analyzers[name]
    
    @classmethod
    def discover_analyzers(cls) -> None:
        """
        Discover and register all analyzer classes in the analyzers package.
        """
        # Import the analyzers package
        import analyzers
        
        # Discover all modules in the analyzers package
        for _, name, is_pkg in pkgutil.iter_modules(analyzers.__path__, analyzers.__name__ + '.'):
            if not is_pkg:  # Skip packages, only import modules
                try:
                    # Import the module
                    importlib.import_module(name)
                except ImportError as e:
                    print(f"Failed to import analyzer module {name}: {e}")


def register_analyzer(cls):
    """
    Decorator to register an analyzer class.
    
    Args:
        cls: The analyzer class to register
        
    Returns:
        The registered analyzer class
    """
    return AnalyzerRegistry.register(cls)
