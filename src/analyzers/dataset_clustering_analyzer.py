"""
Analyzer for clustering datasets based on their characteristics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
from analyzers.base import register_analyzer

MAX_PCA_COMPONENTS = 20 
PCA_COMPONENT_CONFIGS = [2, 3, 5, 10, 15, MAX_PCA_COMPONENTS]

@register_analyzer
class DatasetClusteringAnalyzer:
    """Analyzer that clusters datasets based on their characteristics."""
    
    @staticmethod
    def get_name():
        return "dataset_clustering"
    
    @staticmethod
    def get_description():
        return "Groups datasets into clusters based on their characteristics and documentation patterns"
    
    @staticmethod
    def analyze(data, output_dir='outputs'):
        """
        Cluster datasets based on their characteristics.
        
        Args:
            data: The dataset information to analyze
            output_dir: Directory to write output files
            
        Returns:
            Path to the output file
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        output_file = os.path.join(output_dir, 'dataset_clustering_analysis.txt')
        plot_file_base = os.path.join(output_dir, 'dataset_clustering_plot')
        
        # Convert data to a pandas DataFrame for easier manipulation
        df = DatasetClusteringAnalyzer._prepare_dataframe(data)
        
        # Check if we have enough data points for clustering
        if len(df) < 3:
            with open(output_file, 'w') as f:
                f.write("Not enough datasets for clustering analysis (minimum 3 required).")
            return output_file
        
        # Perform clustering
        n_clusters, labels, silhouette_avg, feature_importance = DatasetClusteringAnalyzer._perform_clustering(df)
        
        # Test different numbers of PCA components and visualize
        pca_plots = {}
        
        # Filter component configs based on the available dimensions
        max_possible_components = min(MAX_PCA_COMPONENTS, len(df.columns))
        valid_component_configs = [n for n in PCA_COMPONENT_CONFIGS if n <= max_possible_components]
        
        for n_components in valid_component_configs:
            component_plot_file = f"{plot_file_base}_{n_components}components.png"
            pca_plots[n_components] = component_plot_file
            DatasetClusteringAnalyzer._visualize_clusters(df, labels, n_clusters, component_plot_file, n_components)
        
        # Write analysis results
        DatasetClusteringAnalyzer._write_analysis(output_file, df, labels, n_clusters, 
                                                silhouette_avg, feature_importance, data, pca_plots, max_possible_components)
        
        return output_file
    
    @staticmethod
    def _prepare_dataframe(data):
        """Convert the dataset information to a DataFrame for analysis."""
        # Identify all possible fields across all datasets
        all_fields = set()
        standard_fields = {'name', 'period', 'citation_sum', 'status', 
                          'DS DOI', 'DS Citation', 'Done for?', 'Empty', 
                          'Outcome', 'Link to Data', 'Notes', 'Additional Links'}
        
        for dataset_info in data.values():
            for field in dataset_info.keys():
                if field not in standard_fields:
                    all_fields.add(field)
        
        # Extract features from dataset information
        features = []
        dataset_names = []
        
        for dataset_name, dataset_info in data.items():
            dataset_names.append(dataset_name)
            
            # Extract numeric features
            citation_count = 0
            if 'citation_sum' in dataset_info and dataset_info['citation_sum']:
                try:
                    citation_count = int(dataset_info['citation_sum'])
                except (ValueError, TypeError):
                    pass
            
            # Count missing fields
            missing_fields = 0
            total_fields = 0
            
            for field, value in dataset_info.items():
                if field not in standard_fields:
                    total_fields += 1
                    # Check if field is missing
                    if value is None or value == "" or value == "N/A" or "No information" in str(value) or "Not applicable" in str(value):
                        missing_fields += 1
            
            # Calculate missing percentage
            missing_percentage = 0
            if total_fields > 0:
                missing_percentage = (missing_fields / total_fields) * 100
            
            # Extract period information (if available)
            period_year = 0
            if 'period' in dataset_info and dataset_info['period']:
                # Try to extract a year from period field
                period_str = str(dataset_info['period'])
                
                # Look for 4-digit years
                import re
                year_match = re.search(r'(\d{4})', period_str)
                if year_match:
                    period_year = int(year_match.group(1))
            
            # Start with base features
            feature_vector = [
                citation_count,
                missing_percentage,
                period_year,
            ]
            
            # Add binary features for all possible fields
            for field in sorted(all_fields):
                # Check if the field exists and has meaningful content
                has_field = 1 if (field in dataset_info and 
                                  dataset_info[field] and 
                                  dataset_info[field] != "N/A" and 
                                  "No information" not in str(dataset_info[field]) and
                                  "Not applicable" not in str(dataset_info[field])) else 0
                feature_vector.append(has_field)
            
            features.append(feature_vector)
        
        # Create column names
        columns = [
            'citation_count', 
            'missing_percentage', 
            'period_year',
        ]
        
        # Add binary feature names
        for field in sorted(all_fields):
            columns.append(f'has_{field.lower().replace(" ", "_")}')
        
        df = pd.DataFrame(features, index=dataset_names, columns=columns)
        
        # Remove any rows with NaN values
        df = df.fillna(0)
        
        return df
    
    @staticmethod
    def _perform_clustering(df):
        """Perform KMeans clustering on the dataset."""
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df.values)
        
        # Determine optimal number of clusters using silhouette score
        silhouette_scores = []
        max_clusters = min(10, len(df) - 1)
        range_n_clusters = range(2, max_clusters + 1)
        
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_features)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(scaled_features, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Choose optimal number of clusters
        if len(silhouette_scores) > 0:
            optimal_clusters = range_n_clusters[np.argmax(silhouette_scores)]
            best_silhouette = max(silhouette_scores)
        else:
            # Default to 3 clusters if silhouette method fails
            optimal_clusters = 3
            best_silhouette = 0
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(scaled_features)
        
        # Calculate feature importance based on cluster centers
        importance = {}
        for i, column in enumerate(df.columns):
            # Calculate the spread of cluster centers for this feature
            center_variance = np.var(kmeans.cluster_centers_[:, i])
            importance[column] = center_variance
        
        # Normalize importance values
        total_importance = sum(importance.values())
        if total_importance > 0:
            for column in importance:
                importance[column] = (importance[column] / total_importance) * 100
        
        # Sort by importance
        feature_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return optimal_clusters, labels, best_silhouette, feature_importance
    
    @staticmethod
    def _visualize_clusters(df, labels, n_clusters, plot_file, n_components=2):
        """Create a visualization of the clusters using PCA for dimensionality reduction.
        
        Args:
            df: DataFrame containing dataset features
            labels: Cluster labels for each dataset
            n_clusters: Number of clusters
            plot_file: Path to save the plot
            n_components: Number of PCA components to use (default: 2)
        """
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df.values)
        
        # Apply PCA with the specified number of components
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_features)
        
        # Calculate explained variance for all components
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # For 2D plots, we only need the first 2 components
        plot_components = min(2, n_components)
        
        # Create dataframe for plotting
        pc_columns = [f'PC{i+1}' for i in range(plot_components)]
        pca_df = pd.DataFrame(data=principal_components[:, :plot_components], 
                            columns=pc_columns,
                            index=df.index)
        pca_df['Cluster'] = labels
        
        # Create plot
        plt.figure(figsize=(12, 10))
        
        # Colors for each cluster
        colors = plt.cm.rainbow(np.linspace(0, 1, n_clusters))
        
        # Plot each cluster
        for i in range(n_clusters):
            cluster_data = pca_df[pca_df['Cluster'] == i]
            plt.scatter(
                cluster_data['PC1'], 
                cluster_data['PC2'],
                s=100, 
                c=colors[i].reshape(1, -1),
                label=f'Cluster {i+1}'
            )
        
        # Add dataset names as annotations
        for i, txt in enumerate(pca_df.index):
            plt.annotate(
                txt, 
                (pca_df.iloc[i, 0], pca_df.iloc[i, 1]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
        
        plt.title(f'Dataset Clusters based on Documentation Patterns and Citation Count')
        plt.xlabel(f'Principal Component 1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'Principal Component 2 ({explained_variance[1]:.2%} variance)')
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save the main plot
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def _write_analysis(output_file, df, labels, n_clusters, silhouette_score, feature_importance, raw_data, pca_plots, max_components):
        """Write clustering analysis results to the output file.
        
        Args:
            output_file: Path to the output file
            df: DataFrame containing dataset features
            labels: Cluster labels for each dataset
            n_clusters: Number of clusters
            silhouette_score: Silhouette score of the clustering
            feature_importance: List of (feature, importance) tuples
            raw_data: Original dataset information
            pca_plots: Dictionary mapping number of components to plot files
            max_components: Maximum number of PCA components used
        """
        with open(output_file, 'w') as f:
            f.write("=== Dataset Clustering Analysis ===\n\n")
            
            f.write(f"Number of clusters identified: {n_clusters}\n")
            f.write(f"Silhouette score: {silhouette_score:.4f} (closer to 1 is better)\n\n")
            
            # Calculate PCA explained variance
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(df.values)
            
            # Get the maximum number of components we can use
            n_components = min(max_components, len(df.columns))
            pca = PCA(n_components=n_components)
            pca.fit_transform(scaled_features)
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            f.write("PCA Explained Variance:\n")
            f.write("-" * 50 + "\n")
            
            # Report on all available components
            for i, var in enumerate(explained_variance):
                f.write(f"Principal Component {i+1}: {var:.2%} variance explained\n")
                
                # Add cumulative variance milestones
                if (i+1) in [5, 10, 15, 20, 25, 30] or i == len(explained_variance)-1:
                    f.write(f"Cumulative variance with {i+1} components: {cumulative_variance[i]:.2%}\n\n")
            
            # If we didn't add a final cumulative variance, add it now
            if not any(i == len(explained_variance)-1 for i in [4, 9, 14, 19, 24, 29]):
                f.write(f"Cumulative variance with {len(explained_variance)} components: {cumulative_variance[-1]:.2%}\n\n")
            
            # Feature importance
            f.write("Feature Importance for Clustering:\n")
            f.write("-" * 50 + "\n")
            for feature, importance in feature_importance:
                f.write(f"{feature}: {importance:.2f}%\n")
            
            # Add cluster information to DataFrame
            df_with_clusters = df.copy()
            df_with_clusters['cluster'] = labels
            
            # Analysis for each cluster
            f.write("\nCluster Analysis:\n")
            f.write("-" * 50 + "\n")
            
            for i in range(n_clusters):
                cluster_df = df_with_clusters[df_with_clusters['cluster'] == i]
                cluster_size = len(cluster_df)
                
                f.write(f"\nCluster {i+1} ({cluster_size} datasets):\n")
                
                # Cluster characteristics
                f.write("\nCharacteristics:\n")
                for column in df.columns:
                    cluster_mean = cluster_df[column].mean()
                    overall_mean = df[column].mean()
                    difference = ((cluster_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else 0
                    
                    # Determine if this feature is higher or lower than average
                    comparison = "higher" if difference > 0 else "lower"
                    f.write(f"- {column}: {cluster_mean:.2f} ({abs(difference):.1f}% {comparison} than average)\n")
                
                # Datasets in this cluster
                f.write("\nDatasets in this cluster:\n")
                for dataset in cluster_df.index:
                    citation_count = raw_data[dataset].get('citation_sum', 'N/A')
                    f.write(f"- {dataset} (Citations: {citation_count})\n")
                
                f.write("-" * 40 + "\n")
            
            # Suggestions for documentation improvements
            f.write("\nSuggestions for Documentation Improvements:\n")
            f.write("-" * 50 + "\n")
            
            # Analyze each cluster for missing documentation
            for i in range(n_clusters):
                cluster_df = df_with_clusters[df_with_clusters['cluster'] == i]
                
                # Check if this cluster has systematically missing documentation
                if cluster_df['missing_percentage'].mean() > df['missing_percentage'].mean():
                    f.write(f"\nCluster {i+1} Improvement Suggestions:\n")
                    
                    # Get datasets in this cluster
                    datasets_in_cluster = cluster_df.index.tolist()
                    
                    # Find commonly missing fields in this cluster
                    missing_field_counts = {}
                    
                    for dataset in datasets_in_cluster:
                        dataset_info = raw_data[dataset]
                        
                        # Skip standard metadata fields
                        standard_fields = {'name', 'period', 'citation_sum', 'status'}
                        
                        for field, value in dataset_info.items():
                            if field not in standard_fields:
                                # Check if field is missing
                                if field not in missing_field_counts:
                                    missing_field_counts[field] = 0
                                
                                if value is None or value == "" or value == "N/A" or "No information" in str(value) or "Not applicable" in str(value):
                                    missing_field_counts[field] += 1
                    
                    # Calculate percentage missing for each field
                    missing_percentages = {}
                    for field, count in missing_field_counts.items():
                        percentage = (count / len(datasets_in_cluster)) * 100
                        missing_percentages[field] = percentage
                    
                    # Get top 5 missing fields
                    top_missing = sorted(missing_percentages.items(), key=lambda x: x[1], reverse=True)[:5]
                    
                    if top_missing:
                        f.write("Commonly missing fields in this cluster:\n")
                        for field, percentage in top_missing:
                            if percentage > 50:  # Only suggest fields that are missing in >50% of datasets
                                f.write(f"- {field}: Missing in {percentage:.1f}% of datasets in this cluster\n")
                        
                        # Suggest looking at other clusters as examples
                        other_clusters = [j for j in range(n_clusters) if j != i]
                        better_clusters = []
                        
                        for j in other_clusters:
                            other_cluster_df = df_with_clusters[df_with_clusters['cluster'] == j]
                            if other_cluster_df['missing_percentage'].mean() < cluster_df['missing_percentage'].mean():
                                better_clusters.append(j)
                        
                        if better_clusters:
                            f.write("\nExample datasets with better documentation:\n")
                            for j in better_clusters:
                                other_cluster_df = df_with_clusters[df_with_clusters['cluster'] == j]
                                # Get the dataset with lowest missing percentage
                                best_documented = other_cluster_df.loc[other_cluster_df['missing_percentage'].idxmin()]
                                dataset_name = best_documented.name
                                f.write(f"- {dataset_name} (from Cluster {j+1})\n")
            
            # Add information about the generated visualizations
            f.write("\nVisualizations generated:\n")
            for n_components, plot_path in pca_plots.items():
                f.write(f"- {os.path.basename(plot_path)}: PCA visualization with {n_components} components\n")
        
        return output_file
