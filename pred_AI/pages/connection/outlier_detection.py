import os
from flask import current_app, jsonify, request, session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib

from pages.connection.data_ops import convert_for_json, safe_read_file, save_dataframe_to_file
from pages.connection.data_processing import remove_outliers
matplotlib.use('Agg')

def detect_outliers(df, column, method="IQR"):
    """
    Detect outliers in a DataFrame column using specified method.
    Returns outlier indices and statistics.
    """
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {
            'outlier_indices': [],
            'total_outliers': 0,
            'outlier_percentage': 0,
            'bounds': None,
            'error': 'Column is not numeric'
        }
    
    outlier_indices = []
    bounds = {}
    
    if method == "IQR":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_indices = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index.tolist()
        bounds = {'lower': lower_bound, 'upper': upper_bound, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR}
    
    elif method == "Z-score":
        if df[column].std() == 0:
            return {
                'outlier_indices': [],
                'total_outliers': 0,
                'outlier_percentage': 0,
                'bounds': None,
                'error': 'Standard deviation is zero'
            }
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outlier_indices = df[z_scores > 3].index.tolist()
        bounds = {'mean': df[column].mean(), 'std': df[column].std(), 'threshold': 3}
    
    elif method == "percentile":
        lower = df[column].quantile(0.05)
        upper = df[column].quantile(0.95)
        outlier_indices = df[(df[column] < lower) | (df[column] > upper)].index.tolist()
        bounds = {'lower': lower, 'upper': upper}
    
    total_outliers = len(outlier_indices)
    outlier_percentage = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
    
    return {
        'outlier_indices': outlier_indices,
        'total_outliers': total_outliers,
        'outlier_percentage': outlier_percentage,
        'bounds': bounds,
        'method': method
    }

def plot_outliers_detection(df, column, method="IQR", figsize=(12, 8)):
    """
    Create comprehensive outlier detection visualization.
    Returns base64 encoded image string.
    """
    try:
        if not pd.api.types.is_numeric_dtype(df[column]):
            return None, "Column is not numeric"
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle(f'Outlier Detection: {column} (Method: {method})', fontsize=16, fontweight='bold')
        
        # Get outlier detection results
        outlier_info = detect_outliers(df, column, method)
        outlier_indices = outlier_info['outlier_indices']
        
        # Create masks for normal and outlier data
        is_outlier = df.index.isin(outlier_indices)
        normal_data = df[~is_outlier][column]
        outlier_data = df[is_outlier][column]
        
        # 1. Box Plot (top-left)
        ax1 = axes[0, 0]
        box_data = [normal_data.dropna(), outlier_data.dropna()] if len(outlier_data) > 0 else [normal_data.dropna()]
        box_labels = ['Normal', 'Outliers'] if len(outlier_data) > 0 else ['All Data']
        colors = ['lightblue', 'red'] if len(outlier_data) > 0 else ['lightblue']
        
        bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Box Plot Comparison')
        ax1.set_ylabel('Values')
        ax1.grid(True, alpha=0.3)
        
        # 2. Histogram with outliers highlighted (top-right)
        ax2 = axes[0, 1]
        ax2.hist(normal_data.dropna(), bins=30, alpha=0.7, color='lightblue', label='Normal', density=True)
        if len(outlier_data) > 0:
            ax2.hist(outlier_data.dropna(), bins=20, alpha=0.8, color='red', label='Outliers', density=True)
        
        # Add threshold lines based on method
        if method == "IQR" and outlier_info['bounds']:
            ax2.axvline(outlier_info['bounds']['lower'], color='red', linestyle='--', alpha=0.8, label='Lower Bound')
            ax2.axvline(outlier_info['bounds']['upper'], color='red', linestyle='--', alpha=0.8, label='Upper Bound')
        elif method == "percentile" and outlier_info['bounds']:
            ax2.axvline(outlier_info['bounds']['lower'], color='red', linestyle='--', alpha=0.8, label='5th Percentile')
            ax2.axvline(outlier_info['bounds']['upper'], color='red', linestyle='--', alpha=0.8, label='95th Percentile')
        
        ax2.set_title('Distribution with Outliers')
        ax2.set_xlabel('Values')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Scatter plot showing outliers (bottom-left)
        ax3 = axes[1, 0]
        ax3.scatter(range(len(normal_data)), normal_data, alpha=0.6, color='lightblue', s=20, label='Normal')
        if len(outlier_data) > 0:
            outlier_positions = [i for i, idx in enumerate(df.index) if idx in outlier_indices]
            ax3.scatter(outlier_positions, outlier_data, alpha=0.8, color='red', s=30, label='Outliers')
        
        ax3.set_title('Data Points (Index vs Value)')
        ax3.set_xlabel('Data Point Index')
        ax3.set_ylabel('Values')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Statistics summary (bottom-right)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create statistics text
        stats_text = f"""Statistics Summary:
        
        Total Data Points: {len(df)}
        Outliers Detected: {outlier_info['total_outliers']}
        Outlier Percentage: {outlier_info['outlier_percentage']:.2f}%

        Method: {method}
        """
        
        if method == "IQR" and outlier_info['bounds']:
            stats_text += f"""
        IQR Method Details:
        Q1: {outlier_info['bounds']['Q1']:.2f}
        Q3: {outlier_info['bounds']['Q3']:.2f}
        IQR: {outlier_info['bounds']['IQR']:.2f}
        Lower Bound: {outlier_info['bounds']['lower']:.2f}
        Upper Bound: {outlier_info['bounds']['upper']:.2f}
        """
        elif method == "Z-score" and outlier_info['bounds']:
            stats_text += f"""
        Z-Score Method Details:
        Mean: {outlier_info['bounds']['mean']:.2f}
        Std Dev: {outlier_info['bounds']['std']:.2f}
        Threshold: ±{outlier_info['bounds']['threshold']}
        """
        elif method == "percentile" and outlier_info['bounds']:
            stats_text += f"""
        Percentile Method Details:
        5th Percentile: {outlier_info['bounds']['lower']:.2f}
        95th Percentile: {outlier_info['bounds']['upper']:.2f}
        """
                
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        canvas = FigureCanvasAgg(fig)
        canvas.print_png(img_buffer)
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(fig)  # Important: close the figure to free memory
        
        return img_base64, None
        
    except Exception as e:
        return None, str(e)

def get_outlier_summary(df, column, method="IQR"):
    """
    Get a summary of outlier detection for display.
    """
    outlier_info = detect_outliers(df, column, method)
    
    if 'error' in outlier_info:
        return outlier_info
    
    # Add sample outlier values
    if outlier_info['total_outliers'] > 0:
        outlier_values = df.loc[outlier_info['outlier_indices'], column].tolist()
        # Show up to 10 sample outlier values
        sample_outliers = outlier_values[:10]
        outlier_info['sample_outliers'] = sample_outliers
        outlier_info['showing_sample'] = len(outlier_values) > 10
    
    return outlier_info

# Add these imports at the top of your existing outlier_routes.py
import json
from datetime import datetime

# Add these version management functions
def get_version_metadata_path_outlier(filename):
    """Get the metadata file path for version tracking"""
    versions_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "versions")
    os.makedirs(versions_dir, exist_ok=True)
    return os.path.join(versions_dir, f"{filename}_metadata.json")

def load_version_metadata_outlier(filename):
    """Load version metadata from JSON file"""
    metadata_path = get_version_metadata_path_outlier(filename)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"versions": [], "current_version": None}

def save_version_metadata_outlier(filename, metadata):
    """Save version metadata to JSON file"""
    metadata_path = get_version_metadata_path_outlier(filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def create_new_version_outlier(filename, df, changes="Outlier removal", user="system"):
    """Create a new version when outliers are removed"""
    try:
        metadata = load_version_metadata_outlier(filename)
        
        # Determine new version number
        if not metadata["versions"]:
            new_version = 1
        else:
            new_version = max([v["version"] for v in metadata["versions"]]) + 1
        
        # Save the new version file
        versions_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "versions")
        os.makedirs(versions_dir, exist_ok=True)
        version_file_path = os.path.join(versions_dir, f"{filename}_v{new_version}")
        df.to_csv(version_file_path, index=False)
        
        # Get file stats
        file_size = os.path.getsize(version_file_path)
        size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
        
        # Create version entry
        version_entry = {
            "version": new_version,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user,
            "changes": changes,
            "rows": len(df),
            "columns": len(df.columns),
            "size": size_str,
            "status": "current"
        }
        
        # Update previous current version status
        for version in metadata["versions"]:
            if version["status"] == "current":
                version["status"] = "previous"
        
        # Add new version
        metadata["versions"].append(version_entry)
        metadata["current_version"] = new_version
        
        # Save metadata
        save_version_metadata_outlier(filename, metadata)
        
        return new_version
    except Exception as e:
        print(f"Error creating version: {str(e)}")
        return None

# Modify the existing remove_outliers_ajax function
def remove_outliers_ajax(domain):
    """AJAX endpoint for removing detected outliers with version support"""
    if domain != "connection":
        return jsonify(success=False, message="Invalid domain.")
    
    filename = session.get("uploaded_file")
    if not filename:
        return jsonify(success=False, message="No file uploaded.")
    
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = safe_read_file(filepath)
    if df is None:
        return jsonify(success=False, message="Error reading file.")
    
    data = request.json
    column = data.get("column")
    method = data.get("method", "IQR")
    
    if not column or column not in df.columns:
        return jsonify(success=False, message="Invalid column selected.")
    
    try:
        # Store original shape
        original_shape = df.shape
        
        # Remove outliers
        df_cleaned = remove_outliers(df, column, method)
        final_shape = df_cleaned.shape
        
        # Calculate removed rows
        removed_rows = original_shape[0] - final_shape[0]
        
        # Create a new version if outliers were actually removed
        if removed_rows > 0:
            changes_description = f"Removed {removed_rows} outliers from column '{column}' using {method} method"
            version_number = create_new_version_outlier(filename, df_cleaned, changes_description, "user")
        
        # Save the cleaned data back to the file
        file_saved = save_dataframe_to_file(df_cleaned, filepath)
        
        if not file_saved:
            return jsonify(success=False, message="Error saving cleaned data.")
        
        # Convert data for JSON serialization - preview first 20 rows
        preview_data = []
        for _, row in df_cleaned.head(20).iterrows():
            row_dict = {}
            for col in df_cleaned.columns:
                value = row[col]
                row_dict[col] = convert_for_json(value)
            preview_data.append(row_dict)
        
        success_message = f"Successfully removed {removed_rows} outlier rows from column '{column}' using {method} method."
        if removed_rows > 0 and 'version_number' in locals():
            success_message += f" Created version {version_number}."
        
        return jsonify(
            success=True,
            message=success_message,
            original_rows=original_shape[0],
            final_rows=final_shape[0],
            removed_rows=removed_rows,
            columns=df_cleaned.columns.tolist(),
            preview_rows=preview_data
        )
        
    except Exception as e:
        return jsonify(success=False, message=f"Error removing outliers: {str(e)}")