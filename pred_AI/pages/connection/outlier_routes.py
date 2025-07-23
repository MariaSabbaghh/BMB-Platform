from flask import request, jsonify, render_template, session, current_app
import os
import pandas as pd
from pages.connection.data_ops import safe_read_file, convert_for_json, save_dataframe_to_file
from pages.connection.outlier_detection import plot_outliers_detection, get_outlier_summary, detect_outliers
from pages.connection.data_processing import remove_outliers

def outlier_analysis_view(domain):
    """View for outlier analysis page"""
    if domain != "connection":
        return jsonify(success=False, message="Invalid domain.")
    
    filename = session.get("uploaded_file")
    if not filename:
        return jsonify(success=False, message="No file uploaded.")
    
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = safe_read_file(filepath)
    if df is None:
        return jsonify(success=False, message="Error reading file.")
    
    # Get only numeric columns for outlier analysis
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Domain display mapping
    domain_display_names = {
        'connection': 'Data Connection'
    }
    
    return render_template("connection/outlier_analysis.html",
                         filename=filename,
                         columns=numeric_columns,
                         domain=domain,
                         domain_display=domain_display_names.get(domain, domain.title()))

def detect_outliers_ajax(domain):
    """AJAX endpoint for outlier detection and visualization"""
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
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return jsonify(success=False, message="Selected column is not numeric.")
    
    try:
        # Generate outlier detection plot
        plot_base64, plot_error = plot_outliers_detection(df, column, method)
        
        if plot_error:
            return jsonify(success=False, message=f"Error creating plot: {plot_error}")
        
        # Get outlier summary
        outlier_summary = get_outlier_summary(df, column, method)
        
        if 'error' in outlier_summary:
            return jsonify(success=False, message=outlier_summary['error'])
        
        return jsonify(
            success=True,
            plot_image=plot_base64,
            outlier_summary=outlier_summary,
            column=column,
            method=method
        )
        
    except Exception as e:
        return jsonify(success=False, message=f"Error analyzing outliers: {str(e)}")

def remove_outliers_ajax(domain):
    """AJAX endpoint for removing detected outliers"""
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
        
        return jsonify(
            success=True,
            message=f"Successfully removed {removed_rows} outlier rows from column '{column}' using {method} method.",
            original_rows=original_shape[0],
            final_rows=final_shape[0],
            removed_rows=removed_rows,
            columns=df_cleaned.columns.tolist(),
            preview_rows=preview_data
        )
        
    except Exception as e:
        return jsonify(success=False, message=f"Error removing outliers: {str(e)}")

def outlier_summary_ajax(domain):
    """AJAX endpoint for getting outlier summary for multiple columns"""
    if domain != "connection":
        return jsonify(success=False, message="Invalid domain.")
    
    filename = session.get("uploaded_file")
    if not filename:
        return jsonify(success=False, message="No file uploaded.")
    
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = safe_read_file(filepath)
    if df is None:
        return jsonify(success=False, message="Error reading file.")
    
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Get outlier summary for each numeric column using IQR method
    column_summaries = {}
    for col in numeric_columns:
        try:
            summary = get_outlier_summary(df, col, "IQR")
            if 'error' not in summary:
                column_summaries[col] = {
                    'total_outliers': summary['total_outliers'],
                    'outlier_percentage': round(summary['outlier_percentage'], 2)
                }
        except:
            continue
    
    return jsonify(
        success=True,
        column_summaries=column_summaries,
        total_rows=len(df)
    )