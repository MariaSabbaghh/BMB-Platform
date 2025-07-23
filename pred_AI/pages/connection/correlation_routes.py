from flask import request, jsonify, render_template, session, current_app
import os
import pandas as pd
import numpy as np
from pages.connection.data_ops import safe_read_file, convert_for_json

def correlation_analysis_view(domain):
    """View for correlation analysis page"""
    if domain != "connection":
        return jsonify(success=False, message="Invalid domain.")
        
    filename = session.get("uploaded_file")
    if not filename:
        return jsonify(success=False, message="No file uploaded.")
    
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = safe_read_file(filepath)
    if df is None:
        return jsonify(success=False, message="Error reading file.")
    
    # Get only numeric columns for correlation analysis
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Domain display mapping
    domain_display_names = {
        'connection': 'Data Connection'
    }
    
    return render_template("connection/correlation_analysis.html",
                         filename=filename,
                         columns=numeric_columns,
                         domain=domain,
                         domain_display=domain_display_names.get(domain, domain.title()))

def compute_correlations_ajax(domain):
    """AJAX endpoint for computing correlations"""
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
    columns = data.get("columns", [])
    method = data.get("method", "pearson")
    
    if not columns or len(columns) < 2:
        return jsonify(success=False, message="Please select at least 2 columns.")
    
    try:
        # Compute correlations
        corr_df = df[columns].corr(method=method.lower())
        
        # Convert to dictionary for JSON response
        correlations = {}
        for col in corr_df.columns:
            correlations[col] = {k: round(v, 4) for k, v in corr_df[col].items()}
        
        return jsonify(
            success=True,
            correlations=correlations,
            method=method
        )
        
    except Exception as e:
        return jsonify(success=False, message=f"Error computing correlations: {str(e)}")