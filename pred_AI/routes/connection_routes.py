# ===== FIXED connection routes - Complete Routes =====
from flask import Blueprint, render_template, request, session, jsonify, send_from_directory, flash, redirect, url_for, current_app
import os
import pandas as pd
from datetime import datetime

from pages.connection.data_ops import (
    data_ops_view, apply_function_view, save_processed_file_view,
    list_previous_work_view, load_previous_work_view
)
from pages.connection.outlier_routes import (
    outlier_analysis_view, detect_outliers_ajax, remove_outliers_ajax, outlier_summary_ajax
)
from pages.connection.correlation_routes import correlation_analysis_view, compute_correlations_ajax

connection_bp = Blueprint('connection', __name__)

# Serve uploaded files
@connection_bp.route('/uploads/<filename>')
def uploaded_file_static(filename):
    uploads_folder = os.path.join(os.getcwd(), "uploads")
    return send_from_directory(uploads_folder, filename)

# Serve cleaned data files
@connection_bp.route('/cleaned_data/<filename>')
def cleaned_file_static(filename):
    cleaned_data_folder = os.path.join(os.getcwd(), "cleaned_data")
    return send_from_directory(cleaned_data_folder, filename)

# Upload CSV or Excel file with enhanced error handling
@connection_bp.route("/upload_csv/<domain>", methods=["POST"])
def upload_csv(domain):
    try:
        file = request.files.get("csv_file") or request.files.get("file")
        if not file:
            return jsonify(success=False, message="No file part"), 400
        if file.filename == '':
            return jsonify(success=False, message="No selected file"), 400

        allowed_exts = (".csv", ".xls", ".xlsx")
        if not file.filename.lower().endswith(allowed_exts):
            return jsonify(success=False, message="Invalid file. Please upload a CSV or Excel file."), 400

        max_size = 200 * 1024 * 1024
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)
        if size > max_size:
            return jsonify(success=False, message="File is too large. Maximum size is 200MB."), 400

        filename = file.filename
        uploads_folder = os.path.join(os.getcwd(), "uploads")
        os.makedirs(uploads_folder, exist_ok=True)  # Ensure directory exists
        
        filepath = os.path.join(uploads_folder, filename)
        
        # Save file
        file.save(filepath)
        
        # Verify file was saved correctly
        if not os.path.exists(filepath):
            return jsonify(success=False, message="Failed to save file"), 500
            
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            return jsonify(success=False, message="Uploaded file is empty"), 400
        
        # Set session variables
        session["uploaded_file"] = filename
        session["original_file"] = filename
        session["current_domain"] = domain
        
        print(f"✅ File uploaded successfully: {filename} ({file_size} bytes)")
        
        # Try to read the file to validate it
        try:
            if filename.lower().endswith(".csv"):
                # Try to read first few rows to get columns
                df = pd.read_csv(filepath, nrows=5)
                columns = df.columns.tolist()
                print(f"✅ CSV validation successful: {len(columns)} columns found")
                return jsonify(
                    success=True, 
                    filename=filename, 
                    columns=columns, 
                    message=f"CSV uploaded successfully! Found {len(columns)} columns."
                )
            else:
                # For Excel files, just confirm upload
                return jsonify(
                    success=True, 
                    filename=filename, 
                    message="Excel file uploaded successfully."
                )
        except Exception as read_error:
            print(f"⚠️ File validation warning: {read_error}")
            # File uploaded but might have reading issues - let data_ops handle it
            return jsonify(
                success=True, 
                filename=filename, 
                message=f"File uploaded as {filename}. Note: There may be encoding or format issues."
            )
            
    except Exception as e:
        print(f"💥 Upload error: {str(e)}")
        return jsonify(success=False, message=f"Upload error: {str(e)}"), 500

# Set uploaded file in session
@connection_bp.route("/set_uploaded_file/<domain>", methods=["POST"])
def set_uploaded_file(domain):
    try:
        data = request.get_json()
        filename = data.get("filename")
        uploads_folder = os.path.join(os.getcwd(), "uploads")
        
        if filename and os.path.exists(os.path.join(uploads_folder, filename)):
            session["uploaded_file"] = filename
            session["current_domain"] = domain
            print(f"✅ Set uploaded file: {filename}")
            return jsonify(success=True)
        else:
            print(f"❌ File not found: {filename}")
            return jsonify(success=False, message="File not found"), 400
    except Exception as e:
        print(f"💥 Error setting uploaded file: {str(e)}")
        return jsonify(success=False, message=str(e)), 500

# Connect to database
@connection_bp.route("/connect_db/<domain>", methods=["POST"])
def connect_db(domain):
    db_uri = request.form.get("db_uri")
    flash(f"Attempted to connect to {db_uri}", "info")
    return redirect(request.referrer or url_for("index"))

# Data operations with enhanced error handling
@connection_bp.route("/data_ops", methods=["GET"])
def data_ops():
    try:
        print("🔍 Data ops route called")
        return data_ops_view("connection")
    except Exception as e:
        print(f"💥 Error in data_ops route: {str(e)}")
        flash(f"Error loading data operations: {str(e)}", "error")
        return redirect(url_for("index"))

@connection_bp.route("/apply_function", methods=["POST"])
def apply_function():
    try:
        print("🔧 Apply function route called")
        return apply_function_view("connection")
    except Exception as e:
        print(f"💥 Error in apply_function route: {str(e)}")
        return jsonify(success=False, message=f"Error applying function: {str(e)}")

@connection_bp.route("/save_processed_file", methods=["POST"])
def save_processed_file():
    try:
        print("💾 Save processed file route called")
        return save_processed_file_view()
    except Exception as e:
        print(f"💥 Error in save_processed_file route: {str(e)}")
        return jsonify(success=False, message=f"Error saving file: {str(e)}")

# Outlier analysis
@connection_bp.route("/outlier_analysis", methods=["GET"])
def outlier_analysis():
    try:
        print("📊 Outlier analysis route called")
        return outlier_analysis_view("connection")
    except Exception as e:
        print(f"💥 Error in outlier_analysis route: {str(e)}")
        flash(f"Error loading outlier analysis: {str(e)}", "error")
        return redirect(url_for("connection.data_ops"))

@connection_bp.route("/detect_outliers", methods=["POST"])
def detect_outliers():
    try:
        return detect_outliers_ajax("connection")
    except Exception as e:
        print(f"💥 Error in detect_outliers route: {str(e)}")
        return jsonify(success=False, message=f"Error detecting outliers: {str(e)}")

@connection_bp.route("/remove_outliers", methods=["POST"])
def remove_outliers_endpoint():
    try:
        return remove_outliers_ajax("connection")
    except Exception as e:
        print(f"💥 Error in remove_outliers route: {str(e)}")
        return jsonify(success=False, message=f"Error removing outliers: {str(e)}")

@connection_bp.route("/outlier_summary", methods=["GET"])
def outlier_summary():
    try:
        return outlier_summary_ajax("connection")
    except Exception as e:
        print(f"💥 Error in outlier_summary route: {str(e)}")
        return jsonify(success=False, message=f"Error getting outlier summary: {str(e)}")

# Correlation analysis
@connection_bp.route("/correlation_analysis", methods=["GET"])
def correlation_analysis():
    try:
        print("📈 Correlation analysis route called")
        return correlation_analysis_view("connection")
    except Exception as e:
        print(f"💥 Error in correlation_analysis route: {str(e)}")
        flash(f"Error loading correlation analysis: {str(e)}", "error")
        return redirect(url_for("connection.data_ops"))

@connection_bp.route("/compute_correlations", methods=["POST"])
def compute_correlations():
    try:
        return compute_correlations_ajax("connection")
    except Exception as e:
        print(f"💥 Error in compute_correlations route: {str(e)}")
        return jsonify(success=False, message=f"Error computing correlations: {str(e)}")

# Previous work management
@connection_bp.route("/list_previous_work", methods=["GET"])
def list_previous_work():
    try:
        return list_previous_work_view("connection")
    except Exception as e:
        print(f"💥 Error in list_previous_work route: {str(e)}")
        return jsonify(files=[])

@connection_bp.route("/load_previous_work", methods=["POST"])
def load_previous_work():
    try:
        return load_previous_work_view("connection")
    except Exception as e:
        print(f"💥 Error in load_previous_work route: {str(e)}")
        return jsonify(success=False, message=f"Error loading previous work: {str(e)}")

# Health check endpoint
@connection_bp.route("/health", methods=["GET"])
def health_check():
    """Simple health check endpoint"""
    try:
        uploads_folder = os.path.join(os.getcwd(), "uploads")
        cleaned_data_folder = os.path.join(os.getcwd(), "cleaned_data")
        
        # Check if directories exist and are writable
        uploads_exists = os.path.exists(uploads_folder) and os.access(uploads_folder, os.W_OK)
        cleaned_exists = os.path.exists(cleaned_data_folder) and os.access(cleaned_data_folder, os.W_OK)
        
        session_file = session.get("uploaded_file")
        
        return jsonify({
            "status": "healthy",
            "uploads_folder": uploads_exists,
            "cleaned_data_folder": cleaned_exists,
            "session_file": session_file,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

# Debug endpoint for development
@connection_bp.route("/debug_session", methods=["GET"])
def debug_session():
    """Debug endpoint to check session state"""
    if current_app.debug:  # Only in debug mode
        return jsonify({
            "session_data": dict(session),
            "uploads_folder": os.path.join(os.getcwd(), "uploads"),
            "files_in_uploads": os.listdir(os.path.join(os.getcwd(), "uploads")) if os.path.exists(os.path.join(os.getcwd(), "uploads")) else []
        })
    else:
        return jsonify({"error": "Debug endpoint not available in production"}), 404