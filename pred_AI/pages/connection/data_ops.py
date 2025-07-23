import traceback
from flask import session, request, jsonify, render_template, flash, redirect, url_for, current_app
import os
import pandas as pd
import numpy as np
import shutil
import json
from datetime import datetime
from pages.connection.data_processing import handle_missing_values, fix_data_types    
from pages.connection.data_processing import encode_categorical_column, group_by_columns
from pages.connection.data_processing import handle_missing_values, fix_data_types, delete_duplicates    
from pages.connection.data_processing import encode_categorical_column, group_by_columns



def safe_read_file(filepath):
    """Safely read CSV or Excel files with multiple encoding attempts"""
    try:
        if filepath.lower().endswith('.csv'):
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            separators = [',', ';', '\t']
            for encoding in encodings:
                for sep in separators:
                    try:
                        df = pd.read_csv(filepath, encoding=encoding, sep=sep)
                        if len(df.columns) > 1 or not df.empty:
                            return df
                    except:
                        continue
            # Fallback: try with engine='python' and let pandas auto-detect
            try:
                df = pd.read_csv(filepath, engine='python', encoding='utf-8', sep=None)
                return df
            except:
                # Last resort: try basic read_csv
                return pd.read_csv(filepath)
        elif filepath.lower().endswith(('.xls', '.xlsx')):
            return pd.read_excel(filepath)
    except Exception as e:
        print(f"Error reading file {filepath}: {str(e)}")
        return None
    return None

def convert_for_json(value):
    """Convert pandas/numpy types to JSON-serializable Python types."""
    if pd.isna(value):
        return None
    elif isinstance(value, (pd.Int64Dtype, pd.Int32Dtype, pd.Int16Dtype, pd.Int8Dtype)):
        return int(value) if not pd.isna(value) else None
    elif isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    elif isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    elif isinstance(value, (np.bool_, bool)):
        return bool(value)
    elif hasattr(value, 'item'):  # Handle other numpy types
        return value.item()
    else:
        return value

def save_dataframe_to_file(df, filepath):
    """Save DataFrame to file with appropriate format."""
    try:
        if filepath.lower().endswith('.csv'):
            df.to_csv(filepath, index=False, encoding='utf-8')
        elif filepath.lower().endswith(('.xls', '.xlsx')):
            df.to_excel(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error saving file {filepath}: {str(e)}")
        return False

def get_domain_folder_name(domain):
    """Map domain to folder name"""
    domain_folders = current_app.config.get("DOMAIN_FOLDERS", {})
    return domain_folders.get(domain, domain)

def get_describe_stats(df):
    """
    Returns a dict of describe() stats for each numeric column.
    """
    describe = df.describe().to_dict()
    stats = {}
    for col, metrics in describe.items():
        stats[col] = {
            "count": metrics.get("count"),
            "mean": metrics.get("mean"),
            "std": metrics.get("std"),
            "min": metrics.get("min"),
            "25%": metrics.get("25%"),
            "50%": metrics.get("50%"),
            "75%": metrics.get("75%"),
            "max": metrics.get("max"),
        }
    return stats

def data_ops_view(domain):
    """Data operations view for the single 'connection' domain"""
    filename = session.get("uploaded_file")
    if not filename:
        flash("No file uploaded.", "warning")
        return redirect(url_for("index"))
    
    filepath = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    df = safe_read_file(filepath)
    if df is None:
        flash("Error reading file. Please check file format.", "error")
        return redirect(url_for("index"))
    
    columns = df.columns.tolist()
    
    # Handle empty dataframes
    if df.empty:
        preview_rows = []
        flash("Warning: The uploaded file appears to be empty or contains only headers.", "warning")
    else:
        preview_data = []
        for _, row in df.head(20).iterrows():
            row_dict = {}
            for col in columns:
                value = row[col]
                row_dict[col] = convert_for_json(value)
            preview_data.append(row_dict)
        preview_rows = preview_data

    describe_stats = get_describe_stats(df)

    domain_display_name = "Data Connection"
    
    return render_template(
        "connection/data_ops.html", 
        filename=filename, 
        columns=columns, 
        preview_rows=preview_rows,
        domain="connection",
        domain_display=domain_display_name,
        describe_stats=describe_stats
    )

def get_version_metadata_path(filename):
    """Get the metadata file path for version tracking"""
    # Check if we're working with cleaned_data version or uploads version
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    uploads_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.root_path, "uploads"))
    
    # First check cleaned_data versions
    cleaned_versions_dir = os.path.join(cleaned_data_dir, "versions")
    cleaned_metadata_path = os.path.join(cleaned_versions_dir, f"{filename}_metadata.json")
    
    if os.path.exists(cleaned_metadata_path):
        return cleaned_metadata_path
    
    # Fallback to uploads versions
    uploads_versions_dir = os.path.join(uploads_dir, "versions")
    os.makedirs(uploads_versions_dir, exist_ok=True)
    return os.path.join(uploads_versions_dir, f"{filename}_metadata.json")

def load_version_metadata(filename):
    """Load version metadata from JSON file"""
    metadata_path = get_version_metadata_path(filename)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading version metadata for {filename}: {str(e)}")
    return {"versions": [], "current_version": None}

def save_version_metadata(filename, metadata):
    """Save version metadata to JSON file"""
    try:
        metadata_path = get_version_metadata_path(filename)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"Error saving version metadata for {filename}: {str(e)}")

def update_existing_version(filename, version, df, changes="Updated cleaning operations", user="user"):
    """Update an existing version instead of creating a new one"""
    try:
        print(f"Updating version {version} for file {filename}")
        
        # Load metadata
        metadata = load_version_metadata(filename)
        print(f"Loaded metadata: {metadata}")
        
        # Find the version to update
        version_to_update = None
        for v in metadata["versions"]:
            if v["version"] == version:
                version_to_update = v
                break
        
        if not version_to_update:
            print(f"Version {version} not found in metadata")
            return None
        
        # Determine where to save the version file
        cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
        cleaned_versions_dir = os.path.join(cleaned_data_dir, "versions")
        
        # Create versions directory if it doesn't exist
        os.makedirs(cleaned_versions_dir, exist_ok=True)
        
        # Save the updated version file
        version_file_path = os.path.join(cleaned_versions_dir, f"{filename}_v{version}")
        print(f"Saving version file to: {version_file_path}")
        
        df.to_csv(version_file_path, index=False, encoding='utf-8')
        
        # Update version metadata
        file_size = os.path.getsize(version_file_path)
        size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
        
        version_to_update.update({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user": user,
            "changes": changes,
            "rows": len(df),
            "columns": len(df.columns),
            "size": size_str
        })
        
        # Make sure this version is marked as current
        for v in metadata["versions"]:
            if v["version"] == version:
                v["status"] = "current"
            else:
                v["status"] = "previous"
        
        metadata["current_version"] = version
        
        # Save metadata
        save_version_metadata(filename, metadata)
        
        # Update the main file in cleaned_data
        main_file_path = os.path.join(cleaned_data_dir, filename)
        print(f"Updating main file: {main_file_path}")
        shutil.copy2(version_file_path, main_file_path)
        
        print(f"Successfully updated version {version}")
        return version
        
    except Exception as e:
        print(f"Error updating version: {str(e)}")
        traceback.print_exc()
        return None

def create_new_version(filename, df, changes="Data processing update", user="system"):
    """Create a new version when data is modified"""
    try:
        metadata = load_version_metadata(filename)
        
        # Determine new version number
        if not metadata["versions"]:
            new_version = 1
        else:
            new_version = max([v["version"] for v in metadata["versions"]]) + 1
        
        # Save the new version file
        uploads_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.root_path, "uploads"))
        versions_dir = os.path.join(uploads_dir, "versions")
        os.makedirs(versions_dir, exist_ok=True)
        version_file_path = os.path.join(versions_dir, f"{filename}_v{new_version}")
        
        df.to_csv(version_file_path, index=False, encoding='utf-8')
        
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
        save_version_metadata(filename, metadata)
        
        return new_version
    except Exception as e:
        print(f"Error creating version: {str(e)}")
        traceback.print_exc()
        return None

def apply_function_view(domain):
    """Generic apply function view for any domain - with version support"""
    if domain != "connection":
        return jsonify(success=False, message="Invalid domain.")
    
    filename = session.get("uploaded_file")
    if not filename:
        return jsonify(success=False, message="No file uploaded.")
    
    # Always work with a copy of the original file to preserve it
    original_filename = session.get("original_file", filename)
    uploads_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.root_path, "uploads"))
    original_filepath = os.path.join(uploads_dir, original_filename)
    current_filepath = os.path.join(uploads_dir, filename)
    
    # If working file doesn't exist, copy from original
    if not os.path.exists(current_filepath) and os.path.exists(original_filepath):
        shutil.copy2(original_filepath, current_filepath)
    
    df = safe_read_file(current_filepath)
    if df is None:
        return jsonify(success=False, message="Error reading file.")

    data = request.json
    if not data:
        return jsonify(success=False, message="No data provided.")
    
    operations = data.get("operations", [])

    if not operations:
        return jsonify(success=False, message="No operations provided.")

    try:
        original_shape = df.shape
        operation_results = []
        changes_description = []
        
        for op in operations:
            func = op.get("function")
            columns = op.get("columns") or []
            
            # Handle delete_duplicates operation specially since it doesn't need column iteration
            if func == "delete_duplicates":
                try:
                    method = op.get("method", "all_columns")
                    keep = op.get("keep", "first")
                    selected_columns = op.get("columns", None)
                    
                    # For selected_columns method, we need the columns parameter
                    if method == "selected_columns" and not selected_columns:
                        operation_results.append("Error: Selected columns method requires at least one column to be selected")
                        continue
                    
                    original_rows = len(df)
                    df, message = delete_duplicates(df, method, keep, selected_columns)
                    new_rows = len(df)
                    
                    operation_results.append(message)
                    changes_description.append(f"Deleted duplicates (method: {method}, keep: {keep})")
                    
                except Exception as e:
                    operation_results.append(f"Error processing delete duplicates: {str(e)}")
                    print(f"Error processing delete duplicates: {str(e)}")
                
                continue  # Skip the normal column iteration for this operation
            
            # For all other operations, continue with column iteration
            if not columns:
                continue
                
            for col in columns:
                if col not in df.columns:
                    operation_results.append(f"Warning: Column '{col}' not found")
                    continue
                    
                before_missing = df[col].isnull().sum() if col in df.columns else 0
                
                # Store original data type for reporting
                original_dtype = str(df[col].dtype)
                
                try:
                    if func == "handle_missing_values":
                        method = op.get("method", "drop")
                        df = handle_missing_values(df, col, method)
                        after_missing = df[col].isnull().sum() if col in df.columns else 0
                        operation_results.append(f"Column '{col}': Missing values {before_missing} → {after_missing} (method: {method})")
                        changes_description.append(f"Handled missing values in {col} using {method}")
                        
                    elif func == "fix_data_types":
                        dtype = op.get("dtype", "str")
                        df = fix_data_types(df, col, dtype)
                        new_dtype = str(df[col].dtype)
                        operation_results.append(f"Column '{col}': Data type {original_dtype} → {new_dtype}")
                        changes_description.append(f"Converted {col} from {original_dtype} to {dtype}")
                    
                    elif func == "encode_categorical":
                        method = op.get("method", "onehot")
                        df = encode_categorical_column(df, col, method)
                        operation_results.append(f"Column '{col}' encoded using {method} encoding.")
                        changes_description.append(f"Applied {method} encoding to {col}")
                    
                    elif func == "group_by":
                        # Handle group by operation - only process once per operation, not per column
                        if col == columns[0]:  # Only process on first column to avoid duplication
                            group_by_col = op.get("group_by_column")
                            aggregate_col = op.get("aggregate_column") 
                            aggregation_method = op.get("aggregation_method", "list")
                            
                            if not group_by_col or not aggregate_col:
                                operation_results.append(f"Error: Group by requires both group_by_column and aggregate_column")
                                continue
                                
                            if group_by_col not in df.columns:
                                operation_results.append(f"Error: Group by column '{group_by_col}' not found")
                                continue
                                
                            if aggregate_col not in df.columns:
                                operation_results.append(f"Error: Aggregate column '{aggregate_col}' not found")
                                continue
                            
                            original_rows = len(df)
                            df = group_by_columns(df, group_by_col, aggregate_col, aggregation_method)
                            new_rows = len(df)
                            
                            operation_results.append(f"Grouped by '{group_by_col}', aggregated '{aggregate_col}' using {aggregation_method}: {original_rows} → {new_rows} rows")
                            changes_description.append(f"Grouped by {group_by_col}, aggregated {aggregate_col} using {aggregation_method}")
                    
                    elif func == "remove_outliers":
                        operation_results.append(f"Column '{col}': Outlier removal is now handled in the dedicated Outlier Analysis section")
                        
                except Exception as e:
                    operation_results.append(f"Error processing column '{col}': {str(e)}")
                    print(f"Error processing column {col} with function {func}: {str(e)}")
        final_shape = df.shape

        # Create a new version if data was actually modified
        if original_shape != final_shape or len(changes_description) > 0:
            changes_summary = "; ".join(changes_description) if changes_description else "Data processing operations applied"
            version_number = create_new_version(filename, df, changes_summary, "user")
            
            if version_number:
                operation_results.append(f"Created version {version_number} with changes")

        # Save the processed data to the current working file
        file_saved = save_dataframe_to_file(df, current_filepath)
        
        # Convert data for JSON serialization - handle different data types properly
        preview_data = []
        for _, row in df.head(20).iterrows():
            row_dict = {}
            for col in df.columns:
                value = row[col]
                row_dict[col] = convert_for_json(value)
            preview_data.append(row_dict)

        shape_change = f"Data shape: {original_shape[0]} → {final_shape[0]} rows, {original_shape[1]} → {final_shape[1]} columns"
        success_message = f"Operations applied successfully! {shape_change}"
        if operation_results:
            success_message += "\n\nDetails:\n" + "\n".join(operation_results[:5])

        return jsonify(
            success=True, 
            message=success_message,
            columns=df.columns.tolist(), 
            preview_rows=preview_data,
            original_shape=original_shape,
            final_shape=final_shape,
            operations_applied=len(operations),
            operation_details=operation_results,
            file_saved=file_saved,
            saved_filename=filename
        )
    except Exception as e:
        error_message = f"Error applying operations: {str(e)}"
        print(f"Error in apply_function_view: {error_message}")
        traceback.print_exc()
        return jsonify(success=False, message=error_message)

def save_processed_file_view():
    """Handle saving processed data - updated to support version editing"""
    current_app.logger.info("Entered save_processed_file_view")
    current_app.logger.info(f"Request data: {request.get_json()}")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify(success=False, message="No data provided")
            
        filename = data.get("filename", "").strip()
        edit_version = data.get("edit_version") or session.get("edit_version")
        original_filename = data.get("original_filename") or session.get("original_catalog_filename")
        
        # Get current uploaded file (the processed version)
        current_filename = session.get("uploaded_file")
        if not current_filename:
            return jsonify(success=False, message="No file in session")
        
        uploads_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.root_path, "uploads"))
        current_filepath = os.path.join(uploads_dir, current_filename)
        if not os.path.exists(current_filepath):
            return jsonify(success=False, message="Current file not found")
        
        # Read the data
        df = safe_read_file(current_filepath)
        if df is None:
            return jsonify(success=False, message="Error reading processed file")
        
        # Check if we're in edit mode (updating an existing version)
        if edit_version and original_filename:
            print(f"Edit mode: updating version {edit_version} of {original_filename}")
            
            # Update the existing version instead of creating new
            updated_version = update_existing_version(
                original_filename, 
                int(edit_version), 
                df, 
                "Updated through cleaning operations", 
                "user"
            )
            
            if updated_version:
                # Clear edit mode session variables
                session.pop("edit_mode", None)
                session.pop("edit_version", None)
                session.pop("original_catalog_filename", None)
                
                return jsonify(
                    success=True,
                    saved_filename=original_filename,
                    saved_path=f"cleaned_data/{original_filename}",
                    version_updated=updated_version,
                    message=f"Version {updated_version} updated successfully"
                )
            else:
                return jsonify(success=False, message="Failed to update version")
        
        else:
            # Normal save process
            if not filename:
                return jsonify(success=False, message="No filename provided")
                
            # Ensure filename has proper extension
            if not filename.lower().endswith(('.csv', '.xls', '.xlsx')):
                _, ext = os.path.splitext(current_filename)
                filename += ext
            
            # Create a final version before saving to cleaned_data
            version_number = create_new_version(current_filename, df, "Final version before saving to cleaned_data", "user")
            
            # Save directly to cleaned_data folder
            cleaned_data_folder = current_app.config.get("CLEANED_DATA_FOLDER", os.path.join(current_app.root_path, "cleaned_data"))
            os.makedirs(cleaned_data_folder, exist_ok=True)
            cleaned_filepath = os.path.join(cleaned_data_folder, filename)
            
            # Save the file
            df.to_csv(cleaned_filepath, index=False, encoding='utf-8')
            
            # Initialize version tracking for the cleaned file
            file_size = os.path.getsize(cleaned_filepath)
            size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
            
            cleaned_metadata = {
                "versions": [{
                    "version": 1,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": "user",
                    "changes": "Initial cleaned version",
                    "rows": len(df),
                    "columns": len(df.columns),
                    "size": size_str,
                    "status": "current"
                }],
                "current_version": 1
            }
            
            # Save metadata for the cleaned file
            cleaned_versions_dir = os.path.join(cleaned_data_folder, "versions")
            os.makedirs(cleaned_versions_dir, exist_ok=True)
            cleaned_metadata_path = os.path.join(cleaned_versions_dir, f"{filename}_metadata.json")
            
            with open(cleaned_metadata_path, 'w') as f:
                json.dump(cleaned_metadata, f, indent=2)
            
            # Save the initial version file
            cleaned_version_file = os.path.join(cleaned_versions_dir, f"{filename}_v1")
            shutil.copy2(cleaned_filepath, cleaned_version_file)
            
            return jsonify(
                success=True,
                saved_filename=filename,
                saved_path=f"cleaned_data/{filename}",
                message=f"File saved successfully to cleaned_data/{filename} with version tracking initialized"
            )
    
    except Exception as e:
        traceback.print_exc()
        return jsonify(success=False, message=f"Error saving file: {str(e)}")

def list_previous_work_view(domain):
    """List files from cleaned_data folders for specific domain"""
    try:
        previous_work_files = []
        
        # Map domain to folder name
        folder_name = get_domain_folder_name(domain)
        cleaned_data_folder = current_app.config.get("CLEANED_DATA_FOLDER", os.path.join(current_app.root_path, "cleaned_data"))
        domain_cleaned_folder = os.path.join(cleaned_data_folder, folder_name)
        
        if os.path.exists(domain_cleaned_folder):
            for filename in os.listdir(domain_cleaned_folder):
                if filename.lower().endswith(('.csv', '.xls', '.xlsx')):
                    filepath = os.path.join(domain_cleaned_folder, filename)
                    created_time = os.path.getctime(filepath)
                    created_date = datetime.fromtimestamp(created_time).strftime("%Y-%m-%d %H:%M")
                    previous_work_files.append({
                        "id": f"cleaned_{filename}",
                        "name": f"🧹 {filename}",
                        "date": created_date,
                        "type": "cleaned",
                        "filename": filename,
                        "path": f"/cleaned_data/{domain}/{filename}"
                    })
        
        # Sort by date (newest first)
        previous_work_files.sort(key=lambda x: x["date"], reverse=True)
        
        return jsonify(files=previous_work_files)
    except Exception as e:
        print(f"Error in list_previous_work_view: {str(e)}")
        return jsonify(files=[])

def load_previous_work_view(domain):
    """Load previous work files for specific domain"""
    try:
        data = request.get_json()
        if not data:
            return jsonify(success=False, message="No data provided")
            
        work_id = data.get("work_id")
        filename = data.get("filename")
        work_type = data.get("type", "cleaned")
        
        if work_type == "cleaned":
            # Map domain to folder name
            folder_name = get_domain_folder_name(domain)
            cleaned_data_folder = current_app.config.get("CLEANED_DATA_FOLDER", os.path.join(current_app.root_path, "cleaned_data"))
            source_path = os.path.join(cleaned_data_folder, folder_name, filename)
            
            if os.path.exists(source_path):
                # Copy to uploads folder for processing
                uploads_dir = current_app.config.get("UPLOAD_FOLDER", os.path.join(current_app.root_path, "uploads"))
                target_path = os.path.join(uploads_dir, filename)
                shutil.copy2(source_path, target_path)
                session["uploaded_file"] = filename
                session["current_domain"] = domain
                return jsonify(success=True, message=f"Loaded cleaned file: {filename}")
        
        return jsonify(success=False, message="File not found")
    except Exception as e:
        print(f"Error in load_previous_work_view: {str(e)}")
        return jsonify(success=False, message=f"Error loading file: {str(e)}")

# Helper function to validate file operations
def validate_file_operation(filename, operation="read"):
    """Validate file operations for security"""
    if not filename:
        return False, "No filename provided"
    # Check for path traversal attacks
    if ".." in filename or "/" in filename or "\\" in filename:
        return False, "Invalid filename"
    # Check file extension
    allowed_extensions = ['.csv', '.xls', '.xlsx']
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        return False, "Invalid file type"
    return True, "Valid"

# Enhanced error handling wrapper
def handle_errors(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"Error in {func.__name__}: {str(e)}")
            traceback.print_exc()
            return jsonify(success=False, message=f"Internal error: {str(e)}")
    return wrapper

# Apply error handling to main functions
apply_function_view = handle_errors(apply_function_view)
save_processed_file_view = handle_errors(save_processed_file_view)
list_previous_work_view = handle_errors(list_previous_work_view)
load_previous_work_view = handle_errors(load_previous_work_view)