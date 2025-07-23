import os
import pandas as pd
import numpy as np
import shutil
import json
from datetime import datetime
from flask import Blueprint, render_template, request, jsonify, send_from_directory, redirect, url_for, current_app

catalog_bp = Blueprint('catalog', __name__, url_prefix="/catalog")

def get_version_file_path(filename, version=None):
    """Get the file path for a specific version"""
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    if version:
        return os.path.join(cleaned_data_dir, "versions", f"{filename}_v{version}")
    return os.path.join(cleaned_data_dir, filename)

def get_version_metadata_path(filename):
    """Get the metadata file path for version tracking"""
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    versions_dir = os.path.join(cleaned_data_dir, "versions")
    os.makedirs(versions_dir, exist_ok=True)
    return os.path.join(versions_dir, f"{filename}_metadata.json")

def load_version_metadata(filename):
    """Load version metadata from JSON file"""
    metadata_path = get_version_metadata_path(filename)
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except:
            pass
    return {"versions": [], "current_version": None}

def save_version_metadata(filename, metadata):
    """Save version metadata to JSON file"""
    metadata_path = get_version_metadata_path(filename)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

def create_new_version(filename, df, changes="Manual update", user="system"):
    """Create a new version of a file"""
    metadata = load_version_metadata(filename)
    if not metadata["versions"]:
        new_version = 1
    else:
        new_version = max([v["version"] for v in metadata["versions"]]) + 1
    version_file_path = get_version_file_path(filename, new_version)
    df.to_csv(version_file_path, index=False)
    file_size = os.path.getsize(version_file_path)
    size_str = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
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
    for version in metadata["versions"]:
        if version["status"] == "current":
            version["status"] = "previous"
    metadata["versions"].append(version_entry)
    metadata["current_version"] = new_version
    save_version_metadata(filename, metadata)
    main_file_path = get_version_file_path(filename)
    shutil.copy2(version_file_path, main_file_path)
    return new_version

def update_existing_version(filename, version, df, changes="Edited version", user="system"):
    version_file_path = get_version_file_path(filename, version)
    if not os.path.exists(version_file_path):
        return False, "Version file does not exist"
    try:
        df.to_csv(version_file_path, index=False)
    except Exception as e:
        return False, f"Failed to save CSV: {e}"
    metadata = load_version_metadata(filename)
    updated = False
    for v in metadata["versions"]:
        if v["version"] == version:
            v["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            v["user"] = user
            v["changes"] = changes
            v["rows"] = len(df)
            v["columns"] = len(df.columns)
            file_size = os.path.getsize(version_file_path)
            v["size"] = f"{file_size / 1024:.2f} KB" if file_size < 1024*1024 else f"{file_size / (1024*1024):.2f} MB"
            v["status"] = "current"
            updated = True
        else:
            if v["status"] == "current":
                v["status"] = "previous"
    if not updated:
        return False, "Version metadata entry not found"
    metadata["current_version"] = version
    try:
        save_version_metadata(filename, metadata)
    except Exception as e:
        return False, f"Failed to save metadata: {e}"
    main_file_path = get_version_file_path(filename)
    try:
        shutil.copy2(version_file_path, main_file_path)
    except Exception as e:
        return False, f"Failed to update main file: {e}"
    return True, version

def detect_feature_type(column_data):
    if pd.api.types.is_numeric_dtype(column_data):
        unique_ratio = column_data.nunique() / len(column_data)
        if unique_ratio > 0.95 and column_data.nunique() > 50:
            return "Categorical", "fas fa-hashtag"
        elif column_data.dtype in ['int64', 'int32', 'int16', 'int8'] and column_data.nunique() < 20:
            return "Categorical", "fas fa-tags"
        else:
            return "Numeric", "fas fa-hashtag"
    elif pd.api.types.is_datetime64_any_dtype(column_data):
        return "Date/Time", "fas fa-calendar"
    else:
        if column_data.dtype == 'object':
            sample = column_data.dropna().head(100)
            if len(sample) > 0:
                try:
                    pd.to_datetime(sample.iloc[0])
                    return "Date/Time", "fas fa-calendar"
                except:
                    pass
        return "Categorical", "fas fa-tags"

def calculate_feature_quality_score(column_data, feature_type):
    score = 100
    issues = []
    total_count = len(column_data)
    null_count = column_data.isnull().sum()
    null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
    unique_count = column_data.nunique()
    if null_percentage > 80:
        score -= 50
        issues.append("Very high missing data")
    elif null_percentage > 50:
        score -= 30
        issues.append("High missing data")
    elif null_percentage > 20:
        score -= 15
        issues.append("Moderate missing data")
    elif null_percentage > 5:
        score -= 5
        issues.append("Some missing data")
    if feature_type == "Categorical":
        if unique_count == 1:
            score -= 40
            issues.append("Single value only")
        elif unique_count == total_count:
            score -= 35
            issues.append("All values unique")
        elif unique_count > total_count * 0.95:
            score -= 20
            issues.append("Very high cardinality")
    elif feature_type == "Numeric":
        if unique_count == 1:
            score -= 45
            issues.append("Constant value")
        try:
            Q1 = column_data.quantile(0.25)
            Q3 = column_data.quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = column_data[(column_data < lower_bound) | (column_data > upper_bound)]
                outlier_percentage = len(outliers) / total_count * 100
                if outlier_percentage > 10:
                    score -= 15
                    issues.append("Many outliers detected")
                elif outlier_percentage > 5:
                    score -= 8
                    issues.append("Some outliers detected")
        except:
            pass
    if not issues:
        issues.append("Good quality")
    elif score > 85:
        issues.insert(0, "High quality")
    elif score > 70:
        issues.insert(0, "Good quality")
    elif score > 50:
        issues.insert(0, "Fair quality")
    else:
        issues.insert(0, "Poor quality")
    return max(0, min(100, score)), issues

def get_feature_statistics(column_data, feature_type):
    stats = {}
    if feature_type == "Numeric":
        try:
            stats = {
                "min": float(column_data.min()) if pd.notnull(column_data.min()) else None,
                "max": float(column_data.max()) if pd.notnull(column_data.max()) else None,
                "mean": float(column_data.mean()) if pd.notnull(column_data.mean()) else None,
                "median": float(column_data.median()) if pd.notnull(column_data.median()) else None,
                "std": float(column_data.std()) if pd.notnull(column_data.std()) else None,
                "q25": float(column_data.quantile(0.25)) if pd.notnull(column_data.quantile(0.25)) else None,
                "q75": float(column_data.quantile(0.75)) if pd.notnull(column_data.quantile(0.75)) else None
            }
        except:
            stats = {}
    elif feature_type == "Categorical":
        try:
            value_counts = column_data.value_counts()
            stats = {
                "unique_values": int(column_data.nunique()),
                "most_frequent": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                "most_frequent_count": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                "top_values": [
                    {"value": str(val), "count": int(count), "percentage": round(count/len(column_data)*100, 2)}
                    for val, count in value_counts.head(5).items()
                ]
            }
        except:
            stats = {"unique_values": int(column_data.nunique())}
    elif feature_type == "Date/Time":
        try:
            if not pd.api.types.is_datetime64_any_dtype(column_data):
                date_col = pd.to_datetime(column_data, errors='coerce')
            else:
                date_col = column_data
            valid_dates = date_col.dropna()
            if len(valid_dates) > 0:
                stats = {
                    "earliest": str(valid_dates.min()),
                    "latest": str(valid_dates.max()),
                    "range_days": (valid_dates.max() - valid_dates.min()).days if len(valid_dates) > 1 else 0
                }
            else:
                stats = {}
        except:
            stats = {}
    return stats

@catalog_bp.route("/")
def catalog_index():
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    files = [fname for fname in os.listdir(cleaned_data_dir) if fname.endswith(".csv")]
    return render_template("catalog/catalog.html", files=files)

@catalog_bp.route("/view/<filename>")
def catalog_view(filename):
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    path = os.path.join(cleaned_data_dir, filename)
    if not os.path.exists(path):
        return jsonify({"success": False, "message": "File not found", "preview": "", "total": 0})
    try:
        start = int(request.args.get("start", 0))
        length = int(request.args.get("length", 10))
    except Exception:
        start = 0
        length = 10
    try:
        df = pd.read_csv(path)
        total = len(df)
        preview = df.iloc[start:start+length].to_html(classes="table table-striped", index=False)
        return jsonify({"success": True, "preview": preview, "total": total})
    except Exception as e:
        return jsonify({"success": False, "message": str(e), "preview": "", "total": 0})

@catalog_bp.route("/delete/<filename>", methods=["POST"])
def catalog_delete(filename):
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    path = os.path.join(cleaned_data_dir, filename)
    if os.path.exists(path):
        os.remove(path)
        versions_dir = os.path.join(cleaned_data_dir, "versions")
        metadata_path = get_version_metadata_path(filename)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        if os.path.exists(versions_dir):
            for file in os.listdir(versions_dir):
                if file.startswith(f"{filename}_v"):
                    os.remove(os.path.join(versions_dir, file))
        return jsonify({"success": True})
    return jsonify({"success": False, "message": "File not found"})

@catalog_bp.route("/download/<filename>")
def catalog_download(filename):
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    return send_from_directory(cleaned_data_dir, filename, as_attachment=True)

@catalog_bp.route("/train/<filename>")
def catalog_train(filename):
    return redirect(url_for("train", file=filename))

@catalog_bp.route("/info/<filename>")
def catalog_info(filename):
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    path = os.path.join(cleaned_data_dir, filename)
    if not os.path.exists(path):
        return jsonify({"success": False, "message": "File not found"})
    try:
        df = pd.read_csv(path)
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()
        datetime_cols = df.select_dtypes(include='datetime').columns.tolist()
        for col in df.columns:
            if col not in datetime_cols:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        datetime_cols.append(col)
                except Exception:
                    pass
        missing_data = []
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count > 0:
                missing_percentage = round(missing_count / len(df) * 100, 2) if len(df) else 0
                missing_data.append({
                    "column": col,
                    "missing_count": missing_count,
                    "missing_percentage": missing_percentage
                })
        total_cells = df.shape[0] * df.shape[1]
        total_missing = int(df.isnull().sum().sum())
        missing_percentage = round((total_missing / total_cells) * 100, 2) if total_cells else 0
        info = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "file_size": f"{os.path.getsize(path) / 1024:.2f} KB",
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB",
            "numeric_features": numeric_cols,
            "categorical_features": categorical_cols,
            "datetime_features": datetime_cols,
            "numeric_columns": len(numeric_cols),
            "categorical_columns": len(categorical_cols),
            "datetime_columns": len(datetime_cols),
            "total_missing": total_missing,
            "missing_percentage": missing_percentage,
            "duplicate_rows": int(df.duplicated().sum()),
            "missing_data": missing_data
        }
        return jsonify({"success": True, "info": info})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@catalog_bp.route("/features/<filename>")
def catalog_features(filename):
    cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
    path = os.path.join(cleaned_data_dir, filename)
    if not os.path.exists(path):
        return jsonify({"success": False, "message": "File not found"})
    try:
        df = pd.read_csv(path)
        features = []
        for col in df.columns:
            try:
                column_data = df[col]
                feature_type, type_icon = detect_feature_type(column_data)
                total_count = len(df)
                non_null_count = column_data.count()
                null_count = column_data.isnull().sum()
                null_percentage = (null_count / total_count) * 100 if total_count > 0 else 0
                unique_count = column_data.nunique()
                quality_score, quality_issues = calculate_feature_quality_score(column_data, feature_type)
                stats = get_feature_statistics(column_data, feature_type)
                importance = min(100, quality_score * (1 - null_percentage/100))
                if feature_type == "Categorical" and unique_count > 1 and unique_count < total_count * 0.8:
                    importance += 10
                elif feature_type == "Numeric" and unique_count > 10:
                    importance += 15
                feature_info = {
                    "name": col,
                    "type": feature_type,
                    "type_icon": type_icon,
                    "total_count": int(total_count),
                    "non_null_count": int(non_null_count),
                    "null_count": int(null_count),
                    "null_percentage": round(null_percentage, 2),
                    "unique_count": int(unique_count),
                    "quality_score": int(quality_score),
                    "quality_issues": quality_issues,
                    "importance": round(min(100, importance), 1),
                    "stats": stats
                }
                features.append(feature_info)
            except Exception as e:
                features.append({
                    "name": col,
                    "type": "Unknown",
                    "type_icon": "fas fa-question",
                    "total_count": len(df),
                    "non_null_count": 0,
                    "null_count": len(df),
                    "null_percentage": 100.0,
                    "unique_count": 0,
                    "quality_score": 0,
                    "quality_issues": ["Error processing feature"],
                    "importance": 0,
                    "stats": {}
                })
                print(f"Error processing column {col}: {str(e)}")
        features.sort(key=lambda x: x["importance"], reverse=True)
        return jsonify({"success": True, "features": features})
    except Exception as e:
        print(f"Error in catalog_features: {str(e)}")
        return jsonify({"success": False, "message": f"Error analyzing features: {str(e)}"})

@catalog_bp.route("/versions/<filename>")
def catalog_versions(filename):
    try:
        metadata = load_version_metadata(filename)
        cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
        main_file_path = os.path.join(cleaned_data_dir, filename)
        if not metadata["versions"] and os.path.exists(main_file_path):
            df = pd.read_csv(main_file_path)
            create_new_version(filename, df, "Initial version", "system")
            metadata = load_version_metadata(filename)
        versions = sorted(metadata["versions"], key=lambda x: x["version"], reverse=True)
        return jsonify({"success": True, "versions": versions})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@catalog_bp.route("/restore_version/<filename>/<int:version>", methods=["POST"])
def restore_version(filename, version):
    try:
        version_file_path = get_version_file_path(filename, version)
        if not os.path.exists(version_file_path):
            return jsonify({"success": False, "message": f"Version {version} not found"})
        df = pd.read_csv(version_file_path)
        data = request.get_json()
        user = data.get("user", "system")
        new_version = create_new_version(filename, df, f"Restored from version {version}", user)
        return jsonify({
            "success": True, 
            "message": f"Version {version} restored as version {new_version}",
            "new_version": new_version
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@catalog_bp.route("/download_version/<filename>/<int:version>")
def download_version(filename, version):
    version_file_path = get_version_file_path(filename, version)
    if not os.path.exists(version_file_path):
        return jsonify({"success": False, "message": "Version not found"}), 404
    versions_dir = os.path.join(current_app.root_path, "cleaned_data", "versions")
    version_filename = f"{filename}_v{version}"
    return send_from_directory(versions_dir, version_filename, as_attachment=True)

@catalog_bp.route("/create_version/<filename>", methods=["POST"])
def create_version(filename):
    try:
        cleaned_data_dir = os.path.join(current_app.root_path, "cleaned_data")
        main_file_path = os.path.join(cleaned_data_dir, filename)
        if not os.path.exists(main_file_path):
            return jsonify({"success": False, "message": "File not found"})
        df = pd.read_csv(main_file_path)
        data = request.get_json()
        changes = data.get("changes", "Manual version creation")
        user = data.get("user", "system")
        new_version = create_new_version(filename, df, changes, user)
        return jsonify({
            "success": True,
            "message": f"Version {new_version} created successfully",
            "version": new_version
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

@catalog_bp.route("/prepare_cleaning/<filename>/<int:version>", methods=["POST"])
def prepare_cleaning(filename, version):
    try:
        from flask import session
        version_file_path = get_version_file_path(filename, version)
        if not os.path.exists(version_file_path):
            return jsonify({"success": False, "message": f"Version {version} not found"})
        uploads_folder = os.path.join(current_app.root_path, "uploads")
        os.makedirs(uploads_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        edit_filename = f"edit_{version}_{timestamp}_{filename}"
        target_path = os.path.join(uploads_folder, edit_filename)
        shutil.copy2(version_file_path, target_path)
        session["uploaded_file"] = edit_filename
        session["original_file"] = edit_filename
        session["current_domain"] = "connection"
        session["edit_mode"] = True
        session["edit_version"] = version
        session["original_catalog_filename"] = filename
        return jsonify({
            "success": True,
            "message": f"Version {version} prepared for editing",
            "edit_filename": edit_filename,
            "original_filename": filename,
            "edit_version": version
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})