import os
import json
import pickle
import shutil
import zipfile
import tempfile
import pandas as pd
import numpy as np
from urllib.parse import unquote
from flask import Blueprint, request, jsonify, current_app, render_template, send_file
from pages.train.train_supervised import (
    get_model_configs, detect_problem_type, train_models,
    is_target_candidate, generate_enhanced_preview
)
from pages.train.train_unsupervised import (
    CLUSTERING_MODELS, DIMENSIONALITY_REDUCTION_MODELS, ASSOCIATION_RULE_MODELS, ANOMALY_DETECTION_MODELS,
    train_clustering_model, train_dimensionality_reduction_model, train_anomaly_model,
    evaluate_dimensionality_reduction, preprocess_association_data, train_association_model, preprocess_data
)
from datetime import datetime
import uuid

train_bp = Blueprint('train_bp', __name__, url_prefix='/train', template_folder='templates')

def get_display_score(results, problem_type):
    def pick_not_none(*args):
        for v in args:
            if v is not None:
                return v
        return None

    score = 0.0
    label = ""
    for model_key, result in results.items():
        if isinstance(result, dict) and 'error' in result:
            continue
            
        # Clustering: Silhouette Score (0-1, negative treated as 0)
        if problem_type == "clustering" or (result.get('type') == 'clustering'):
            val = pick_not_none(
                result.get('silhouette_score'),
                result.get('metrics', {}).get('silhouette_score'),
                result.get('metrics', {}).get('test', {}).get('silhouette_score')
            )
            if val is not None:
                try:
                    val = max(float(val), 0)  # Ensure non-negative
                    scaled_val = val * 100
                    if scaled_val > score:
                        score = scaled_val
                        label = "Silhouette Score"
                except Exception:
                    continue
                    
        # Dimensionality Reduction: Variance Explained (0-1)
        elif problem_type == "dimensionality_reduction" or (result.get('type') == 'dimensionality_reduction'):
            val = pick_not_none(
                result.get('variance_explained'),
                result.get('metrics', {}).get('variance_explained'),
                result.get('metrics', {}).get('test', {}).get('variance_explained')
            )
            if val is not None:
                try:
                    scaled_val = float(val) * 100
                    if scaled_val > score:
                        score = scaled_val
                        label = "Variance Explained"
                except Exception:
                    continue
                    
        # Anomaly: Outlier Ratio (0-1)
        elif problem_type == "anomaly" or (result.get('type') == 'anomaly_detection'):
            val = pick_not_none(
                result.get('anomaly_ratio'),
                result.get('metrics', {}).get('anomaly_ratio'),
                result.get('metrics', {}).get('test', {}).get('anomaly_ratio')
            )
            if val is not None:
                try:
                    scaled_val = float(val) * 100
                    if scaled_val > score:
                        score = scaled_val
                        label = "Anomaly Ratio"
                except Exception:
                    continue
                    
        # Association: Rules Found (scale as min(rules_count/10, 1.0) * 100)
        elif problem_type == "association" or (result.get('type') == 'association_rules'):
            rules = pick_not_none(
                result.get('rules_count'),
                result.get('metrics', {}).get('rules_count'),
                result.get('metrics', {}).get('test', {}).get('rules_count'),
                0
            )
            try:
                rules = int(rules)
                scaled_val = min(rules / 10.0, 1.0) * 100
                if scaled_val > score:
                    score = scaled_val
                    label = "Rules Found"
            except Exception:
                continue
                
        # Classification: Accuracy (0-1)
        elif problem_type == "classification":
            acc = pick_not_none(
                result.get('metrics', {}).get('test', {}).get('accuracy'),
                result.get('metrics', {}).get('accuracy')
            )
            if acc is not None:
                try:
                    scaled_val = float(acc) * 100
                    if scaled_val > score:
                        score = scaled_val
                        label = "Accuracy"
                except Exception:
                    continue
                    
        # Regression: R2 Score (negative treated as 0)
        elif problem_type == "regression":
            r2 = pick_not_none(
                result.get('metrics', {}).get('test', {}).get('r2'),
                result.get('metrics', {}).get('r2')
            )
            if r2 is not None:
                try:
                    scaled_val = max(float(r2), 0) * 100  # Ensure non-negative
                    if scaled_val > score:
                        score = scaled_val
                        label = "R² Score"
                except Exception:
                    continue
                    
    return round(score, 1), label

# Project Management Routes and all your original logic...
@train_bp.route('/projects', methods=['GET'])
def get_projects():
    """Get all available projects"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')
        if not os.path.exists(projects_dir):
            os.makedirs(projects_dir)
        projects = {}
        for project_folder in os.listdir(projects_dir):
            project_path = os.path.join(projects_dir, project_folder)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, 'project_config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            project_config = json.load(f)
                        # Count models in project
                        models_dir = os.path.join(project_path, 'models')
                        model_count = 0
                        if os.path.exists(models_dir):
                            model_count = len([d for d in os.listdir(models_dir) 
                                             if os.path.isdir(os.path.join(models_dir, d))])
                        project_config['model_count'] = model_count
                        projects[project_folder] = project_config
                    except Exception as e:
                        print(f"Error loading project {project_folder}: {e}")
                        continue
        return jsonify({'success': True, 'projects': projects})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading projects: {str(e)}'})

@train_bp.route('/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.get_json()
        project_name = data.get('name', '').strip()
        project_description = data.get('description', '').strip()
        if not project_name:
            return jsonify({'success': False, 'message': 'Project name is required'})
        project_id = f"proj_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"
        projects_dir = os.path.join(current_app.root_path, 'projects')
        os.makedirs(projects_dir, exist_ok=True)
        project_path = os.path.join(projects_dir, project_id)
        os.makedirs(project_path)
        os.makedirs(os.path.join(project_path, 'models'))
        project_config = {
            'id': project_id,
            'name': project_name,
            'description': project_description,
            'created_at': datetime.now().isoformat(),
            'model_count': 0,
            'last_updated': datetime.now().isoformat()
        }
        config_path = os.path.join(project_path, 'project_config.json')
        with open(config_path, 'w') as f:
            json.dump(project_config, f, indent=2)
        return jsonify({
            'success': True, 
            'message': 'Project created successfully',
            'project_id': project_id,
            'project': project_config
        })
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error creating project: {str(e)}'})

@train_bp.route('/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project and all its models"""
    try:
        project_path = os.path.join(current_app.root_path, 'projects', project_id)
        if not os.path.exists(project_path):
            return jsonify({'success': False, 'message': 'Project not found'})
        if '..' in project_id or '/' in project_id or '\\' in project_id:
            return jsonify({'success': False, 'message': 'Invalid project ID'})
        shutil.rmtree(project_path)
        return jsonify({'success': True, 'message': 'Project deleted successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting project: {str(e)}'})

@train_bp.route('/projects/<project_id>/models', methods=['GET'])
def get_project_models(project_id):
    """Get all models in a specific project"""
    try:
        project_path = os.path.join(current_app.root_path, 'projects', project_id)
        models_dir = os.path.join(project_path, 'models')
        if not os.path.exists(models_dir):
            return jsonify({'success': True, 'models': []})
        models = []
        for model_id in os.listdir(models_dir):
            model_dir = os.path.join(models_dir, model_id)
            if os.path.isdir(model_dir):
                results_path = os.path.join(model_dir, 'results.json')
                if os.path.exists(results_path):
                    try:
                        with open(results_path, 'r') as f:
                            data = json.load(f)
                        config = data.get('config', {})
                        results = data.get('results', {})
                        display_score, display_score_label = get_display_score(results, config.get('problem_type', 'Unknown'))
                        model_summary = {
                            'id': model_id,
                            'config': config,
                            'results': results,
                            'created_at': config.get('created_at', 'Unknown'),
                            'problem_type': config.get('problem_type', 'Unknown'),
                            'training_mode': config.get('training_mode', 'self-train'),
                            'n_models': len(results),
                            'best_metric': get_best_metric(results, config.get('problem_type')),
                            'display_score': display_score,
                            'display_score_label': display_score_label
                        }
                        models.append(model_summary)
                    except Exception as e:
                        print(f"Error loading model {model_id}: {e}")
                        continue
        models.sort(key=lambda x: x['created_at'], reverse=True)
        return jsonify({'success': True, 'models': models})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error loading project models: {str(e)}'})


# Model Comparison Routes
@train_bp.route('/projects/<project_id>/compare/<model_a_id>/<model_b_id>', methods=['GET'])
def compare_project_models(project_id, model_a_id, model_b_id):
    """Compare two models within the same project"""
    try:
        project_path = os.path.join(current_app.root_path, 'projects', project_id)
        models_dir = os.path.join(project_path, 'models')
        
        # Load both models
        model_a_path = os.path.join(models_dir, model_a_id, 'results.json')
        model_b_path = os.path.join(models_dir, model_b_id, 'results.json')
        
        if not os.path.exists(model_a_path) or not os.path.exists(model_b_path):
            return jsonify({'success': False, 'message': 'One or both models not found'})
        
        with open(model_a_path, 'r') as f:
            model_a_data = json.load(f)
        
        with open(model_b_path, 'r') as f:
            model_b_data = json.load(f)
        
        # Perform comparison
        comparison_data = {
            'model_a': {
                'id': model_a_id,
                'data': model_a_data
            },
            'model_b': {
                'id': model_b_id,
                'data': model_b_data
            },
            'winner': determine_winner(model_a_data, model_b_data),
            'comparison_metrics': generate_comparison_metrics(model_a_data, model_b_data)
        }
        
        return jsonify({'success': True, 'comparison': comparison_data})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error comparing models: {str(e)}'})

def determine_winner(model_a_data, model_b_data):
    """Determine which model performs better"""
    try:
        best_a = find_best_model_in_data(model_a_data)
        best_b = find_best_model_in_data(model_b_data)
        
        if not best_a or not best_b:
            return 'Unable to determine'
        
        if best_a['score'] > best_b['score']:
            return 'model_a'
        elif best_b['score'] > best_a['score']:
            return 'model_b'
        else:
            return 'tie'
    except:
        return 'Unable to determine'

def find_best_model_in_data(model_data):
    """Find the best performing model in the results"""
    try:
        results = model_data.get('results', {})
        problem_type = model_data.get('config', {}).get('problem_type', 'classification')
        
        best_score = -1
        best_model = None
        
        for model_key, result in results.items():
            if 'metrics' in result and 'test' in result['metrics']:
                test_metrics = result['metrics']['test']
                if problem_type == 'classification':
                    score = test_metrics.get('accuracy', 0)
                elif problem_type == 'regression':
                    score = test_metrics.get('r2', 0)
                else:
                    score = 0.5
                
                if score > best_score:
                    best_score = score
                    best_model = {
                        'key': model_key,
                        'score': score,
                        'metrics': test_metrics
                    }
        
        return best_model
    except:
        return None

def generate_comparison_metrics(model_a_data, model_b_data):
    """Generate detailed comparison metrics"""
    try:
        best_a = find_best_model_in_data(model_a_data)
        best_b = find_best_model_in_data(model_b_data)
        
        if not best_a or not best_b:
            return {}
        
        problem_type = model_a_data.get('config', {}).get('problem_type', 'classification')
        
        if problem_type == 'classification':
            return {
                'accuracy': {
                    'model_a': best_a['metrics'].get('accuracy', 0),
                    'model_b': best_b['metrics'].get('accuracy', 0)
                },
                'precision': {
                    'model_a': best_a['metrics'].get('precision', 0),
                    'model_b': best_b['metrics'].get('precision', 0)
                },
                'recall': {
                    'model_a': best_a['metrics'].get('recall', 0),
                    'model_b': best_b['metrics'].get('recall', 0)
                },
                'f1': {
                    'model_a': best_a['metrics'].get('f1', 0),
                    'model_b': best_b['metrics'].get('f1', 0)
                }
            }
        else:  # regression
            return {
                'r2': {
                    'model_a': best_a['metrics'].get('r2', 0),
                    'model_b': best_b['metrics'].get('r2', 0)
                },
                'rmse': {
                    'model_a': best_a['metrics'].get('rmse', 0),
                    'model_b': best_b['metrics'].get('rmse', 0)
                },
                'mae': {
                    'model_a': best_a['metrics'].get('mae', 0),
                    'model_b': best_b['metrics'].get('mae', 0)
                }
            }
    except:
        return {}

# XAI Page Route
@train_bp.route('/xai')
def xai_page():
    """Enhanced XAI page to view and manage projects and models"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')
        
        if not os.path.exists(projects_dir):
            return render_template('xai/xai.html', projects=[])
        
        # Get all projects with their models
        projects = []
        for project_folder in os.listdir(projects_dir):
            project_path = os.path.join(projects_dir, project_folder)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, 'project_config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            project_config = json.load(f)
                        
                        # Get models in this project
                        models_dir = os.path.join(project_path, 'models')
                        project_models = []
                        
                        if os.path.exists(models_dir):
                            for model_id in os.listdir(models_dir):
                                model_dir = os.path.join(models_dir, model_id)
                                if os.path.isdir(model_dir):
                                    results_path = os.path.join(model_dir, 'results.json')
                                    if os.path.exists(results_path):
                                        try:
                                            with open(results_path, 'r') as f:
                                                data = json.load(f)
                                            
                                            config = data.get('config', {})
                                            results = data.get('results', {})
                                            
                                            model_summary = {
                                                'id': model_id,
                                                'config': config,
                                                'results': results,
                                                'created_at': config.get('created_at', 'Unknown'),
                                                'problem_type': config.get('problem_type', 'Unknown'),
                                                'training_mode': config.get('training_mode', 'self-train'),
                                                'n_models': len(results),
                                                'best_metric': get_best_metric(results, config.get('problem_type'))
                                            }
                                            
                                            project_models.append(model_summary)
                                        except Exception as e:
                                            print(f"Error loading model {model_id}: {e}")
                                            continue
                        
                        # Sort models by creation date (newest first)
                        project_models.sort(key=lambda x: x['created_at'], reverse=True)
                        
                        # Update model count
                        project_config['model_count'] = len(project_models)
                        project_config['models'] = project_models
                        project_config['id'] = project_folder  # Add project ID
                        
                        projects.append(project_config)
                        
                    except Exception as e:
                        print(f"Error loading project {project_folder}: {e}")
                        continue
        
        # Sort projects by creation date (newest first)
        projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return render_template('xai/xai.html', projects=projects)
    except Exception as e:
        return f"Error loading projects: {str(e)}", 500

@train_bp.route('/debug/list_files')
def debug_list_files():
    """Debug route to list available files"""
    try:
        cleaned_data_dir = os.path.join(current_app.root_path, 'cleaned_data')
        if os.path.exists(cleaned_data_dir):
            files = os.listdir(cleaned_data_dir)
            csv_files = [f for f in files if f.endswith('.csv')]
            return jsonify({
                'success': True,
                'files': files, 
                'csv_files': csv_files,
                'directory': cleaned_data_dir,
                'file_count': len(files),
                'csv_count': len(csv_files),
                'root_path': current_app.root_path
            })
        else:
            return jsonify({
                'success': False,
                'error': 'cleaned_data directory not found', 
                'directory': cleaned_data_dir,
                'root_path': current_app.root_path
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@train_bp.route('/get_dataset_info')
def get_dataset_info():
    """Get comprehensive dataset information for training interface"""
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'success': False, 'message': 'Filename is required'})
    
    # Decode URL-encoded filename
    filename = unquote(filename)
    
    try:
        # Correct path: cleaned_data is at the project root
        cleaned_data_dir = os.path.join(current_app.root_path, 'cleaned_data')
        filepath = os.path.join(cleaned_data_dir, filename)
        
        if not os.path.isfile(filepath):
            # List available files for debugging
            available_files = []
            if os.path.exists(cleaned_data_dir):
                available_files = [f for f in os.listdir(cleaned_data_dir) if f.endswith('.csv')]
            
            error_message = f'File not found: {filename}'
            if available_files:
                error_message += f'. Available files: {", ".join(available_files[:5])}'
                if len(available_files) > 5:
                    error_message += f' (and {len(available_files) - 5} more)'
            
            return jsonify({
                'success': False, 
                'message': error_message,
                'available_files': available_files
            })

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({'success': False, 'message': f'Error reading file: {str(e)}'})
        
        # Basic dataset info
        row_count = len(df)
        column_count = len(df.columns)
        missing_values = int(df.isnull().sum().sum())
        
        # Enhanced column information for training interface
        columns_info = []
        for col in df.columns:
            col_info = {
                'name': col,
                'dtype': str(df[col].dtype),
                'missing_count': int(df[col].isnull().sum()),
                'missing_percentage': round((df[col].isnull().sum() / len(df)) * 100, 2),
                'unique_values': int(df[col].nunique()),
                'is_numeric': pd.api.types.is_numeric_dtype(df[col]),
                'is_target_candidate': is_target_candidate(df[col]),
                'sample_values': df[col].dropna().head(3).tolist() if not df[col].empty else []
            }
            columns_info.append(col_info)
        
        # Determine recommended task type for each target candidate
        recommended_tasks = {}
        for col in df.columns:
            if is_target_candidate(df[col]):
                try:
                    recommended_tasks[col] = detect_problem_type(df[col])
                except Exception:
                    recommended_tasks[col] = 'classification'
        
        return jsonify({
            'success': True,
            'row_count': row_count,
            'column_count': column_count,
            'missing_values': missing_values,
            'columns': list(df.columns),
            'columns_info': columns_info,
            'data_shape': [row_count, column_count],
            'target_candidates': [col['name'] for col in columns_info if col['is_target_candidate']],
            'recommended_tasks': recommended_tasks
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Unexpected error: {str(e)}'})

@train_bp.route('/start_training', methods=['POST'])
def start_training():
    """Enhanced training endpoint with proper handling for supervised and unsupervised tasks"""
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'message': 'No data provided'})

        # Validate required fields
        required_fields = ['filename', 'project_id']
        if data.get('problem_type') != 'unsupervised':
            required_fields.extend(['feature_columns', 'target_column'])

        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'message': f"Missing required fields: {', '.join(missing_fields)}"
            })

        # Load dataset
        cleaned_data_dir = os.path.join(current_app.root_path, 'cleaned_data')
        filepath = os.path.join(cleaned_data_dir, data['filename'])

        if not os.path.isfile(filepath):
            return jsonify({
                'success': False,
                'message': f"Data file not found: {data['filename']}"
            })

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error reading data file: {str(e)}'
            })

        # Determine model directory
        project_path = os.path.join(current_app.root_path, 'projects', data['project_id'])
        model_dir = os.path.join(project_path, 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Dispatch to appropriate training function
        if data.get('problem_type') == 'unsupervised':
            print(f"[DEBUG] Starting unsupervised training with methods: {data.get('selected_methods', [])}")
            
            # Use all numeric columns if feature_columns not specified
            feature_columns = data.get('feature_columns', df.select_dtypes(include=[np.number]).columns.tolist())
            
            result = handle_unsupervised_training(
                df=df,
                feature_columns=feature_columns,
                selected_methods=data.get('selected_methods', ['kmeans']),
                model_params=data.get('model_params', {}),
                model_id=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                project_path=project_path,
                model_dir=model_dir,
                project_id=data['project_id']
            )
            
            return result
        else:
            train_result = train_models(
                df=df,
                target_column=data['target_column'],
                feature_columns=data['feature_columns'],
                problem_type=data.get('problem_type', 'auto'),
                model_name=f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                selected_models=data.get('selected_models', []),
                train_test_config=data.get('train_test_config', {}),
                cv_config=data.get('cv_config', {}),
                model_params=data.get('model_params', {}),
                model_dir=model_dir
            )

            if not train_result['success']:
                return jsonify({
                    'success': False,
                    'message': train_result['error'],
                    'details': train_result.get('traceback', 'No additional details')
                })

            return jsonify({
                'success': True,
                'model_id': train_result.get('model_id'),
                'results': train_result['results'],
                'config': train_result['config'],
                'message': 'Training completed successfully'
            })

    except Exception as e:
        print(f"Error in start_training: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'Unexpected error during training: {str(e)}'
        })

def handle_unsupervised_training(df, feature_columns, selected_methods, model_params, model_id, project_path, model_dir, project_id):
    """Handle unsupervised learning training (supports clustering, anomaly, dimensionality_reduction, association)"""
    try:
        from pages.train.train_unsupervised import train_unsupervised_models
        
        if not feature_columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                return jsonify({'success': False, 'message': 'No numeric columns available for unsupervised learning'})
            feature_columns = numeric_cols

        # Determine the unsupervised type from the selected methods
        method_to_type_map = {
            'kmeans': 'clustering',
            'dbscan': 'clustering', 
            'agglomerative': 'clustering',
            'pca': 'dimensionality_reduction',
            'tsne': 'dimensionality_reduction',
            'isolation_forest': 'anomaly',
            'one_class_svm': 'anomaly',
            'apriori': 'association'
        }
        
        unsupervised_type = 'clustering'  # default
        if selected_methods:
            unsupervised_type = method_to_type_map.get(selected_methods[0], 'clustering')
        
        # Use the corrected training function
        result = train_unsupervised_models(
            df=df,
            feature_columns=feature_columns,
            unsupervised_type=unsupervised_type,
            selected_methods=selected_methods,
            model_params=model_params,
            model_dir=model_dir  # Pass the base models directory, not the specific model dir
        )
        
        if result['success']:
            # Update project config
            config_path = os.path.join(project_path, 'project_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    project_config = json.load(f)
                
                # Recount models
                models_dir = os.path.join(project_path, 'models')
                if os.path.exists(models_dir):
                    model_count = len([d for d in os.listdir(models_dir) 
                                     if os.path.isdir(os.path.join(models_dir, d))])
                else:
                    model_count = 0
                
                project_config['model_count'] = model_count
                project_config['last_updated'] = datetime.now().isoformat()
                
                with open(config_path, 'w') as f:
                    json.dump(project_config, f, indent=2)

            return jsonify({
                'success': True,
                'model_id': result['model_id'],
                'results': result['results'],
                'config': result['config']
            })
        else:
            return jsonify({
                'success': False,
                'message': result['error']
            })
        
    except Exception as e:
        print(f"Error in unsupervised training handler: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': f'Unsupervised training failed: {str(e)}'})
    
def save_unsupervised_results_to_project(model_id, model_objects, results, config, model_dir):
    """Save unsupervised learning results"""
    try:
        os.makedirs(model_dir, exist_ok=True)
        
        # Save models as .pkl
        for model_key, model_data in model_objects.items():
            if model_data['model'] is not None:
                model_path = os.path.join(model_dir, f'{model_key}.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data['model'], f)
        
        # Save metadata as JSON
        enhanced_config = {
            **config,
            'created_at': datetime.now().isoformat(),
            'model_type': 'unsupervised',
            'training_mode': 'self-train'
        }
        
        with open(os.path.join(model_dir, 'results.json'), 'w') as f:
            json.dump({
                'config': enhanced_config,
                'results': results
            }, f, indent=2)
        
    except Exception as e:
        print(f"Error saving unsupervised models: {e}")

# Model Management Routes
@train_bp.route('/get_training_status/<model_id>')
def get_training_status(model_id):
    """Get training status and results"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')
        if os.path.exists(projects_dir):
            for project_folder in os.listdir(projects_dir):
                project_path = os.path.join(projects_dir, project_folder)
                if os.path.isdir(project_path):
                    model_path = os.path.join(project_path, 'models', model_id, 'results.json')
                    if os.path.exists(model_path):
                        with open(model_path, 'r') as f:
                            data = json.load(f)
                        return jsonify({
                            'success': True,
                            'model_id': model_id,
                            'results': data['results'],
                            'config': data['config'],
                            'project_id': project_folder,
                            'created_at': data['config'].get('created_at', 'Unknown')
                        })

        return jsonify({'success': False, 'message': 'Training results not found'})
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error retrieving results: {str(e)}'})

@train_bp.route('/download_model/<model_id>')
def download_model(model_id):
    """Download the best performing model as a PKL file"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')
        model_dir = None
        
        if os.path.exists(projects_dir):
            for project_folder in os.listdir(projects_dir):
                project_path = os.path.join(projects_dir, project_folder)
                if os.path.isdir(project_path):
                    potential_model_dir = os.path.join(project_path, 'models', model_id)
                    if os.path.exists(potential_model_dir):
                        model_dir = potential_model_dir
                        break

        if not model_dir:
            return jsonify({'success': False, 'message': 'Model not found'})
        
        # Security check
        if '..' in model_id or '/' in model_id or '\\' in model_id:
            return jsonify({'success': False, 'message': 'Invalid model ID'})
        
        # Find first available .pkl file
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        if not pkl_files:
            return jsonify({'success': False, 'message': 'No model files found'})
        
        model_file = pkl_files[0]  # Default to first found
        model_path = os.path.join(model_dir, model_file)
        
        return send_file(
            model_path,
            as_attachment=True,
            download_name=f"{model_id}_model.pkl",
            mimetype='application/octet-stream'
        )
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error downloading model: {str(e)}'})

@train_bp.route('/download_all_models/<model_id>')
def download_all_models(model_id):
    """Download all trained model files as a ZIP archive"""
    try:
        # Check projects only
        projects_dir = os.path.join(current_app.root_path, 'projects')
        model_dir = None
        
        if os.path.exists(projects_dir):
            for project_folder in os.listdir(projects_dir):
                project_path = os.path.join(projects_dir, project_folder)
                if os.path.isdir(project_path):
                    potential_model_dir = os.path.join(project_path, 'models', model_id)
                    if os.path.exists(potential_model_dir):
                        model_dir = potential_model_dir
                        break
        
        if not model_dir:
            return jsonify({'success': False, 'message': 'Model not found'})
        
        # Security check
        if '..' in model_id or '/' in model_id or '\\' in model_id:
            return jsonify({'success': False, 'message': 'Invalid model ID'})
        
        pkl_files = [f for f in os.listdir(model_dir) if f.endswith('.pkl')]
        
        if not pkl_files:
            return jsonify({'success': False, 'message': 'No model files found'})
        
        # Create temporary zip file
        temp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(temp_dir, f'{model_id}_all_models.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for pkl_file in pkl_files:
                file_path = os.path.join(model_dir, pkl_file)
                zipf.write(file_path, pkl_file)
            
            # Include results.json
            results_path = os.path.join(model_dir, 'results.json')
            if os.path.exists(results_path):
                zipf.write(results_path, 'model_info.json')
        
        return send_file(zip_path, as_attachment=True,
                        download_name=f'{model_id}_all_models.zip',
                        mimetype='application/zip')
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error downloading all models: {str(e)}'})

@train_bp.route('/export_results/<model_id>')
def export_results(model_id):
    """Export training results as JSON"""
    try:
        # Check projects only
        projects_dir = os.path.join(current_app.root_path, 'projects')
        results_path = None
        
        if os.path.exists(projects_dir):
            for project_folder in os.listdir(projects_dir):
                project_path = os.path.join(projects_dir, project_folder)
                if os.path.isdir(project_path):
                    potential_results_path = os.path.join(project_path, 'models', model_id, 'results.json')
                    if os.path.exists(potential_results_path):
                        results_path = potential_results_path
                        break
        
        if not results_path:
            return jsonify({'success': False, 'message': 'Results not found'})
        
        return send_file(results_path, as_attachment=True,
                        download_name=f'{model_id}_results.json')
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error exporting results: {str(e)}'})

@train_bp.route('/delete_model/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a trained model and all its associated files"""
    try:
        # Check projects only
        projects_dir = os.path.join(current_app.root_path, 'projects')
        model_dir = None
        project_id = None
        
        if os.path.exists(projects_dir):
            for project_folder in os.listdir(projects_dir):
                project_path = os.path.join(projects_dir, project_folder)
                if os.path.isdir(project_path):
                    potential_model_dir = os.path.join(project_path, 'models', model_id)
                    if os.path.exists(potential_model_dir):
                        model_dir = potential_model_dir
                        project_id = project_folder
                        break
        
        if not model_dir:
            return jsonify({'success': False, 'message': 'Model not found'})
        
        # Security check
        if '..' in model_id or '/' in model_id or '\\' in model_id:
            return jsonify({'success': False, 'message': 'Invalid model ID'})
        
        # Remove the model directory
        shutil.rmtree(model_dir)
        
        # Update project model count if it's in a project
        if project_id:
            project_path = os.path.join(projects_dir, project_id)
            config_path = os.path.join(project_path, 'project_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    project_config = json.load(f)
                
                # Recount models
                models_dir = os.path.join(project_path, 'models')
                if os.path.exists(models_dir):
                    model_count = len([d for d in os.listdir(models_dir) 
                                     if os.path.isdir(os.path.join(models_dir, d))])
                else:
                    model_count = 0
                
                project_config['model_count'] = model_count
                project_config['last_updated'] = datetime.now().isoformat()
                
                with open(config_path, 'w') as f:
                    json.dump(project_config, f, indent=2)
        
        return jsonify({'success': True, 'message': f'Model {model_id} deleted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'message': f'Error deleting model: {str(e)}'})

def get_best_metric(results, problem_type):
    """Get the best metric value across all models (supports both supervised and unsupervised)"""
    if not results or not problem_type:
        return None
    
    best_value = None
    best_model = None
    metric_name = None

    for model_key, result in results.items():
        if isinstance(result, dict) and 'error' in result:
            continue  # Skip models with errors
            
        # Handle supervised models
        if 'metrics' in result and 'test' in result['metrics']:
            test_metrics = result['metrics']['test']

            if problem_type == 'classification':
                metric_value = test_metrics.get('accuracy', 0)
                metric_name = 'accuracy'
            elif problem_type == 'regression':
                metric_value = test_metrics.get('r2', 0)
                metric_name = 'r2'
            else:
                metric_value = 0
                metric_name = 'score'
                
        # Handle unsupervised models
        elif problem_type in ['clustering', 'anomaly', 'dimensionality_reduction', 'association', 'unsupervised']:
            if problem_type == 'clustering' or (result.get('type') == 'clustering'):
                # For clustering, prefer silhouette score, fallback to cluster count
                if 'metrics' in result and result['metrics']:
                    metric_value = result['metrics'].get('silhouette_score', 0)
                    metric_name = 'silhouette_score'
                elif 'n_clusters' in result:
                    metric_value = result['n_clusters']
                    metric_name = 'clusters_found'
                else:
                    metric_value = 1
                    metric_name = 'completed'
                    
            elif problem_type == 'anomaly' or (result.get('type') == 'anomaly_detection'):
                if 'anomaly_ratio' in result:
                    metric_value = result['anomaly_ratio']
                    metric_name = 'outlier_ratio'
                elif 'anomalies_detected' in result:
                    metric_value = result['anomalies_detected']
                    metric_name = 'anomalies_found'
                else:
                    metric_value = 1
                    metric_name = 'completed'
                    
            elif problem_type == 'dimensionality_reduction' or (result.get('type') == 'dimensionality_reduction'):
                if 'variance_explained' in result:
                    metric_value = result['variance_explained']
                    metric_name = 'variance_ratio'
                elif 'kl_divergence' in result:
                    metric_value = 1 / (1 + result['kl_divergence'])
                    metric_name = 'kl_score'
                else:
                    metric_value = 1
                    metric_name = 'completed'
                    
            elif problem_type == 'association' or (result.get('type') == 'association_rules'):
                if 'rules_count' in result:
                    metric_value = result['rules_count']
                    metric_name = 'max_lift'
                elif 'frequent_itemsets_count' in result:
                    metric_value = result['frequent_itemsets_count']
                    metric_name = 'itemsets_found'
                else:
                    metric_value = 1
                    metric_name = 'completed'
            else:
                metric_value = 1
                metric_name = 'completed'
        else:
            continue

        if best_value is None or metric_value > best_value:
            best_value = metric_value
            best_model = result.get('name', model_key)

    if best_value is not None:
        return {
            'value': round(best_value, 4) if isinstance(best_value, (int, float)) else best_value,
            'model': best_model,
            'metric': metric_name
        }
    return None

@train_bp.route('/')
def train_page():
    """Main training page"""
    filename = request.args.get('filename')
    file_missing = request.args.get('file_missing')

    # If no filename provided at all, show the "select dataset" prompt
    if not filename:
        return render_template('train/train.html', 
                             filename=None, 
                             file_missing=file_missing)

    # Decode the filename
    decoded_filename = unquote(filename)
    
    # Validate file exists
    cleaned_data_dir = os.path.join(current_app.root_path, 'cleaned_data')
    filepath = os.path.join(cleaned_data_dir, decoded_filename)
    
    if not os.path.exists(filepath):
        # File doesn't exist, show the "select dataset" prompt with error message
        return render_template('train/train.html', 
                             filename=None, 
                             file_missing=decoded_filename)
    
    # File exists, show training interface
    return render_template('train/train.html', filename=filename)
    
    return render_template('train/train.html', filename=filename)

@train_bp.route('/xai', endpoint='xai')
def xai_page():
    """Enhanced XAI page to view and manage projects and models"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')
        
        if not os.path.exists(projects_dir):
            return render_template('xai/xai.html', projects=[])
        
        # Get all projects with their models
        projects = []
        for project_folder in os.listdir(projects_dir):
            project_path = os.path.join(projects_dir, project_folder)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, 'project_config.json')
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            project_config = json.load(f)
                        
                        # Get models in this project
                        models_dir = os.path.join(project_path, 'models')
                        project_models = []
                        
                        if os.path.exists(models_dir):
                            for model_id in os.listdir(models_dir):
                                model_dir = os.path.join(models_dir, model_id)
                                if os.path.isdir(model_dir):
                                    results_path = os.path.join(model_dir, 'results.json')
                                    if os.path.exists(results_path):
                                        try:
                                            with open(results_path, 'r') as f:
                                                data = json.load(f)
                                            
                                            config = data.get('config', {})
                                            results = data.get('results', {})
                                            
                                            # Compute display_score and display_score_label
                                            display_score, display_score_label = get_display_score(
                                                results, config.get('problem_type', 'Unknown')
                                            )
                                            
                                            model_summary = {
                                                'id': model_id,
                                                'config': config,
                                                'results': results,
                                                'created_at': config.get('created_at', 'Unknown'),
                                                'problem_type': config.get('problem_type', 'Unknown'),
                                                'training_mode': config.get('training_mode', 'self-train'),
                                                'n_models': len(results),
                                                'best_metric': get_best_metric(results, config.get('problem_type')),
                                                'display_score': display_score,
                                                'display_score_label': display_score_label
                                            }
                                            
                                            project_models.append(model_summary)
                                        except Exception as e:
                                            print(f"Error loading model {model_id}: {e}")
                                            continue
                        # Sort models by creation date (newest first)
                        project_models.sort(key=lambda x: x['created_at'], reverse=True)
                        
                        # Update model count
                        project_config['model_count'] = len(project_models)
                        project_config['models'] = project_models
                        project_config['id'] = project_folder  # Add project ID
                        
                        projects.append(project_config)
                        
                    except Exception as e:
                        print(f"Error loading project {project_folder}: {e}")
                        continue
        
        # Sort projects by creation date (newest first)
        projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return render_template('xai/xai.html', projects=projects)
    except Exception as e:
        return f"Error loading projects: {str(e)}", 500

@train_bp.route('/get_model_hyperparameters')
def get_model_hyperparameters():
    """Get hyperparameter schema/config for a given task type and method"""
    task_type = request.args.get('task_type')
    method = request.args.get('method')
    if not task_type or not method:
        return jsonify({'success': False, 'message': 'Missing task_type or method'})
    
    try:
        if task_type == 'classification':
            from pages.train.train_supervised import get_model_configs
            model_configs = get_model_configs('classification')
        elif task_type == 'regression':
            from pages.train.train_supervised import get_model_configs
            model_configs = get_model_configs('regression')
        elif task_type == 'clustering':
            from pages.train.train_unsupervised import CLUSTERING_MODELS
            model_configs = CLUSTERING_MODELS
        elif task_type == 'dimensionality_reduction':
            from pages.train.train_unsupervised import DIMENSIONALITY_REDUCTION_MODELS
            model_configs = DIMENSIONALITY_REDUCTION_MODELS
        elif task_type == 'association':
            try:
                from pages.train.train_unsupervised import ASSOCIATION_RULE_MODELS
                model_configs = ASSOCIATION_RULE_MODELS
            except ImportError:
                return jsonify({'success': False, 'message': 'Association rule models not available'})
        elif task_type == 'anomaly':
            from pages.train.train_unsupervised import ANOMALY_DETECTION_MODELS
            model_configs = ANOMALY_DETECTION_MODELS
        else:
            return jsonify({'success': False, 'message': f'Unknown task_type: {task_type}'})
        
        model = model_configs.get(method)
        if not model:
            return jsonify({'success': False, 'message': f'Method {method} not found for {task_type}'})
        
        return jsonify({
            'success': True, 
            'params': model.get('params', {}), 
            'name': model.get('name', method), 
            'description': model.get('description', '')
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@train_bp.route('/health')
def health_check():
    """Simple health check route"""
    return jsonify({
        'status': 'healthy',
        'train_blueprint': 'active',
        'timestamp': pd.Timestamp.now().isoformat()
    })
