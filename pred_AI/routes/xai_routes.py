import os
import json
from datetime import datetime
import uuid
from flask import Blueprint, render_template, current_app, jsonify, request, send_file, Response, redirect, url_for
from mlxtend.frequent_patterns import apriori, association_rules
import zipfile
import tempfile
import pickle


xai_bp = Blueprint('xai', __name__)

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
        # Association rule mining functions it is giving 0% 
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


def get_best_metric(results, problem_type):
    """Get the best metric value across all models (supports unsupervised types)"""
    if not results or not problem_type:
        return {
            'value': None,
            'model': None,
            'metric': 'No models trained'
        }

    best_value = None
    best_model = None
    metric_name = ''

    print(f"[DEBUG] get_best_metric called with problem_type: {problem_type}")
    print(f"[DEBUG] Results keys: {list(results.keys()) if results else 'None'}")

    for model_key, result in results.items():
        print(f"[DEBUG] Processing model: {model_key}")
        print(f"[DEBUG] Result structure: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
        if isinstance(result, dict) and 'error' in result:
            print(f"[DEBUG] Skipping model {model_key} due to error: {result['error']}")
            continue
            
        # Handle supervised models (existing logic)
        if 'metrics' in result and isinstance(result['metrics'], dict) and 'test' in result['metrics']:
            test_metrics = result['metrics']['test']
            print(f"[DEBUG] Found supervised metrics: {test_metrics}")

            if problem_type == 'classification':
                metric_value = test_metrics.get('accuracy', 0)
                metric_name = 'accuracy'
            elif problem_type == 'regression':
                metric_value = test_metrics.get('r2', 0)
                metric_name = 'r2'
            else:
                continue
                
        # Handle unsupervised models - Check direct properties first
        elif problem_type in ['clustering', 'anomaly', 'dimensionality_reduction', 'association', 'unsupervised']:
            print(f"[DEBUG] Processing unsupervised model type: {problem_type}")
            print(f"[DEBUG] Result content: {result}")
            
            if problem_type == 'clustering' or (result.get('type') == 'clustering'):
                # Check direct properties first, then metrics
                if 'silhouette_score' in result and result['silhouette_score'] is not None:
                    metric_value = result['silhouette_score']
                    metric_name = 'silhouette_score'
                    print(f"[DEBUG] Found direct silhouette_score: {metric_value}")
                elif 'metrics' in result and result['metrics'] and 'silhouette_score' in result['metrics']:
                    metric_value = result['metrics']['silhouette_score']
                    metric_name = 'silhouette_score'
                    print(f"[DEBUG] Found silhouette_score in metrics: {metric_value}")
                elif 'n_clusters' in result and result['n_clusters'] is not None:
                    metric_value = result['n_clusters']
                    metric_name = 'clusters_found'
                    print(f"[DEBUG] Using cluster count: {metric_value}")
                else:
                    metric_value = 1
                    metric_name = 'completed'
                    print(f"[DEBUG] Using default completed metric")
                    
            elif problem_type == 'anomaly' or (result.get('type') == 'anomaly_detection'):
                if 'anomaly_ratio' in result and result['anomaly_ratio'] is not None:
                    metric_value = result['anomaly_ratio']
                    metric_name = 'outlier_ratio'
                    print(f"[DEBUG] Found anomaly_ratio: {metric_value}")
                elif 'anomalies_detected' in result and result['anomalies_detected'] is not None:
                    metric_value = result['anomalies_detected']
                    metric_name = 'anomalies_found'
                    print(f"[DEBUG] Found anomalies_detected: {metric_value}")
                else:
                    metric_value = 1
                    metric_name = 'completed'
                    print(f"[DEBUG] Using default completed metric for anomaly")
                    
            elif problem_type == 'dimensionality_reduction' or (result.get('type') == 'dimensionality_reduction'):
                if 'variance_explained' in result and result['variance_explained'] is not None:
                    metric_value = result['variance_explained']
                    metric_name = 'variance_explained'
                    print(f"[DEBUG] Found variance_explained: {metric_value}")
                elif 'kl_divergence' in result and result['kl_divergence'] is not None:
                    metric_value = 1 / (1 + result['kl_divergence'])
                    metric_name = 'kl_score'
                    print(f"[DEBUG] Calculated kl_score: {metric_value}")
                else:
                    metric_value = 1
                    metric_name = 'completed'
                    print(f"[DEBUG] Using default completed metric for dim reduction")
                    
            elif problem_type == 'association' or (result.get('type') == 'association_rules'):
                print(f"[DEBUG] Processing association rules model")
                # FIX: Check multiple possible metric locations for association rules
                if 'rules_count' in result and result['rules_count'] is not None:
                    metric_value = result['rules_count']
                    metric_name = 'rules_found'
                    print(f"[DEBUG] Found rules_count: {metric_value}")
                elif 'max_lift' in result and result['max_lift'] is not None:
                    metric_value = result['max_lift']
                    metric_name = 'max_lift'
                    print(f"[DEBUG] Found max_lift: {metric_value}")
                elif 'frequent_itemsets_count' in result and result['frequent_itemsets_count'] is not None:
                    metric_value = result['frequent_itemsets_count']
                    metric_name = 'itemsets_found'
                    print(f"[DEBUG] Found itemsets_count: {metric_value}")
                # FIX: Also check nested metrics structure for association rules
                elif 'metrics' in result and result['metrics']:
                    if isinstance(result['metrics'], dict):
                        if 'test' in result['metrics'] and 'max_lift' in result['metrics']['test']:
                            metric_value = result['metrics']['test']['max_lift']
                            metric_name = 'max_lift'
                            print(f"[DEBUG] Found max_lift in test metrics: {metric_value}")
                        elif 'test' in result['metrics'] and 'rules_count' in result['metrics']['test']:
                            metric_value = result['metrics']['test']['rules_count']
                            metric_name = 'rules_found'
                            print(f"[DEBUG] Found rules_count in test metrics: {metric_value}")
                        elif 'max_lift' in result['metrics']:
                            metric_value = result['metrics']['max_lift']
                            metric_name = 'max_lift'
                            print(f"[DEBUG] Found max_lift in metrics: {metric_value}")
                        elif 'rules_count' in result['metrics']:
                            metric_value = result['metrics']['rules_count']
                            metric_name = 'rules_found'
                            print(f"[DEBUG] Found rules_count in metrics: {metric_value}")
                        else:
                            metric_value = 1
                            metric_name = 'completed'
                            print(f"[DEBUG] Using default completed for association (no metrics found)")
                    else:
                        metric_value = 1
                        metric_name = 'completed'
                        print(f"[DEBUG] Using default completed for association (metrics not dict)")
                # FIX: Check if there's a message indicating no rules found
                elif 'message' in result and 'no frequent itemsets' in result['message'].lower():
                    metric_value = 0
                    metric_name = 'rules_found'
                    print(f"[DEBUG] No frequent itemsets found: {metric_value}")
                else:
                    metric_value = 1
                    metric_name = 'completed'
                    print(f"[DEBUG] Using default completed metric for association")
            else:
                # Generic unsupervised - just mark as completed
                metric_value = 1
                metric_name = 'completed'
                print(f"[DEBUG] Using generic completed metric")
        else:
            print(f"[DEBUG] No valid metrics structure found for {model_key}")
            continue

        print(f"[DEBUG] Extracted metric: {metric_name} = {metric_value}")

        # Update best model logic
        if best_value is None:
            best_value = metric_value
            best_model = result.get('name', model_key)
            print(f"[DEBUG] First model, setting as best: {best_model} with {metric_value}")
        else:
            # For clustering silhouette score, higher is better
            # For anomaly ratio, depends on use case - we'll use higher as "more anomalies detected"
            # For variance explained, higher is better
            # For association rules, more rules is better
            if metric_name in ['silhouette_score', 'variance_explained', 'rules_found', 'itemsets_found', 'anomalies_found', 'clusters_found']:
                if metric_value > best_value:
                    best_value = metric_value
                    best_model = result.get('name', model_key)
                    print(f"[DEBUG] New best model (higher better): {best_model} with {metric_value}")
            # For other metrics like KL divergence or generic completion
            elif metric_name in ['outlier_ratio', 'kl_score', 'completed', 'max_lift']:
                if metric_value >= best_value:  # Use >= for ties
                    best_value = metric_value
                    best_model = result.get('name', model_key)
                    print(f"[DEBUG] New best model (>= better): {best_model} with {metric_value}")
            # For supervised metrics (accuracy, r2)
            elif metric_value > best_value:
                best_value = metric_value
                best_model = result.get('name', model_key)
                print(f"[DEBUG] New best supervised model: {best_model} with {metric_value}")

    final_result = {
        'value': round(best_value, 4) if isinstance(best_value, (int, float)) else best_value,
        'model': best_model,
        'metric': metric_name
    }
    
    print(f"[DEBUG] Final best metric result: {final_result}")
    return final_result if best_value is not None else {
        'value': None,
        'model': None,
        'metric': 'No valid metrics found'
    }

def load_model_data(model_id):
    """Load model data from the training results - FIXED VERSION"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')
        
        if os.path.exists(projects_dir):
            for project_folder in os.listdir(projects_dir):
                project_path = os.path.join(projects_dir, project_folder)
                if os.path.isdir(project_path):
                    models_dir = os.path.join(project_path, 'models')
                    if os.path.exists(models_dir):
                        model_dir = os.path.join(models_dir, model_id)
                        if os.path.exists(model_dir):
                            results_path = os.path.join(model_dir, 'results.json')
                            if os.path.exists(results_path):
                                with open(results_path, 'r') as f:
                                    data = json.load(f)
                                data['project_id'] = project_folder
                                data['model_id'] = model_id
                                data['model_directory'] = model_dir
                                return data
        return None
    except Exception as e:
        print(f"Error loading model data: {e}")
        return None

def find_model_files(model_id, data=None):
    """Find all files for a given model ID - ENHANCED VERSION"""
    try:
        print(f"[DEBUG] Finding model files for: {model_id}")
        
        # If we have the data with model_directory, use that
        if data and 'model_directory' in data:
            model_dir = data['model_directory']
            print(f"[DEBUG] Using stored model directory: {model_dir}")
            
            if os.path.exists(model_dir):
                files = os.listdir(model_dir)
                print(f"[DEBUG] Files in directory: {files}")
                return model_dir, files
        
        # Otherwise, search for the model
        search_directories = []
        
        # Try new structure first (projects/project_id/models/model_id)
        projects_dir = os.path.join(current_app.root_path, 'projects')
        if os.path.exists(projects_dir):
            for project_folder in os.listdir(projects_dir):
                project_path = os.path.join(projects_dir, project_folder)
                if os.path.isdir(project_path):
                    models_dir = os.path.join(project_path, 'models')
                    if os.path.exists(models_dir):
                        model_dir = os.path.join(models_dir, model_id)
                        if os.path.exists(model_dir):
                            search_directories.append(model_dir)

        # Try old structure (trained_models/model_id)
        trained_models_dir = os.path.join(current_app.root_path, 'trained_models')
        model_dir = os.path.join(trained_models_dir, model_id)
        if os.path.exists(model_dir):
            search_directories.append(model_dir)

        print(f"[DEBUG] Found directories: {search_directories}")

        if search_directories:
            # Use the first found directory
            model_dir = search_directories[0]
            files = os.listdir(model_dir)
            print(f"[DEBUG] Using directory: {model_dir}")
            print(f"[DEBUG] Files found: {files}")
            return model_dir, files
        
        print(f"[DEBUG] No model directories found for {model_id}")
        return None, []
        
    except Exception as e:
        print(f"[ERROR] Error finding model files: {e}")
        import traceback
        traceback.print_exc()
        return None, []

@xai_bp.route('/')
def xai_page():
    """XAI page to view and manage projects and models"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')

        if not os.path.exists(projects_dir):
            # No projects directory at all
            return render_template('xai/xai.html')

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
                        if os.path.isdir(models_dir):
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
                                            problem_type = config.get('problem_type', 'Unknown')
                                            training_mode = config.get('training_mode', 'Unknown')
                                            n_models = len(results) if results else 1
                                            best_metric = get_best_metric(results, problem_type)
                                            
                                            # ADD THIS: Calculate display_score and display_score_label
                                            display_score, display_score_label = get_display_score(results, problem_type)
                                            
                                            project_models.append({
                                                'id': model_id,
                                                'created_at': config.get('created_at', 'Unknown'),
                                                'problem_type': problem_type,
                                                'training_mode': training_mode,
                                                'n_models': n_models,
                                                'best_metric': best_metric,
                                                'display_score': display_score,  # ADD THIS
                                                'display_score_label': display_score_label,  # ADD THIS
                                                'config': config,
                                                'results': results
                                            })
                                        except Exception as e:
                                            print(f"Error loading model {model_id}: {e}")
                                            continue

                        # Sort models by creation date (newest first)
                        project_models.sort(key=lambda x: x['created_at'], reverse=True)

                        project_config['models'] = project_models
                        project_config['model_count'] = len(project_models)
                        project_config['id'] = project_folder  # Add project ID for template
                        projects.append(project_config)
                    except Exception as e:
                        print(f"Error loading project config for {project_folder}: {e}")
                        continue

        # Sort projects by creation date (newest first)
        projects.sort(key=lambda x: x.get('created_at', ''), reverse=True)

        # If there are no projects or all projects have zero models, show no_projects page
        if not projects or all(p.get('model_count', 0) == 0 for p in projects):
            return render_template('xai/xai.html')

        return render_template('xai/xai.html', projects=projects)
    except Exception as e:
        print(f"Error in xai_page: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@xai_bp.route('/api/projects', methods=['POST'])
def create_project_api():
    """API endpoint to create a new project"""
    try:
        data = request.get_json()
        project_name = data.get('name', '').strip()
        project_description = data.get('description', '').strip()

        if not project_name:
            return jsonify({'success': False, 'message': 'Project name is required'})

        # Generate unique project ID
        project_id = f"proj_{int(datetime.now().timestamp())}_{str(uuid.uuid4())[:8]}"

        # Create project directory structure
        projects_dir = os.path.join(current_app.root_path, 'projects')
        os.makedirs(projects_dir, exist_ok=True)

        project_path = os.path.join(projects_dir, project_id)
        os.makedirs(project_path)
        os.makedirs(os.path.join(project_path, 'models'))

        # Create project configuration
        project_config = {
            'id': project_id,
            'name': project_name,
            'description': project_description,
            'created_at': datetime.now().isoformat(),
            'model_count': 0
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

@xai_bp.route('/api/models/<model_id>/analysis')
def get_model_analysis(model_id):
    """Get comprehensive model analysis data"""
    try:
        print(f"[DEBUG] Getting analysis for model: {model_id}")

        data = load_model_data(model_id)
        if not data:
            return jsonify({
                'success': False,
                'message': f'Model {model_id} not found'
            }), 404

        # Add model_id to the data if not present
        if 'model_id' not in data:
            data['model_id'] = model_id

        print(f"[DEBUG] Model data loaded successfully: {data.get('config', {}).get('problem_type', 'unknown')}")

        return jsonify({
            'success': True,
            'data': data
        })

    except Exception as e:
        print(f"[ERROR] Error getting model analysis: {e}")
        return jsonify({
            'success': False,
            'message': f'Error loading model analysis: {str(e)}'
        }), 500

@xai_bp.route('/api/models/<model_id>/download/best', methods=['GET', 'HEAD'])
def download_best_model(model_id):
    """Download the best model as .pkl file - FIXED VERSION"""
    try:
        print(f"[DEBUG] Download best model request for: {model_id}")

        data = load_model_data(model_id)
        if not data:
            print(f"[ERROR] Model {model_id} not found")
            return jsonify({'success': False, 'message': 'Model not found'}), 404

        # Find the model files using the enhanced function
        model_dir, files = find_model_files(model_id, data)
        
        if not model_dir or not files:
            print(f"[ERROR] No model directory or files found for {model_id}")
            return jsonify({'success': False, 'message': 'Model files not found'}), 404

        # Look for ANY .pkl file first (since we know one exists)
        model_file_path = None
        pkl_files = [f for f in files if f.endswith('.pkl')]
        
        print(f"[DEBUG] Available .pkl files: {pkl_files}")
        
        if pkl_files:
            # Use the first .pkl file found
            model_file_path = os.path.join(model_dir, pkl_files[0])
            print(f"[DEBUG] Using .pkl file: {model_file_path}")
        
        if not model_file_path or not os.path.exists(model_file_path):
            print(f"[ERROR] No valid .pkl file found")
            return jsonify({
                'success': False, 
                'message': 'No valid .pkl model files found',
                'available_files': files,
                'pkl_files': pkl_files
            }), 404

        print(f"[DEBUG] Sending file: {model_file_path}")

        # HEAD request handling
        if request.method == "HEAD":
            response = Response()
            response.headers['Content-Disposition'] = f'attachment; filename="{model_id}_best_model.pkl"'
            response.headers['Content-Length'] = str(os.path.getsize(model_file_path))
            response.headers['Content-Type'] = 'application/octet-stream'
            response.headers['Cache-Control'] = 'no-cache'
            return response

        # Send the file
        filename = f'{model_id}_best_model.pkl'
        
        return send_file(
            model_file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/octet-stream'
        )

    except Exception as e:
        print(f"[ERROR] Error downloading best model: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@xai_bp.route('/api/models/<model_id>/download/all', methods=['GET', 'HEAD'])
def download_all_models(model_id):
    """Download all models as a zip file - FIXED VERSION"""
    try:
        print(f"[DEBUG] Download all models request for: {model_id}")

        # Find model directory using enhanced function
        model_dir, files = find_model_files(model_id)
        
        if not model_dir or not files:
            print(f"[ERROR] No model directory found for {model_id}")
            return jsonify({'success': False, 'message': 'Model directory not found'}), 404

        print(f"[DEBUG] Files to be zipped: {files}")

        # HEAD request handling
        if request.method == "HEAD":
            response = Response()
            response.headers['Content-Disposition'] = f'attachment; filename="{model_id}_all_models.zip"'
            response.headers['Content-Type'] = 'application/zip'
            response.headers['Cache-Control'] = 'no-cache'
            return response

        # Create temporary zip file
        temp_dir = tempfile.mkdtemp()
        zip_filename = f'{model_id}_all_models.zip'
        zip_path = os.path.join(temp_dir, zip_filename)

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in files:
                file_path = os.path.join(model_dir, file)
                if os.path.isfile(file_path):  # Only add actual files
                    zipf.write(file_path, file)
                    print(f"[DEBUG] Added to zip: {file}")

        print(f"[DEBUG] Created zip file: {zip_path} with {len(files)} files")

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=zip_filename,
            mimetype='application/zip'
        )

    except Exception as e:
        print(f"[ERROR] Error downloading all models: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@xai_bp.route('/api/models/<model_id>/download/results', methods=['GET', 'HEAD'])
def download_results(model_id):
    """Download model results as JSON - FIXED VERSION"""
    try:
        print(f"[DEBUG] Download results request for: {model_id}")

        data = load_model_data(model_id)
        if not data:
            return jsonify({'success': False, 'message': 'Model not found'}), 404

        # HEAD request handling
        if request.method == "HEAD":
            response = Response()
            response.headers['Content-Disposition'] = f'attachment; filename="{model_id}_results.json"'
            response.headers['Content-Type'] = 'application/json'
            response.headers['Cache-Control'] = 'no-cache'
            return response

        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        json_filename = f'{model_id}_results.json'
        json_path = os.path.join(temp_dir, json_filename)

        # Write JSON data
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)  # default=str handles datetime objects

        print(f"[DEBUG] Created results file: {json_path}")

        return send_file(
            json_path,
            as_attachment=True,
            download_name=json_filename,
            mimetype='application/json'
        )

    except Exception as e:
        print(f"[ERROR] Error downloading results: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

@xai_bp.route('/api/models/<model_id>/debug')
def debug_model_files(model_id):
    """Debug endpoint to check what files exist for a model - ENHANCED VERSION"""
    try:
        debug_info = {
            'model_id': model_id,
            'found_directories': [],
            'found_files': [],
            'search_paths': [],
            'file_details': []
        }

        # Use enhanced find function
        model_dir, files = find_model_files(model_id)
        
        if model_dir:
            debug_info['found_directories'].append(model_dir)
            debug_info['found_files'] = files
            
            for f in files:
                full_path = os.path.join(model_dir, f)
                file_info = {
                    'name': f,
                    'path': full_path,
                    'size': os.path.getsize(full_path) if os.path.isfile(full_path) else 0,
                    'is_file': os.path.isfile(full_path),
                    'extension': os.path.splitext(f)[1]
                }
                debug_info['file_details'].append(file_info)

        # Also check search paths
        projects_dir = os.path.join(current_app.root_path, 'projects')
        trained_models_dir = os.path.join(current_app.root_path, 'trained_models')
        debug_info['search_paths'] = [projects_dir, trained_models_dir]

        # Add summary statistics
        debug_info['summary'] = {
            'total_files': len([f for f in debug_info['file_details'] if f['is_file']]),
            'total_size': sum(f['size'] for f in debug_info['file_details'] if f['is_file']),
            'pkl_files': len([f for f in debug_info['file_details'] if f['extension'] == '.pkl']),
            'json_files': len([f for f in debug_info['file_details'] if f['extension'] == '.json']),
            'directories_found': len(debug_info['found_directories'])
        }

        return jsonify({
            'success': True,
            'debug_info': debug_info
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@xai_bp.route('/api/models/<model_id>/test-download')
def test_download_route(model_id):
    """Simple test route to verify downloads work"""
    try:
        # Create a simple test file
        temp_dir = tempfile.mkdtemp()
        test_filename = f'test_{model_id}.txt'
        test_path = os.path.join(temp_dir, test_filename)

        with open(test_path, 'w') as f:
            f.write(f"Test download file for model: {model_id}\n")
            f.write(f"Generated at: {datetime.now()}\n")
            f.write("If you can see this, downloads are working!\n")

        return send_file(
            test_path,
            as_attachment=True,
            download_name=test_filename,
            mimetype='text/plain'
        )

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@xai_bp.route('/api/projects/<project_id>/compare/<model_a_id>/<model_b_id>')
def compare_models(project_id, model_a_id, model_b_id):
    """Compare two models within a project with enhanced details"""
    try:
        model_a_data = load_model_data(model_a_id)
        model_b_data = load_model_data(model_b_id)

        if not model_a_data or not model_b_data:
            return jsonify({'success': False, 'message': 'One or both models not found'}), 404

        def extract_model_info(data):
            config = data.get('config', {})
            results = data.get('results', {})
            # Get first or best model result
            model_result = next(iter(results.values()), {})
            return {
                'id': data.get('model_id'),
                'algorithm': (
                    config.get('algorithm') or
                    config.get('estimator') or
                    config.get('model_type') or
                    config.get('model_name') or
                    model_result.get('name', 'Unknown')
                ),
                'created_at': config.get('created_at', 'Unknown'),
                'problem_type': config.get('problem_type', 'Unknown'),
                'training_mode': config.get('training_mode', 'Unknown'),
                'metrics': model_result.get('metrics', {}),
                'full_config': config,
            }

        info_a = extract_model_info(model_a_data)
        info_b = extract_model_info(model_b_data)

        # Determine winner based on best metric (support unsupervised)
        def get_score(info):
            metrics = info['metrics'].get('test', {})
            pt = info['problem_type']
            if pt == 'classification':
                return metrics.get('accuracy', 0)
            elif pt == 'regression':
                return metrics.get('r2', 0)
            elif pt == 'clustering':
                return metrics.get('silhouette_score', -1)
            elif pt == 'anomaly':
                return metrics.get('outlier_ratio', 0)
            elif pt == 'dimensionality_reduction':
                return metrics.get('variance_ratio', 0)
            elif pt == 'association':
                return metrics.get('lift', 0) or metrics.get('confidence', 0) or metrics.get('max_lift', 0)
            else:
                return 0

        score_a = get_score(info_a)
        score_b = get_score(info_b)

        winner = 'A' if score_a > score_b else 'B' if score_b > score_a else 'Tie'

        return jsonify({
            'success': True,
            'model_a': info_a,
            'model_b': info_b,
            'winner': winner,
            'score_a': score_a,
            'score_b': score_b,
            'score_difference': abs(score_a - score_b)
        })

    except Exception as e:
        print(f"Error comparing models: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@xai_bp.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project and all its models"""
    try:
        projects_dir = os.path.join(current_app.root_path, 'projects')
        project_path = os.path.join(projects_dir, project_id)

        if not os.path.exists(project_path):
            return jsonify({'success': False, 'message': 'Project not found'}), 404

        # Remove the entire project directory
        import shutil
        shutil.rmtree(project_path)

        return jsonify({'success': True, 'message': 'Project deleted successfully'})

    except Exception as e:
        print(f"Error deleting project: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@xai_bp.route('/api/models/<model_id>', methods=['DELETE'])
def delete_model(model_id):
    """Delete a specific model"""
    try:
        # Find and delete the model using enhanced function
        model_dir, files = find_model_files(model_id)
        
        if not model_dir:
            return jsonify({'success': False, 'message': 'Model not found'}), 404

        # Remove the model directory
        import shutil
        shutil.rmtree(model_dir)

        return jsonify({'success': True, 'message': 'Model deleted successfully'})
    except Exception as e:
        print(f"Error deleting model: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500
    

@xai_bp.route('/analysis/<model_id>')
def analysis_page(model_id):
    data = load_model_data(model_id)
    training_mode = None
    algo_name = None
    if data and 'config' in data:
        config = data['config']
        training_mode = config.get('training_mode', 'Unknown')
        # Try config keys first
        algo_name = (
            config.get('algorithm') or
            config.get('estimator') or
            config.get('model_type') or
            config.get('model_name')
        )
        # If still None, get from results (pick best or first model)
        if not algo_name:
            results = data.get('results', {})
            # Try to get the best model (if available)
            best_model_key = None
            if results:
                # Try to get the model with best metric
                if hasattr(results, 'items'):
                    # If you have a "best" model logic, use it here
                    best_model_key = next(iter(results.keys()))
                if best_model_key and 'name' in results[best_model_key]:
                    algo_name = results[best_model_key]['name']
                else:
                    # fallback: get first model's name
                    for model_key, model_data in results.items():
                        if 'name' in model_data:
                            algo_name = model_data['name']
                            break
        if not algo_name:
            algo_name = 'Unknown'
    return render_template('xai/analyze.html', model_id=model_id, training_mode=training_mode, algo_name=algo_name)


@xai_bp.route('/compare/<project_id>/<model_a_id>/<model_b_id>')
def compare_page(project_id, model_a_id, model_b_id):
    return render_template(
        'xai/compare.html',
        project_id=project_id,
        model_a_id=model_a_id,
        model_b_id=model_b_id
    )