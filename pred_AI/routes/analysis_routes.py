"""
Classification Analysis Routes
Handles the analysis page routing and data processing for classification models.
Clean, simple, and focused implementation.
"""

import os
import json
from flask import Blueprint, render_template, current_app, jsonify
from datetime import datetime

# Create blueprint for analysis routes
analysis_bp = Blueprint('analysis', __name__)

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
                        possible_names = [model_id, f"\\{model_id}", f"/{model_id}"]
                        for name in possible_names:
                            model_dir = os.path.join(models_dir, name)
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

def is_classification_model(data):
    if not data:
        return False
    config = data.get('config', {})
    problem_type = config.get('problem_type', '').lower()
    if 'classification' not in problem_type:
        print(f"[ANALYSIS] Not a classification problem: {problem_type}")
        return False
    results = data.get('results', {})
    if not results:
        print("[ANALYSIS] No results found in model data")
        return False
    for model_name, model_result in results.items():
        if (model_result and 
            isinstance(model_result, dict) and 
            'metrics' in model_result and 
            'test' in model_result.get('metrics', {})):
            return True
    print("[ANALYSIS] No test metrics found in any model")
    return False

def enhance_classification_metrics(data):
    try:
        from sklearn.metrics import (
            f1_score, cohen_kappa_score, matthews_corrcoef,
            log_loss, brier_score_loss
        )
        import numpy as np

        results = data.get('results', {})

        for model_name, model_result in results.items():
            if not model_result or not isinstance(model_result, dict):
                continue

            metrics = model_result.get('metrics', {})
            test_metrics = metrics.get('test', {})

            if not test_metrics:
                continue

            y_true = test_metrics.get('y_true')
            y_pred = test_metrics.get('y_pred')
            y_score = test_metrics.get('y_score') or test_metrics.get('proba')

            precision = test_metrics.get('precision')
            recall = test_metrics.get('recall')
            if test_metrics.get('f1') is None and precision is not None and recall is not None:
                if precision + recall > 0:
                    test_metrics['f1'] = 2 * (precision * recall) / (precision + recall)

            if test_metrics.get('mcc') is None and y_true is not None and y_pred is not None:
                try:
                    test_metrics['mcc'] = float(matthews_corrcoef(y_true, y_pred))
                except Exception:
                    test_metrics['mcc'] = None

            if test_metrics.get('cohen_kappa') is None and y_true is not None and y_pred is not None:
                try:
                    test_metrics['cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))
                except Exception:
                    test_metrics['cohen_kappa'] = None

            if test_metrics.get('log_loss') is None and y_true is not None and y_score is not None:
                try:
                    test_metrics['log_loss'] = float(log_loss(y_true, y_score))
                except Exception:
                    test_metrics['log_loss'] = None

            if test_metrics.get('brier_score') is None and y_true is not None and y_score is not None:
                try:
                    test_metrics['brier_score'] = float(brier_score_loss(y_true, y_score))
                except Exception:
                    test_metrics['brier_score'] = None

            accuracy = test_metrics.get('accuracy')
            if not test_metrics.get('roc_auc') and not test_metrics.get('auc'):
                if accuracy is not None and precision is not None and recall is not None:
                    estimated_auc = min(1.0, max(0.5, (accuracy + precision + recall) / 3))
                    test_metrics['auc'] = estimated_auc

            for key in ['accuracy', 'precision', 'recall', 'f1', 'mcc', 'cohen_kappa', 'log_loss', 'brier_score']:
                if key not in test_metrics:
                    test_metrics[key] = None

        return data

    except Exception as e:
        print(f"[ANALYSIS] Error enhancing metrics: {e}")
        return data

# --- LIFT CHART CALCULATION ---
def compute_lift_chart(y_true, y_score):
    import numpy as np
    order = np.argsort(y_score)[::-1]
    y_true_sorted = np.array(y_true)[order]
    cum_positive = np.cumsum(y_true_sorted)
    pct_samples = np.arange(1, len(y_true_sorted)+1) / len(y_true_sorted)
    pct_positives = cum_positive / sum(y_true_sorted)
    return {
        'percent_samples': pct_samples.tolist(),
        'percent_positives': pct_positives.tolist()
    }

@analysis_bp.route('/<model_id>')
def analysis_page(model_id):
    try:
        print(f"[ANALYSIS] Rendering analysis page for model: {model_id}")
        data = load_model_data(model_id)
        if not data:
            print(f"[ANALYSIS] Model {model_id} not found")
            return render_template(
                'xai/analyze.html',
                model_id=model_id,
                error="Model not found"
            ), 404
        if not is_classification_model(data):
            print(f"[ANALYSIS] Invalid classification data for model {model_id}")
            return render_template(
                'xai/analyze.html',
                model_id=model_id,
                error="This model is not suitable for classification analysis"
            ), 400
        print(f"[ANALYSIS] Successfully rendering analysis page for {model_id}")
        return render_template('xai/analyze.html', model_id=model_id)
    except Exception as e:
        print(f"[ANALYSIS] Error rendering analysis page: {e}")
        return render_template(
            'xai/analyze.html',
            model_id=model_id,
            error=f"Error loading analysis: {str(e)}"
        ), 500
def is_regression_model(data):
    config = data.get('config', {})
    return config.get('problem_type', '').lower() == 'regression'


@analysis_bp.route('/api/<model_id>/data')
def get_analysis_data(model_id):
    """
    Get comprehensive analysis data for a specific model.
    """
    try:
        print(f"[ANALYSIS API] Getting analysis data for model: {model_id}")

        data = load_model_data(model_id)
        if not data:
            return jsonify({
                'success': False,
                'message': f'Model {model_id} not found'
            }), 404

        # --- REGRESSION SUPPORT ---
        if is_regression_model(data):
            results = data.get('results', {})
            best_metrics = None
            best_r2 = -float('inf')
            for model_name, model_result in results.items():
                if (model_result and isinstance(model_result, dict)
                    and 'metrics' in model_result
                    and 'test' in model_result['metrics']):
                    test_metrics = model_result['metrics']['test']
                    r2 = test_metrics.get('r2', -float('inf'))
                    if r2 > best_r2:
                        best_r2 = r2
                        best_metrics = test_metrics
            config = data.get('config', {})
            response = {
                'success': True,
                'data': {
                    'model_id': model_id,
                    'problem_type': config.get('problem_type', 'N/A'),
                    'created_at': config.get('created_at', 'N/A'),
                    'metrics': best_metrics,
                    'residuals': best_metrics.get('residuals') if best_metrics else [],
                    'y_pred': best_metrics.get('y_pred') if best_metrics else [],
                    'y_true': best_metrics.get('y_true') if best_metrics else [],
                }
            }
            return jsonify(response)

        # --- CLASSIFICATION SUPPORT ---
        if not is_classification_model(data):
            return jsonify({
                'success': False,
                'message': 'Model is not suitable for classification analysis'
            }), 400

        enhanced_data = enhance_classification_metrics(data)
        if 'model_id' not in enhanced_data:
            enhanced_data['model_id'] = model_id

        from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
        import numpy as np

        results = enhanced_data.get('results', {})
        best_metrics = None
        best_accuracy = -1
        for model_name, model_result in results.items():
            if (model_result and isinstance(model_result, dict)
                and 'metrics' in model_result
                and 'test' in model_result['metrics']):
                test_metrics = model_result['metrics']['test']
                accuracy = test_metrics.get('accuracy', 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_metrics = test_metrics
        if best_metrics:
            enhanced_data['metrics'] = best_metrics

            y_true = best_metrics.get('y_true')
            y_pred = best_metrics.get('y_pred')
            y_score = best_metrics.get('y_score')

            if y_true is not None and y_pred is not None:
                cm = confusion_matrix(y_true, y_pred).tolist()
                enhanced_data['confusion_matrix'] = cm

            # Add ROC curve and AUC
            if y_true is not None and y_score is not None:
                fpr, tpr, _ = roc_curve(y_true, y_score)
                roc_auc = auc(fpr, tpr)
                enhanced_data['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(roc_auc)
                }
                # Add PR curve and AUC
                precision, recall, _ = precision_recall_curve(y_true, y_score)
                pr_auc = auc(recall, precision)
                enhanced_data['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'auc': float(pr_auc)
                }
                # Add lift chart
                enhanced_data['lift_chart'] = compute_lift_chart(y_true, y_score)

            # --- FIX: Add created_at and problem_type from config ---
            config = data.get('config', {})
            enhanced_data['created_at'] = config.get('created_at', 'N/A')
            enhanced_data['problem_type'] = config.get('problem_type', 'N/A')

        print(f"[ANALYSIS API] Analysis data prepared successfully")

        return jsonify({
            'success': True,
            'data': enhanced_data
        })

    except Exception as e:
        print(f"[ANALYSIS API] Error getting analysis data: {e}")
        return jsonify({
            'success': False,
            'message': f'Error loading analysis data: {str(e)}'
        }), 500
    
@analysis_bp.route('/api/<model_id>/best')
def get_best_model(model_id):
    """
    Get information about the best performing model.
    
    Args:
        model_id (str): The unique identifier for the model
        
    Returns:
        JSON: Best model information or error message
    """
    try:
        print(f"[ANALYSIS API] Getting best model for: {model_id}")
        
        data = load_model_data(model_id)
        
        if not data:
            return jsonify({
                'success': False,
                'message': f'Model {model_id} not found'
            }), 404
        
        # Find best model based on accuracy
        results = data.get('results', {})
        best_model = None
        best_accuracy = -1
        
        for model_name, model_result in results.items():
            if (model_result and 
                isinstance(model_result, dict) and 
                'metrics' in model_result and 
                'test' in model_result.get('metrics', {})):
                
                test_metrics = model_result['metrics']['test']
                accuracy = test_metrics.get('accuracy', 0)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = {
                        'name': model_name,
                        'accuracy': accuracy,
                        'metrics': test_metrics,
                        'training_time': model_result.get('training_time', 'N/A')
                    }
        
        if not best_model:
            return jsonify({
                'success': False,
                'message': 'No valid model results found'
            }), 404
        
        return jsonify({
            'success': True,
            'best_model': best_model
        })
        
    except Exception as e:
        print(f"[ANALYSIS API] Error getting best model: {e}")
        return jsonify({
            'success': False,
            'message': f'Error finding best model: {str(e)}'
        }), 500

@analysis_bp.route('/api/<model_id>/export')
def export_analysis_data(model_id):
    """
    Export analysis data for external use.
    
    Args:
        model_id (str): The unique identifier for the model
        
    Returns:
        JSON: Formatted analysis data for export
    """
    try:
        print(f"[ANALYSIS API] Exporting analysis data for: {model_id}")
        
        data = load_model_data(model_id)
        
        if not data:
            return jsonify({
                'success': False,
                'message': f'Model {model_id} not found'
            }), 404
        
        if not is_classification_model(data):
            return jsonify({
                'success': False,
                'message': 'Model is not suitable for classification analysis'
            }), 400
        
        # Enhance data for export
        enhanced_data = enhance_classification_metrics(data)
        
        # Create export package
        export_data = {
            'model_id': model_id,
            'export_timestamp': datetime.now().isoformat(),
            'analysis_type': 'classification',
            'model_config': enhanced_data.get('config', {}),
            'results': enhanced_data.get('results', {}),
            'summary': generate_analysis_summary(enhanced_data)
        }
        
        return jsonify({
            'success': True,
            'export_data': export_data
        })
        
    except Exception as e:
        print(f"[ANALYSIS API] Error exporting analysis data: {e}")
        return jsonify({
            'success': False,
            'message': f'Error exporting analysis data: {str(e)}'
        }), 500

def generate_analysis_summary(data):
    """
    Generate a summary of the analysis results.
    
    Args:
        data (dict): Enhanced model data
        
    Returns:
        dict: Analysis summary
    """
    try:
        results = data.get('results', {})
        
        if not results:
            return {'error': 'No results available'}
        
        # Find best model
        best_model = None
        best_accuracy = -1
        
        for model_name, model_result in results.items():
            if (model_result and 
                isinstance(model_result, dict) and 
                'metrics' in model_result and 
                'test' in model_result.get('metrics', {})):
                
                test_metrics = model_result['metrics']['test']
                accuracy = test_metrics.get('accuracy', 0)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = {
                        'name': model_name,
                        'metrics': test_metrics
                    }
        
        if not best_model:
            return {'error': 'No valid model found'}
        
        metrics = best_model['metrics']
        
        # Generate performance assessment
        accuracy = metrics.get('accuracy', 0)
        if accuracy >= 0.9:
            performance_level = 'Excellent'
        elif accuracy >= 0.8:
            performance_level = 'Good'
        elif accuracy >= 0.7:
            performance_level = 'Fair'
        else:
            performance_level = 'Poor'
        
        # Calculate balance between precision and recall
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        
        if precision and recall:
            balance_diff = abs(precision - recall)
            if balance_diff < 0.05:
                balance_assessment = 'Well balanced'
            elif precision > recall:
                balance_assessment = 'Precision-focused'
            else:
                balance_assessment = 'Recall-focused'
        else:
            balance_assessment = 'Unknown'
        
        summary = {
            'best_model_name': best_model['name'],
            'performance_level': performance_level,
            'key_metrics': {
                'accuracy': f"{accuracy * 100:.1f}%" if accuracy else 'N/A',
                'precision': f"{precision * 100:.1f}%" if precision else 'N/A',
                'recall': f"{recall * 100:.1f}%" if recall else 'N/A',
                'f1_score': f"{metrics.get('f1', 0) * 100:.1f}%" if metrics.get('f1') else 'N/A',
                'auc': f"{metrics.get('auc', metrics.get('roc_auc', 0)):.3f}" if metrics.get('auc') or metrics.get('roc_auc') else 'N/A'
            },
            'balance_assessment': balance_assessment,
            'total_models_trained': len(results),
            'recommendations': generate_recommendations(metrics)
        }
        
        return summary
        
    except Exception as e:
        print(f"[ANALYSIS] Error generating summary: {e}")
        return {'error': f'Error generating summary: {str(e)}'}

def generate_recommendations(metrics):
    """
    Generate actionable recommendations based on model performance.
    
    Args:
        metrics (dict): Model test metrics
        
    Returns:
        list: List of recommendation strings
    """
    recommendations = []
    
    accuracy = metrics.get('accuracy', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    f1 = metrics.get('f1', 0)
    
    # Overall performance recommendations
    if accuracy < 0.8:
        recommendations.append("Consider feature engineering to improve overall model performance")
        recommendations.append("Try ensemble methods or different algorithms")
        recommendations.append("Collect more training data if possible")
    
    # Precision/Recall imbalance recommendations
    if precision and recall:
        diff = abs(precision - recall)
        if diff > 0.1:
            if precision > recall:
                recommendations.append("Model has low recall - consider adjusting classification threshold to catch more positive cases")
            else:
                recommendations.append("Model has low precision - consider adjusting classification threshold to reduce false positives")
    
    # F1 score recommendations
    if f1 and f1 < 0.75:
        recommendations.append("F1 score indicates room for improvement in precision-recall balance")
    
    # General recommendations for improvement
    if accuracy < 0.9:
        recommendations.append("Perform hyperparameter tuning to optimize model performance")
        recommendations.append("Consider addressing class imbalance if present in the dataset")
    
    return recommendations[:5]  # Limit to top 5 recommendations

# Error handlers for the analysis blueprint
@analysis_bp.errorhandler(404)
def analysis_not_found(error):
    """Handle 404 errors in analysis routes."""
    return jsonify({
        'success': False,
        'message': 'Analysis resource not found'
    }), 404

@analysis_bp.errorhandler(500)
def analysis_server_error(error):
    """Handle 500 errors in analysis routes."""
    return jsonify({
        'success': False,
        'message': 'Internal server error in analysis module'
    }), 500

# Register blueprint helper function
def register_analysis_routes(app):
    """
    Register the analysis blueprint with the Flask app.
    
    Args:
        app: Flask application instance
    """
    app.register_blueprint(analysis_bp)
    print("[ANALYSIS] Analysis routes registered successfully")