import os
import json
import pickle
import pandas as pd
import numpy as np
import uuid
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error, explained_variance_score

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Skipping XGBoost models.")

import warnings
warnings.filterwarnings('ignore')

# Model configurations
CLASSIFICATION_MODELS = {
    'random_forest': {
        'name': 'Random Forest',
        'class': RandomForestClassifier,
        'description': 'Ensemble method using multiple decision trees',
        'params': {
            'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
            'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50, 'none_allowed': True},
            'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'default': 1, 'min': 1, 'max': 20}
        }
    },
    'logistic_regression': {
        'name': 'Logistic Regression',
        'class': LogisticRegression,
        'description': 'Linear model for binary and multiclass classification',
        'params': {
            'C': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100},
            'max_iter': {'type': 'int', 'default': 1000, 'min': 100, 'max': 5000}
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'class': GradientBoostingClassifier,
        'description': 'Gradient boosting for classification',
        'params': {
            'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
            'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
            'max_depth': {'type': 'int', 'default': 3, 'min': 1, 'max': 10}
        }
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'class': KNeighborsClassifier,
        'description': 'Classification based on k nearest neighbors',
        'params': {
            'n_neighbors': {'type': 'int', 'default': 5, 'min': 1, 'max': 50},
            'weights': {'type': 'select', 'default': 'uniform', 'options': ['uniform', 'distance']}
        }
    }
}

REGRESSION_MODELS = {
    'random_forest': {
        'name': 'Random Forest',
        'class': RandomForestRegressor,
        'description': 'Ensemble method using multiple decision trees',
        'params': {
            'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
            'max_depth': {'type': 'int', 'default': None, 'min': 1, 'max': 50, 'none_allowed': True},
            'min_samples_split': {'type': 'int', 'default': 2, 'min': 2, 'max': 20},
            'min_samples_leaf': {'type': 'int', 'default': 1, 'min': 1, 'max': 20}
        }
    },
    'linear_regression': {
        'name': 'Linear Regression',
        'class': LinearRegression,
        'description': 'Simple linear regression model',
        'params': {}
    },
    'ridge': {
        'name': 'Ridge Regression',
        'class': Ridge,
        'description': 'Linear regression with L2 regularization',
        'params': {
            'alpha': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100}
        }
    },
    'lasso': {
        'name': 'Lasso Regression',
        'class': Lasso,
        'description': 'Linear regression with L1 regularization',
        'params': {
            'alpha': {'type': 'float', 'default': 1.0, 'min': 0.001, 'max': 100},
            'max_iter': {'type': 'int', 'default': 1000, 'min': 100, 'max': 5000}
        }
    },
    'gradient_boosting': {
        'name': 'Gradient Boosting',
        'class': GradientBoostingRegressor,
        'description': 'Gradient boosting for regression',
        'params': {
            'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
            'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
            'max_depth': {'type': 'int', 'default': 3, 'min': 1, 'max': 10}
        }
    },
    'knn': {
        'name': 'K-Nearest Neighbors',
        'class': KNeighborsRegressor,
        'description': 'Regression based on k nearest neighbors',
        'params': {
            'n_neighbors': {'type': 'int', 'default': 5, 'min': 1, 'max': 50},
            'weights': {'type': 'select', 'default': 'uniform', 'options': ['uniform', 'distance']}
        }
    }
}

# Add XGBoost models if available
if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS['xgboost'] = {
        'name': 'XGBoost Classifier',
        'class': XGBClassifier,
        'description': 'Extreme Gradient Boosting for classification',
        'params': {
            'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
            'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
            'max_depth': {'type': 'int', 'default': 3, 'min': 1, 'max': 10}
        }
    }
    REGRESSION_MODELS['xgboost'] = {
        'name': 'XGBoost Regressor',
        'class': XGBRegressor,
        'description': 'Extreme Gradient Boosting for regression',
        'params': {
            'n_estimators': {'type': 'int', 'default': 100, 'min': 10, 'max': 500},
            'learning_rate': {'type': 'float', 'default': 0.1, 'min': 0.01, 'max': 1.0},
            'max_depth': {'type': 'int', 'default': 3, 'min': 1, 'max': 10}
        }
    }

def get_model_configs(problem_type):
    if problem_type == 'classification':
        return CLASSIFICATION_MODELS
    elif problem_type == 'regression':
        return REGRESSION_MODELS
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")

def detect_problem_type(target_series):
    if not isinstance(target_series, pd.Series):
        raise ValueError("target_series must be a pandas Series")
    target_clean = target_series.dropna()
    if len(target_clean) == 0:
        return 'classification'
    if pd.api.types.is_numeric_dtype(target_clean):
        unique_ratio = len(target_clean.unique()) / len(target_clean)
        unique_count = len(target_clean.unique())
        if unique_count <= 10 or unique_ratio < 0.05:
            return 'classification'
        else:
            return 'regression'
    else:
        return 'classification'


def train_models(df, target_column, feature_columns, problem_type, model_name, 
                 selected_models, train_test_config, cv_config, model_params, model_dir):
    """Comprehensive model training pipeline -- FIXED: only JSON-serializable objects in results."""
    model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    model_dir = os.path.join(model_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)
   

    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    try:
        results = {}
        model_objects = {}

        # Prepare data
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].fillna(X[col].mean())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'missing')
        if pd.api.types.is_numeric_dtype(y):
            y = y.fillna(y.mean())
        else:
            y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 'missing')
        
        # Convert all non-numeric features to strings
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = X[col].astype(str)
        # OneHotEncode
        if len(categorical_cols) > 0:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
            X_categorical = ohe.fit_transform(X[categorical_cols])
            feature_names = ohe.get_feature_names_out(categorical_cols)
        else:
            X_categorical = np.array([]).reshape(len(X), 0)
            feature_names = []
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols].values if len(numeric_cols) > 0 else np.zeros((len(X), 0))
        X_processed = np.hstack([X_numeric, X_categorical]) if len(numeric_cols) > 0 else X_categorical
        all_feature_names = list(numeric_cols) + list(feature_names)
        # Encode target for classification if needed
        label_encoder = None
        if problem_type == 'classification' and not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
        # Train-test split
        test_size = train_test_config.get('test_size', 0.2)
        shuffle = train_test_config.get('shuffle', True)
        random_state = train_test_config.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=test_size, shuffle=shuffle, random_state=random_state,
            stratify=y if problem_type == 'classification' and len(set(y)) > 1 else None
        )
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model_configs = get_model_configs(problem_type)
        if not selected_models:
            selected_models = list(model_configs.keys())

        for model_key in selected_models:
            if model_key not in model_configs:
                continue
            model_config = model_configs[model_key]
            params = model_params.get(model_key, {})
            final_params = {}
            for param_name, param_config in model_config['params'].items():
                if param_name in params:
                    final_params[param_name] = params[param_name]
                else:
                    final_params[param_name] = param_config['default']
            for key, value in final_params.items():
                if value == 'None':
                    final_params[key] = None
            try:
                model = model_config['class'](**final_params)
                use_scaled = model_key in ['logistic_regression', 'knn']
                X_train_model = X_train_scaled if use_scaled else X_train
                X_test_model = X_test_scaled if use_scaled else X_test
                model.fit(X_train_model, y_train)
                y_pred_train = model.predict(X_train_model)
                y_pred_test = model.predict(X_test_model)
                if problem_type == 'classification':
                    metrics = calculate_classification_metrics(y_train, y_test, y_pred_train, y_pred_test)
                    if hasattr(model, 'predict_proba'):
                        y_proba = model.predict_proba(X_test_model)
                        metrics['y_proba'] = y_proba.tolist()
                        # Add y_true, y_pred for backend metrics
                        metrics['test']['y_true'] = [float(x) for x in y_test]
                        metrics['test']['y_pred'] = [float(x) for x in y_pred_test]
                        # For binary: use probability for class 1; for multiclass: use max probability
                        if y_proba.shape[1] == 2:
                            metrics['test']['y_score'] = [float(row[1]) for row in y_proba]
                        else:
                            metrics['test']['y_score'] = [float(np.max(row)) for row in y_proba]
                else:
                    metrics = calculate_regression_metrics(y_train, y_test, y_pred_train, y_pred_test)
                    # Add residuals and predictions for plotting
                    metrics['test']['residuals'] = (y_test - y_pred_test).tolist()
                    metrics['test']['y_pred'] = y_pred_test.tolist()
                    metrics['test']['y_true'] = y_test.tolist()
                if cv_config.get('enabled', False):
                    cv_folds = cv_config.get('folds', 5)
                    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state) if problem_type == 'classification' else KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                    scoring = 'accuracy' if problem_type == 'classification' else 'r2'
                    cv_scores = cross_val_score(model, X_train_model, y_train, cv=cv_strategy, scoring=scoring)
                    metrics['cv_scores'] = {
                        'scores': cv_scores.tolist(),
                        'mean': float(cv_scores.mean()),
                        'std': float(cv_scores.std())
                    }
                feature_importance = None
                if hasattr(model, 'feature_importances_'):
                    importance_data = list(zip(all_feature_names, model.feature_importances_))
                    importance_data.sort(key=lambda x: x[1], reverse=True)
                    feature_importance = [(name, float(importance)) for name, importance in importance_data[:20]]
                elif hasattr(model, 'coef_'):
                    coef = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
                    importance_data = list(zip(all_feature_names, np.abs(coef)))
                    importance_data.sort(key=lambda x: x[1], reverse=True)
                    feature_importance = [(name, float(importance)) for name, importance in importance_data[:20]]
                predictions_data = {
                    'y_test': [float(x) for x in y_test],
                    'y_pred': [float(x) for x in y_pred_test]
                }
                # ---- FIX: DO NOT PUT model, scaler, ohe, label_encoder, etc in results ----
                results[model_key] = {
                    'name': model_config['name'],
                    'metrics': convert_numpy_types(metrics),
                    'feature_importance': feature_importance,
                    'predictions': predictions_data,
                    'parameters': convert_numpy_types(final_params)
                }
                # Only put non-serializable objects in model_objects, never in results!
                model_objects[model_key] = {
                    'model': model,
                    'scaler': scaler if use_scaled else None,
                    'label_encoder': label_encoder,
                    'feature_names': all_feature_names,
                    'ohe': ohe if len(categorical_cols) > 0 else None,
                    'categorical_cols': list(categorical_cols),
                    'numeric_cols': list(numeric_cols)
                }
            except Exception as e:
                print(f"Error training model {model_key}: {str(e)}")
                results[model_key] = {
                    'name': model_config['name'],
                    'error': str(e),
                    'parameters': final_params
                }
        # Save models and results in model_dir
        os.makedirs(model_dir, exist_ok=True)
        results_path = os.path.join(model_dir, "results.json")
        # ---- FIX: results dict contains only JSON-serializable objects ----
        with open(results_path, "w") as f:
            json.dump({
                'results': results,
                'config': {
                    'target_column': target_column,
                    'feature_columns': feature_columns,
                    'problem_type': problem_type,
                    'train_test_config': train_test_config,
                    'cv_config': cv_config,
                    'created_at': datetime.now().isoformat(),
                    'training_mode': 'self-train'
                }
            }, f, indent=2)
        for model_key, model_obj in model_objects.items():
            model_path = os.path.join(model_dir, f"{model_key}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model_obj, f)
        return {
            'success': True,
            'model_id': model_id,
            'results': results,
            'config': {
                'problem_type': problem_type,
                'n_features': len(feature_columns),
                'n_samples': len(df),
                'train_size': len(X_train),
                'test_size': len(X_test)
            },
            'models': model_objects  # Only for backend use, not for JSON!
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def calculate_classification_metrics(y_train, y_test, y_pred_train, y_pred_test):
    metrics = {
        'train': {
            'accuracy': float(accuracy_score(y_train, y_pred_train)),
            'precision': float(precision_score(y_train, y_pred_train, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_train, y_pred_train, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_train, y_pred_train, average='weighted', zero_division=0))
        },
        'test': {
            'accuracy': float(accuracy_score(y_test, y_pred_test)),
            'precision': float(precision_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred_test, average='weighted', zero_division=0)),
            'classification_report': classification_report(y_test, y_pred_test, output_dict=True)
        },
        'confusion_matrix': confusion_matrix(y_test, y_pred_test).tolist()
    }
    return metrics


def calculate_regression_metrics(y_train, y_test, y_pred_train, y_pred_test):
    n = len(y_test)
    p = 1  # You can set this to X_test.shape[1] if you want to pass it in
    r2 = r2_score(y_test, y_pred_test)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    metrics = {
        'train': {
            'mse': float(mean_squared_error(y_train, y_pred_train)),
            'rmse': float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
            'mae': float(mean_absolute_error(y_train, y_pred_train)),
            'r2': float(r2_score(y_train, y_pred_train)),
            'mape': float(mean_absolute_percentage_error(y_train, y_pred_train)),
            'median_ae': float(median_absolute_error(y_train, y_pred_train)),
            'explained_variance': float(explained_variance_score(y_train, y_pred_train)),
            'adjusted_r2': float(adj_r2)
        },
        'test': {
            'mse': float(mean_squared_error(y_test, y_pred_test)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
            'mae': float(mean_absolute_error(y_test, y_pred_test)),
            'r2': float(r2),
            'mape': float(mean_absolute_percentage_error(y_test, y_pred_test)),
            'median_ae': float(median_absolute_error(y_test, y_pred_test)),
            'explained_variance': float(explained_variance_score(y_test, y_pred_test)),
            'adjusted_r2': float(adj_r2)
        }
    }
    return metrics

def is_target_candidate(series):
    """
    Make ALL columns available as target candidates (except completely empty ones)
    """
    # Only exclude completely empty columns or columns with no data
    if series.empty or series.isnull().all():
        return False
    
    # Remove null values for analysis
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return False
    
    # Include ALL columns that have at least some data
    # Even if they have only 1 unique value, they could still be targets
    # (though they might not be useful for ML, let the user decide)
    return True

def generate_enhanced_preview(df, columns_info):
    html = '<div class="enhanced-preview-container">'
    html += '<table class="preview-table">'
    html += '<thead><tr>'
    for col_info in columns_info:
        dtype_class = 'numeric' if col_info['is_numeric'] else 'categorical'
        target_indicator = '🎯' if col_info['is_target_candidate'] else ''
        html += f'<th class="col-header {dtype_class}">'
        html += f'<div class="col-name">{col_info["name"]} {target_indicator}</div>'
        html += f'<div class="col-type">{col_info["dtype"]}</div>'
        if col_info['missing_count'] > 0:
            html += f'<div class="col-missing">Missing: {col_info["missing_count"]}</div>'
        html += '</th>'
    html += '</tr></thead>'
    html += '<tbody>'
    for _, row in df.head(10).iterrows():
        html += '<tr>'
        for col in df.columns:
            value = row[col]
            if pd.isna(value):
                html += '<td class="missing-value">NULL</td>'
            else:
                html += f'<td>{str(value)}</td>'
        html += '</tr>'
    html += '</tbody>'
    html += '</table></div>'
    return html