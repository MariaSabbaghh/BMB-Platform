// Global variables
let filename = null;
let selectedTrainingMode = null;
let datasetInfo = null;
let recommendedTasks = {};
let selectedTarget = null;
let selectedTaskType = null;
let selectedMethod = null;
let selectedUnsupervisedType = null;
let selectedUnsupervisedMethod = null;

// Project-related variables
let availableProjects = {};
let selectedProject = null;

// Model options mapping
const methodOptions = {
    regression: [
        { value: "linear_regression", label: "Linear Regression" },
        { value: "ridge", label: "Ridge Regression" },
        { value: "lasso", label: "Lasso Regression" },
        { value: "random_forest", label: "Random Forest" },
        { value: "xgboost", label: "XGBoost" }
    ],
    classification: [
        { value: "logistic_regression", label: "Logistic Regression" },
        { value: "random_forest", label: "Random Forest" },
        { value: "xgboost", label: "XGBoost" },
        { value: "knn", label: "K-Nearest Neighbors" }
    ],
    clustering: [
        { value: "kmeans", label: "K-Means" },
        { value: "dbscan", label: "DBSCAN" },
        { value: "agglomerative", label: "Agglomerative Clustering" }
    ],
    dimensionality_reduction: [
        { value: "pca", label: "PCA" },
        { value: "tsne", label: "t-SNE" }
    ],
    anomaly: [
        { value: "isolation_forest", label: "Isolation Forest" },
        { value: "one_class_svm", label: "One Class SVM" }
    ],
    association: [
        { value: "apriori", label: "Apriori Algorithm" }
    ]
};

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', function() {
    loadAvailableProjects();
    const cvSelect = document.getElementById('cv-type-select');
    if (cvSelect) {
        cvSelect.addEventListener('change', function() {
            updateCVOptionsPanel();
        });
    }
});

function setFilename(templateFilename) {
    filename = templateFilename;
}

// ===================== PROJECT MANAGEMENT FUNCTIONS =====================

function loadAvailableProjects() {
    fetch('/train/projects')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                availableProjects = data.projects;
            }
        });
}

// ===================== PROJECT SELECTION FOR TRAINING =====================

function showProjectSelection(callback) {
    const existingModal = document.getElementById('project-selection-modal');
    if (existingModal) existingModal.remove();

    let html = `
        <div class="modal-overlay" id="project-selection-modal" style="display: flex !important; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.8); justify-content: center; align-items: center; z-index: 10000;">
            <div class="modal-content project-modal" style="background: white; border-radius: 8px; padding: 20px; max-width: 700px; max-height: 80vh; overflow-y: auto; width: 90%; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
                <div class="modal-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 2px solid #eee;">
                    <h3 style="margin: 0; color: #333; font-size: 1.25em;"><i class="fas fa-folder"></i> Select Project</h3>
                    <button class="close-btn" onclick="closeProjectModal()" style="background: none; border: none; font-size: 24px; cursor: pointer; color: #666; padding: 5px; width: 35px; height: 35px; display: flex; align-items: center; justify-content: center; border-radius: 50%;">&times;</button>
                </div>
                <div class="modal-body">
                    <div id="projects-list-container">
                        <h4>Select Existing Project</h4>
                        <div id="projects-list" style="max-height: 250px; overflow-y: auto; border: 1px solid #ddd; border-radius: 6px; margin-bottom: 20px; background: white;">
                            <p style="padding: 20px; text-align: center; color: #666;">Loading projects...</p>
                        </div>
                    </div>
                    <div style="margin-top: 20px;">
                        <h4>Or Create New Project</h4>
                        <div style="border: 1px solid #ddd; padding: 15px; border-radius: 6px; background: #f9f9f9;">
                            <input type="text" id="new-project-name" placeholder="Project name" style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box;">
                            <textarea id="new-project-description" placeholder="Description (optional)" style="width: 100%; padding: 8px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; min-height: 60px; box-sizing: border-box; resize: vertical;"></textarea>
                            <button onclick="createNewProjectForTraining(window._pendingProjectCallback)" style="background: #4CAF50; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Create & Select</button>
                        </div>
                    </div>
                </div>
                <div class="modal-footer" style="display: flex; justify-content: flex-end; gap: 12px; margin-top: 25px; padding-top: 20px; border-top: 2px solid #eee;">
                    <button onclick="closeProjectModal()" style="background: #f44336; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-right: 10px;">Cancel</button>
                </div>
            </div>
        </div>
    `;
    document.body.insertAdjacentHTML('beforeend', html);
    renderProjectsListForTraining();
    window._pendingProjectCallback = callback;
}

function renderProjectsListForTraining() {
    const container = document.getElementById('projects-list');
    if (!container) return;
    
    if (!availableProjects || Object.keys(availableProjects).length === 0) {
        container.innerHTML = '<p style="padding: 20px; text-align: center; color: #666;">No projects available. Create a new one below.</p>';
        return;
    }
    
    let html = '';
    Object.entries(availableProjects).forEach(([id, project]) => {
        html += `
            <div class="project-item" data-project-id="${id}" onclick="selectProjectForTraining('${id}')" style="padding: 12px; border-bottom: 1px solid #eee; cursor: pointer; transition: background 0.2s; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <div style="font-weight: bold; color: #333;">${project.name}</div>
                    <div style="color: #666; font-size: 0.9em;">${project.description || 'No description'}</div>
                    <div style="color: #999; font-size: 0.8em; margin-top: 4px;">
                        ${project.model_count || 0} models
                    </div>
                </div>
                <div class="selected-indicator" style="color: #4CAF50; opacity: 0; transition: opacity 0.2s;">✓</div>
            </div>
        `;
    });
    
    container.innerHTML = html;
    
    // Add hover effects
    container.querySelectorAll('.project-item').forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.backgroundColor = '#f5f5f5';
        });
        item.addEventListener('mouseleave', function() {
            if (!this.classList.contains('selected')) {
                this.style.backgroundColor = 'transparent';
            }
        });
    });
}

function selectProjectForTraining(projectId) {
    console.log('[DEBUG] Selecting project for training:', projectId);
    
    // Remove previous selection
    document.querySelectorAll('.project-item').forEach(item => {
        item.classList.remove('selected');
        item.style.backgroundColor = 'transparent';
        const indicator = item.querySelector('.selected-indicator');
        if (indicator) {
            indicator.style.opacity = '0';
        }
    });

    // Select new project
    const projectItem = document.querySelector(`.project-item[data-project-id="${projectId}"]`);
    if (projectItem) {
        projectItem.classList.add('selected');
        projectItem.style.backgroundColor = '#e3f2fd';
        const indicator = projectItem.querySelector('.selected-indicator');
        if (indicator) {
            indicator.style.opacity = '1';
        }
        
        selectedProject = projectId;
        console.log('[DEBUG] Project selected:', selectedProject);
    }
    
    closeProjectModal();
    if (window._pendingProjectCallback) {
        window._pendingProjectCallback(projectId);
        window._pendingProjectCallback = null;
    }
}

function createNewProjectForTraining(callback) {
    const nameInput = document.getElementById('new-project-name');
    const descriptionInput = document.getElementById('new-project-description');
    
    const projectName = nameInput.value.trim();
    const projectDescription = descriptionInput.value.trim();
    
    if (!projectName) {
        alert('Please enter a project name');
        nameInput.focus();
        return;
    }
    
    console.log('[DEBUG] Creating new project:', projectName);
    
    fetch('/train/projects', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            name: projectName,
            description: projectDescription
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            console.log('[DEBUG] Project created successfully:', data.project_id);
            loadAvailableProjects();
            selectedProject = data.project_id;
            closeProjectModal();
            if (callback) callback(data.project_id);
            else if (window._pendingProjectCallback) window._pendingProjectCallback(data.project_id);
        } else {
            alert('Failed to create project: ' + data.message);
        }
    })
    .catch(error => {
        console.error('[ERROR] Error creating project:', error);
        alert('Error creating project: ' + error.message);
    });
}

function closeProjectModal() {
    console.log('[DEBUG] Closing project modal');
    const modal = document.getElementById('project-selection-modal');
    if (modal) {
        modal.remove();
        console.log('[DEBUG] Modal removed');
    }
}

// ===================== TRAINING MODE SELECTION =====================

function selectTrainingMode(mode) {
    selectedTrainingMode = mode;
    document.querySelectorAll('.train-option-box').forEach(box => {
        box.classList.remove('train-selected');
    });
    if (mode === 'self-train') {
        document.getElementById('self-training-option').classList.add('train-selected');
        document.getElementById('train-selection-section').style.display = 'none';
        loadDatasetInfo();
    } else if (mode === 'auto-ml') {
        document.getElementById('auto-ml-option').classList.add('train-selected');
        showComingSoonMessage();
    }
}

function showComingSoonMessage() {
    document.getElementById('train-config-section').style.display = 'block';
    document.getElementById('train-config-content').innerHTML = `
        <div class="train-config-placeholder">
            <i class="fas fa-rocket"></i>
            <h4>Auto ML Training Coming Soon!</h4>
            <p>We're working hard to bring you automated machine learning capabilities. 
               In the meantime, try our powerful Self Training mode for full control over your models.</p>
        </div>
    `;
}

function loadDatasetInfo() {
    // Check if filename is available
    if (!filename || filename === '""' || filename === 'None' || filename === null) {
        console.log('[DEBUG] No filename available, redirecting to catalog');
        alert('No dataset selected. Please select a dataset from the catalog first.');
        window.location.href = '/catalog';
        return;
    }

    showTrainLoading();
    
    // Properly encode the filename for URL
    const encodedFilename = encodeURIComponent(filename);
    
    console.log('[DEBUG] Loading dataset info for:', filename);
    console.log('[DEBUG] Encoded filename:', encodedFilename);
    
    fetch(`/train/get_dataset_info?filename=${encodedFilename}`)
        .then(response => {
            console.log('[DEBUG] Response status:', response.status);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            hideTrainLoading();
            console.log('[DEBUG] Dataset info response:', data);
            
            if (data.success) {
                datasetInfo = data;
                recommendedTasks = data.recommended_tasks || {};
                updateDatasetStats(data);
                document.getElementById('train-dataset-info').style.display = 'block';
                setupSelfTrainConfig();
            } else {
                let errorMessage = 'Failed to load dataset: ' + data.message;
                
                if (data.available_files && data.available_files.length > 0) {
                    errorMessage += '\n\nAvailable files:\n' + data.available_files.slice(0, 5).join('\n');
                    if (data.available_files.length > 5) {
                        errorMessage += `\n(and ${data.available_files.length - 5} more)`;
                    }
                }
                
                console.error('[ERROR] Dataset loading failed:', data);
                alert(errorMessage);
                window.location.href = '/catalog';
            }
        })
        .catch(error => {
            hideTrainLoading();
            console.error('[ERROR] Error loading dataset:', error);
            alert('Error loading dataset: ' + error.message);
            window.location.href = '/catalog';
        });
}

function updateDatasetStats(data) {
    document.getElementById('train-row-count').textContent = data.row_count.toLocaleString();
    document.getElementById('train-column-count').textContent = data.column_count;
    document.getElementById('train-missing-values').textContent = data.missing_values.toLocaleString();
    document.getElementById('train-target-candidates').textContent = data.target_candidates.length;
}

function setupSelfTrainConfig() {
    document.getElementById('self-train-main-config').style.display = 'block';
    
    // Populate target columns
    const sel = document.getElementById('target-column-select');
    sel.innerHTML = `<option value="" disabled selected>Select target column</option>`;
    
    if (datasetInfo.target_candidates && datasetInfo.target_candidates.length > 0) {
        datasetInfo.target_candidates.forEach(col => {
            sel.innerHTML += `<option value="${col}">${col}</option>`;
        });
    } else {
        sel.innerHTML += `<option disabled>No target candidates found</option>`;
    }
    
    // Set up change handler for target column selection
    sel.addEventListener('change', function() {
        const selected = this.value;
        const recommended = recommendedTasks[selected];
        
        const taskSelect = document.getElementById('target-task-select');
        taskSelect.innerHTML = `
            <option value="" disabled selected>Select task</option>
            <option value="regression"${recommended === 'regression' ? ' selected' : ''}>Regression${recommended === 'regression' ? ' (Recommended)' : ''}</option>
            <option value="classification"${recommended === 'classification' ? ' selected' : ''}>Classification${recommended === 'classification' ? ' (Recommended)' : ''}</option>
        `;
        
        // If a task is recommended and auto-selected, trigger the method display
        if (recommended) {
            selectedTaskType = recommended;
            showSupervisedMethods();
            enableOrDisableStartTraining();
        }
    });
}

// ===================== TASK AND METHOD SELECTION =====================

function onTargetColumnSelected() {
    selectedTarget = document.getElementById('target-column-select').value;
    
    // Check if a task is already selected and show methods
    const taskSelect = document.getElementById('target-task-select');
    if (taskSelect.value) {
        selectedTaskType = taskSelect.value;
        showSupervisedMethods();
    }
    
    enableOrDisableStartTraining();
}

function selectTaskType(taskType) {
    // Get all task type buttons
    const supervisedBtn = document.getElementById('btn-supervised');
    const unsupervisedBtn = document.getElementById('btn-unsupervised');
    
    // Get the sections to show/hide
    const supervisedSection = document.getElementById('supervised-section');
    const unsupervisedSection = document.getElementById('unsupervised-section');
    
    // Get target column elements
    const targetNote = document.getElementById('target-column-note');
    const targetSelect = document.getElementById('target-column-select');
    const noneOption = targetSelect.querySelector('option[value="none"]');
    
    // Reset all buttons to default state
    supervisedBtn.classList.remove('selected');
    unsupervisedBtn.classList.remove('selected');
    
    // Hide all sections
    supervisedSection.style.display = 'none';
    unsupervisedSection.style.display = 'none';
    
    // Handle the selected task type
    if (taskType === 'supervised') {
        // Update UI for supervised learning
        supervisedBtn.classList.add('selected');
        supervisedSection.style.display = 'block';
        
        // Hide the optional note
        targetNote.style.display = 'none';
        
        // Enable target column selection
        targetSelect.disabled = false;
        targetSelect.style.opacity = '1';
        
        // Remove "None" option if it exists (not needed for supervised)
        if (noneOption) {
            noneOption.remove();
        }
        
        // Reset target selection if it was "None"
        if (targetSelect.value === 'none') {
            targetSelect.value = '';
        }
        
        // Enable/disable start button based on selection
        document.getElementById('start-training-btn').disabled = targetSelect.value === '';
    } 
    else if (taskType === 'unsupervised') {
        // Update UI for unsupervised learning
        unsupervisedBtn.classList.add('selected');
        unsupervisedSection.style.display = 'block';
        
        // Show the optional note
        targetNote.style.display = 'block';
        
        // DISABLE target column selection for unsupervised learning
        targetSelect.disabled = true;
        targetSelect.style.opacity = '0.5';
        
        // Set to "None" option and add it if it doesn't exist
        if (!noneOption) {
            const option = document.createElement('option');
            option.value = 'none';
            option.textContent = 'None (for unsupervised learning)';
            targetSelect.insertBefore(option, targetSelect.options[1]);
        }
        
        // Automatically select "None" for unsupervised learning
        targetSelect.value = 'none';
        selectedTarget = 'none';
        
        // Enable start button (target column is not needed for unsupervised)
        document.getElementById('start-training-btn').disabled = false;
    }
    
    // Clear any existing method selections
    document.getElementById('method-select-box').innerHTML = '';
    document.getElementById('unsupervised-method-select-box').innerHTML = '';
    
    // Reset advanced settings if they're visible
    const advSettings = document.getElementById('advanced-settings-panel');
    if (advSettings.style.display !== 'none') {
        showAdvancedSettings(); // This will toggle it closed if open
    }
}

function getRecommendedMethods(taskType) {
    const recommendations = {
        regression: ['random_forest', 'xgboost', 'linear_regression'],
        classification: ['random_forest', 'xgboost', 'logistic_regression'],
        clustering: ['kmeans', 'dbscan'],
        anomaly: ['isolation_forest'],
        association: ['apriori'],  // Add this line
        dimensionality_reduction: ['pca', 'tsne']
    };
    
    return recommendations[taskType] || [];
}
function onTaskTypeSelected() {
    selectedTaskType = document.getElementById('target-task-select').value;
    showSupervisedMethods();
    selectedMethod = null;
    
    // Clear hyperparameters when task changes
    const hyperContainer = document.getElementById('adv-hyperparameters-content');
    if (hyperContainer) {
        hyperContainer.innerHTML = '<em>Select a method to configure hyperparameters</em>';
    }
    
    enableOrDisableStartTraining();
}

function showSupervisedMethods() {
    const methodBox = document.getElementById('method-select-box');
    
    if (!selectedTaskType || !methodOptions[selectedTaskType]) {
        methodBox.innerHTML = '';
        return;
    }
    
    const recommendedMethods = getRecommendedMethods(selectedTaskType);
    
    let html = `<label class="self-train-config-label">Method</label>
                <select class="self-train-select" id="method-select" onchange="onMethodSelected()">
                <option value="" disabled selected>Select method</option>`;
    
    methodOptions[selectedTaskType].forEach(method => {
        const isRecommended = recommendedMethods.includes(method.value);
        html += `<option value="${method.value}" ${isRecommended ? 'data-recommended="true"' : ''}>
                    ${method.label}${isRecommended ? ' (Recommended)' : ''}
                </option>`;
    });
    
    html += `</select>
            <div class="method-recommendations">
                <i class="fas fa-lightbulb"></i> <strong>Recommended for ${selectedTaskType}:</strong> 
                ${recommendedMethods.map(m => methodOptions[selectedTaskType].find(o => o.value === m).label).join(', ')}
            </div>`;
    
    methodBox.innerHTML = html;
}

function onMethodSelected() {
    selectedMethod = document.getElementById('method-select').value;
    renderHyperparameterInputs(selectedMethod, selectedTaskType);
    enableOrDisableStartTraining();
}

function onUnsupervisedTypeSelected() {
    const selectElement = document.getElementById('unsup-task-type-select');
    if (!selectElement) {
        console.error('[ERROR] Unsupervised task type select element not found');
        return;
    }
    
    selectedUnsupervisedType = selectElement.value;
    console.log('[DEBUG] Unsupervised type selected:', selectedUnsupervisedType);
    
    // Show the methods for this type
    showUnsupervisedMethods();
    
    // Reset method selection
    selectedUnsupervisedMethod = null;
    
    // Clear hyperparameters when unsupervised type changes
    const hyperContainer = document.getElementById('adv-hyperparameters-content');
    if (hyperContainer) {
        hyperContainer.innerHTML = '<em>Select a method to configure hyperparameters</em>';
    }
    
    // Update training button state
    enableOrDisableStartTraining();
}

function showUnsupervisedMethods() {
    const methodBox = document.getElementById('unsupervised-method-select-box');
    
    if (!selectedUnsupervisedType || !methodOptions[selectedUnsupervisedType]) {
        console.log('[DEBUG] No unsupervised type selected or no methods available:', selectedUnsupervisedType);
        methodBox.innerHTML = '';
        return;
    }
    
    console.log('[DEBUG] Showing methods for:', selectedUnsupervisedType);
    console.log('[DEBUG] Available methods:', methodOptions[selectedUnsupervisedType]);
    
    const recommendedMethods = getRecommendedMethods(selectedUnsupervisedType);
    
    let html = `<label class="self-train-config-label">Method</label>
                <select class="self-train-select" id="unsup-method-select" onchange="onUnsupervisedMethodSelected()">
                <option value="" disabled selected>Select method</option>`;
    
    methodOptions[selectedUnsupervisedType].forEach(method => {
        const isRecommended = recommendedMethods.includes(method.value);
        html += `<option value="${method.value}" ${isRecommended ? 'data-recommended="true"' : ''}>
                    ${method.label}${isRecommended ? ' (Recommended)' : ''}
                </option>`;
    });
    
    html += `</select>
            <div class="method-recommendations">
                <i class="fas fa-lightbulb"></i> <strong>Recommended for ${selectedUnsupervisedType.replace('_', ' ')}:</strong> 
                ${recommendedMethods.map(m => {
                    const methodObj = methodOptions[selectedUnsupervisedType].find(o => o.value === m);
                    return methodObj ? methodObj.label : m;
                }).join(', ')}
            </div>`;
    
    methodBox.innerHTML = html;
    console.log('[DEBUG] Method selection HTML updated');
}

function onUnsupervisedMethodSelected() {
    selectedUnsupervisedMethod = document.getElementById('unsup-method-select').value;
    renderHyperparameterInputs(selectedUnsupervisedMethod, selectedUnsupervisedType);
    enableOrDisableStartTraining();
}

function renderHyperparameterInputs(method, taskType) {
    const container = document.getElementById('adv-hyperparameters-content');
    if (!method || !taskType) {
        container.innerHTML = '<em>Select a method to configure hyperparameters</em>';
        return;
    }
    
    container.innerHTML = '<em>Loading hyperparameters...</em>';
    
    fetch(`/train/get_model_hyperparameters?task_type=${taskType}&method=${method}`)
        .then(response => response.json())
        .then(data => {
            if (!data.success) {
                container.innerHTML = `<em>${data.message || 'No hyperparameters found'}</em>`;
                return;
            }
            
            const params = data.params;
            if (!params || Object.keys(params).length === 0) {
                container.innerHTML = '<em>No hyperparameters for this model</em>';
                return;
            }
            
            let html = '';
            Object.entries(params).forEach(([key, cfg]) => {
                if (cfg.type === 'int' || cfg.type === 'float') {
                    html += `
                        <div class="adv-param-row">
                            <label>${key} (${cfg.type})</label>
                            <input type="number" name="${key}" value="${cfg.default}" min="${cfg.min}" max="${cfg.max}" ${cfg.type === "float" ? 'step="any"' : ''}>
                        </div>`;
                } else if (cfg.type === 'select') {
                    html += `
                        <div class="adv-param-row">
                            <label>${key}</label>
                            <select name="${key}">
                                ${cfg.options.map(opt => `<option value="${opt}" ${opt === cfg.default ? 'selected' : ''}>${opt}</option>`).join('')}
                            </select>
                        </div>`;
                } else if (cfg.type === 'bool') {
                    html += `
                        <div class="adv-param-row">
                            <label>
                                <input type="checkbox" name="${key}" ${cfg.default ? 'checked' : ''}>
                                ${key}
                            </label>
                        </div>`;
                }
            });
            container.innerHTML = html;
        })
        .catch(error => {
            console.error('[ERROR] Error loading hyperparameters:', error);
            container.innerHTML = '<em>Error loading hyperparameters</em>';
        });
}

// ===================== TRAINING EXECUTION =====================

function showAdvancedSettings() {
    const panel = document.getElementById('advanced-settings-panel');
    panel.style.display = (panel.style.display === 'none' ? 'block' : 'none');
}

function enableOrDisableStartTraining() {
    let enable = false;
    
    if (document.getElementById('unsupervised-section').style.display === 'block') {
        // Unsupervised mode - only need type and method
        enable = !!(selectedUnsupervisedType && selectedUnsupervisedMethod);
    } else {
        // Supervised mode - need target, task type, and method
        enable = !!(selectedTarget && selectedTaskType && selectedMethod);
    }
    
    const btn = document.getElementById('start-training-btn');
    if (btn) {
        btn.disabled = !enable;
        btn.classList.toggle('enabled', enable);
    }
    
    return enable;
}

function startTraining() {
    if (!enableOrDisableStartTraining()) {
        console.log('[DEBUG] Training not enabled, requirements not met');
        return;
    }
    
    if (!selectedProject) {
        showProjectSelection(function(projectId) {
            selectedProject = projectId;
            startTraining();
        });
        return;
    }

    console.log('[DEBUG] Starting training process');
    showTrainLoading();
    
    // Prepare training configuration
    const isUnsupervised = document.getElementById('unsupervised-section').style.display === 'block';
    
    // Collect hyperparameters
    const modelParams = {};
    const currentMethod = isUnsupervised ? selectedUnsupervisedMethod : selectedMethod;
    if (currentMethod) {
        const hyperParamInputs = document.querySelectorAll('#adv-hyperparameters-content input, #adv-hyperparameters-content select');
        hyperParamInputs.forEach(input => {
            if (input.type === 'checkbox') {
                modelParams[input.name] = input.checked;
            } else if (input.type === 'number') {
                const value = parseFloat(input.value);
                if (!isNaN(value)) {
                    modelParams[input.name] = value;
                }
            } else {
                modelParams[input.name] = input.value;
            }
        });
    }
    
    let trainConfig;
    
    if (isUnsupervised) {
        // Unsupervised learning configuration - feature_columns is now optional
        trainConfig = {
            filename: filename,
            training_mode: 'self-train',
            problem_type: 'unsupervised',
            unsupervised_type: selectedUnsupervisedType,
            selected_methods: [selectedUnsupervisedMethod],
            model_params: { [selectedUnsupervisedMethod]: modelParams },
            project_id: selectedProject
        };
        
        // Only add feature_columns if they've been explicitly selected
        if (datasetInfo && datasetInfo.columns && datasetInfo.columns.length > 0) {
            trainConfig.feature_columns = datasetInfo.columns;
        }
    } else {
        // Supervised learning configuration remains the same
        trainConfig = {
            filename: filename,
            training_mode: 'self-train',
            problem_type: selectedTaskType,
            target_column: selectedTarget,
            feature_columns: datasetInfo.columns.filter(col => col !== selectedTarget),
            selected_models: [selectedMethod],
            model_params: { [selectedMethod]: modelParams },
            train_test_config: {
                test_size: (100 - parseInt(document.getElementById('split-train-size').value || 80)) / 100,
                shuffle: document.getElementById('split-shuffle').checked,
                random_state: parseInt(document.getElementById('split-random-state').value) || null
            },
            cv_config: {
                enabled: document.getElementById('cv-type-select').value !== 'none',
                method: document.getElementById('cv-type-select').value,
                folds: parseInt(document.getElementById('cv-k')?.value) || 5
            },
            project_id: selectedProject
        };
    }

    console.log('[DEBUG] Training config:', trainConfig);

    // Start training
    fetch('/train/start_training', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainConfig)
    })
    .then(response => {
        console.log('[DEBUG] Training response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        hideTrainLoading();
        console.log('[DEBUG] Training response:', data);
        
        if (data.success) {
            displayTrainingResults(data);
        } else {
            alert('Training failed: ' + (data.message || data.error));
        }
    })
    .catch(error => {
        hideTrainLoading();
        console.error('[ERROR] Training error:', error);
        alert('Training error: ' + error.message);
    });
}

function displayTrainingResults(data) {
    const resultsSection = document.getElementById('train-results-section');
    const resultsContent = document.getElementById('training-results-content');

    let html = `
        <div class="results-header">
            <h4>🎉 Training Completed Successfully!</h4>
            <p>Model ID: <strong>${data.model_id}</strong></p>
        </div>
        <div class="results-grid">
    `;

    // Display results based on problem type
    if (data.config && (data.config.problem_type === 'unsupervised' || data.config.unsupervised_type)) {
        // Main unsupervised info card
        const unsupervisedType = data.config.unsupervised_type || data.config.problem_type;
        html += `
            <div class="result-card">
                <h5><i class="fas fa-project-diagram"></i> ${unsupervisedType.charAt(0).toUpperCase() + unsupervisedType.slice(1)} Results</h5>
                <div class="result-details">
                    <p><strong>Problem Type:</strong> ${unsupervisedType.replace('_', ' ').charAt(0).toUpperCase() + unsupervisedType.slice(1)}</p>
                    <p><strong>Features Used:</strong> ${data.config.n_features || 'N/A'}</p>
                    <p><strong>Samples:</strong> ${data.config.n_samples || 'N/A'}</p>
                    <p><strong>Methods:</strong> ${data.config.methods_used ? data.config.methods_used.join(', ') : 'N/A'}</p>
                </div>
            </div>
        `;
        
        // Display specific method results if available
        if (data.results) {
            Object.entries(data.results).forEach(([method, result]) => {
                console.log('[DEBUG] Processing method:', method, result);
                
                if (result.error) {
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-exclamation-triangle"></i> ${result.name || method}</h5>
                            <div class="result-details">
                                <p style="color: #e74c3c;"><strong>Error:</strong> ${result.error}</p>
                            </div>
                        </div>
                    `;
                    return;
                }

                // Handle clustering results
                if (result.type === 'clustering' || ['kmeans', 'dbscan', 'agglomerative'].includes(method)) {
                    const silhouetteScore = result.silhouette_score || (result.metrics && result.metrics.test ? result.metrics.test.silhouette_score : null);
                    const calinsiScore = result.calinski_harabasz_score || (result.metrics && result.metrics.test ? result.metrics.test.calinski_harabasz_score : null);
                    const daviesScore = result.davies_bouldin_score || (result.metrics && result.metrics.test ? result.metrics.test.davies_bouldin_score : null);
                    
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-sitemap"></i> ${result.name || method}</h5>
                            <div class="result-details">
                                <p><strong>Algorithm:</strong> ${result.name || method}</p>
                                ${result.n_clusters !== undefined ? `<p><strong>Clusters Found:</strong> ${result.n_clusters}</p>` : ''}
                                ${result.noise_points !== undefined ? `<p><strong>Noise Points:</strong> ${result.noise_points}</p>` : ''}
                                ${silhouetteScore !== null && silhouetteScore !== undefined ? `<p><strong>Silhouette Score:</strong> ${(silhouetteScore * 100).toFixed(1)}%</p>` : ''}
                                ${calinsiScore !== null && calinsiScore !== undefined ? `<p><strong>Calinski-Harabasz Score:</strong> ${calinsiScore.toFixed(3)}</p>` : ''}
                                ${daviesScore !== null && daviesScore !== undefined ? `<p><strong>Davies-Bouldin Score:</strong> ${daviesScore.toFixed(3)}</p>` : ''}
                                ${result.cluster_sizes ? `<p><strong>Cluster Sizes:</strong> ${result.cluster_sizes.join(', ')}</p>` : ''}
                            </div>
                        </div>
                    `;
                }
                
                // Handle dimensionality reduction results
                else if (result.type === 'dimensionality_reduction' || ['pca', 'tsne'].includes(method)) {
                    const varianceExplained = result.variance_explained || (result.metrics && result.metrics.test ? result.metrics.test.variance_explained : null);
                    
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-compress-arrows-alt"></i> ${result.name || method}</h5>
                            <div class="result-details">
                                <p><strong>Algorithm:</strong> ${result.name || method}</p>
                                ${result.original_dimensions ? `<p><strong>Original Dimensions:</strong> ${result.original_dimensions}</p>` : ''}
                                ${result.reduced_dimensions ? `<p><strong>Reduced Dimensions:</strong> ${result.reduced_dimensions}</p>` : ''}
                                ${varianceExplained !== null && varianceExplained !== undefined ? `<p><strong>Variance Explained:</strong> ${(varianceExplained * 100).toFixed(1)}%</p>` : ''}
                                ${result.kl_divergence ? `<p><strong>KL Divergence:</strong> ${result.kl_divergence.toFixed(6)}</p>` : ''}
                                ${result.variance_ratio_per_component ? `<p><strong>Components:</strong> ${result.variance_ratio_per_component.length}</p>` : ''}
                            </div>
                        </div>
                    `;
                }
                
                // Handle anomaly detection results
                else if (result.type === 'anomaly_detection' || ['isolation_forest', 'one_class_svm'].includes(method)) {
                    const anomalyRatio = result.anomaly_ratio || (result.metrics && result.metrics.test ? result.metrics.test.outlier_ratio : null);
                    
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-search"></i> ${result.name || method}</h5>
                            <div class="result-details">
                                <p><strong>Algorithm:</strong> ${result.name || method}</p>
                                ${result.anomalies_detected !== undefined ? `<p><strong>Anomalies Detected:</strong> ${result.anomalies_detected}</p>` : ''}
                                ${result.normal_points !== undefined ? `<p><strong>Normal Points:</strong> ${result.normal_points}</p>` : ''}
                                ${anomalyRatio !== null && anomalyRatio !== undefined ? `<p><strong>Anomaly Ratio:</strong> ${(anomalyRatio * 100).toFixed(2)}%</p>` : ''}
                                ${result.contamination !== undefined ? `<p><strong>Contamination:</strong> ${(result.contamination * 100).toFixed(1)}%</p>` : ''}
                            </div>
                        </div>
                    `;
                }
                
                // Handle association rule mining results
                else if (result.type === 'association_rules' || ['apriori', 'eclat'].includes(method)) {
                    const maxLift = result.max_lift || (result.metrics && result.metrics.test ? result.metrics.test.max_lift : null);
                    
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-network-wired"></i> ${result.name || method}</h5>
                            <div class="result-details">
                                <p><strong>Algorithm:</strong> ${result.name || method}</p>
                                ${result.frequent_itemsets_count !== undefined ? `<p><strong>Frequent Itemsets:</strong> ${result.frequent_itemsets_count}</p>` : ''}
                                ${result.rules_count !== undefined ? `<p><strong>Association Rules:</strong> ${result.rules_count}</p>` : ''}
                                ${maxLift !== null && maxLift !== undefined ? `<p><strong>Max Lift:</strong> ${maxLift.toFixed(3)}</p>` : ''}
                                ${result.min_support !== undefined ? `<p><strong>Min Support:</strong> ${result.min_support}</p>` : ''}
                                ${result.min_confidence !== undefined ? `<p><strong>Min Confidence:</strong> ${result.min_confidence}</p>` : ''}
                                ${result.message ? `<p><strong>Note:</strong> ${result.message}</p>` : ''}
                            </div>
                        </div>
                    `;
                }
                
                // Generic fallback for any other unsupervised method
                else {
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-cog"></i> ${result.name || method}</h5>
                            <div class="result-details">
                                <p><strong>Method:</strong> ${result.name || method}</p>
                                <p><strong>Type:</strong> ${result.type || 'unsupervised'}</p>
                                <p><strong>Status:</strong> ✓ Completed</p>
                                ${result.parameters ? `<p><strong>Parameters:</strong> ${Object.keys(result.parameters).length} configured</p>` : ''}
                            </div>
                        </div>
                    `;
                }
            });
        }
    } 
    
    // Handle supervised learning results (existing code)
    else {
        html += `
            <div class="result-card">
                <h5><i class="fas fa-chart-line"></i> Supervised Learning Results</h5>
                <div class="result-details">
                    <p><strong>Problem Type:</strong> ${data.config ? data.config.problem_type : 'Unknown'}</p>
                    <p><strong>Training Samples:</strong> ${data.config ? data.config.train_size : 'N/A'}</p>
                    <p><strong>Test Samples:</strong> ${data.config ? data.config.test_size : 'N/A'}</p>
                    <p><strong>Features:</strong> ${data.config ? data.config.n_features : 'N/A'}</p>
                </div>
            </div>
        `;
        
        // Display model results if available
        if (data.results) {
            Object.entries(data.results).forEach(([model, result]) => {
                if (result.error) {
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-exclamation-triangle"></i> ${result.name || model}</h5>
                            <div class="result-details">
                                <p style="color: #e74c3c;"><strong>Error:</strong> ${result.error}</p>
                            </div>
                        </div>
                    `;
                    return;
                }

                if (result.metrics && result.metrics.test) {
                    const testMetrics = result.metrics.test;
                    html += `
                        <div class="result-card">
                            <h5><i class="fas fa-robot"></i> ${result.name || model}</h5>
                            <div class="result-details">
                                ${testMetrics.accuracy ? `<p><strong>Accuracy:</strong> ${(testMetrics.accuracy * 100).toFixed(2)}%</p>` : ''}
                                ${testMetrics.r2 ? `<p><strong>R² Score:</strong> ${testMetrics.r2.toFixed(3)}</p>` : ''}
                                ${testMetrics.f1 ? `<p><strong>F1 Score:</strong> ${testMetrics.f1.toFixed(3)}</p>` : ''}
                                ${testMetrics.precision ? `<p><strong>Precision:</strong> ${testMetrics.precision.toFixed(3)}</p>` : ''}
                                ${testMetrics.recall ? `<p><strong>Recall:</strong> ${testMetrics.recall.toFixed(3)}</p>` : ''}
                                ${testMetrics.mse ? `<p><strong>MSE:</strong> ${testMetrics.mse.toFixed(3)}</p>` : ''}
                                ${testMetrics.rmse ? `<p><strong>RMSE:</strong> ${testMetrics.rmse.toFixed(3)}</p>` : ''}
                                ${testMetrics.mae ? `<p><strong>MAE:</strong> ${testMetrics.mae.toFixed(3)}</p>` : ''}
                            </div>
                        </div>
                    `;
                }
            });
        }
    }

    html += `
        </div>
        <div class="results-actions">
            <button class="btn-secondary" onclick="downloadModel('${data.model_id}')">
                <i class="fas fa-download"></i> Download Model
            </button>
            <button class="btn-secondary btn-xai" onclick="goToSavedModels()">
                <i class="fas fa-eye"></i> View in XAI Section
            </button>
        </div>
    `;

    resultsContent.innerHTML = html;
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// ===================== DOWNLOAD FUNCTION =====================

function downloadModel(modelId) {
    if (!modelId) {
        console.error('[ERROR] No model ID provided for download');
        showNotification('Error: No model ID available', 'error');
        return;
    }
    
    showNotification('Starting download...', 'info');
    
    // Use the correct XAI route for best model download!
    const downloadUrl = `/xai/api/models/${modelId}/download/best`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${modelId}_best_model.pkl`;
    link.style.display = 'none';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    console.log('[DEBUG] Download initiated for:', downloadUrl);
}

function goToSavedModels() {
    window.location.href = "/train/xai";
}

// ===================== UTILITY FUNCTIONS =====================

function showTrainLoading() {
    const overlay = document.getElementById('train-loading-overlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
}

function hideTrainLoading() {
    const overlay = document.getElementById('train-loading-overlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function updateCVOptionsPanel() {
    const type = document.getElementById('cv-type-select').value;
    const panel = document.getElementById('cv-options-panel');
    
    if (!panel) return;
    
    if (type === "kfold" || type === "stratifiedkfold") {
        panel.innerHTML = `
            <label for="cv-k">Number of Folds (k)</label>
            <input type="number" id="cv-k" min="2" max="30" value="5">
            <label for="cv-random-state" style="margin-top:10px;">Random Seed</label>
            <input type="number" id="cv-random-state" placeholder="e.g. 42">
        `;
    } else if (type === "shuffle_split") {
        panel.innerHTML = `
            <label for="cv-n-splits">Splits</label>
            <input type="number" id="cv-n-splits" min="2" max="50" value="5">
            <label for="cv-test-size" style="margin-top:10px;">Test Size (%)</label>
            <input type="number" id="cv-test-size" min="1" max="99" value="20">
            <label for="cv-random-state" style="margin-top:10px;">Random Seed</label>
            <input type="number" id="cv-random-state" placeholder="e.g. 42">
        `;
    } else {
        panel.innerHTML = '';
    }
}

function showNotification(message, type = 'info') {
    // Remove existing notifications
    document.querySelectorAll('.notification').forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-triangle' : 'info-circle'}"></i>
        <span>${message}</span>
        <button class="notification-close" onclick="this.parentElement.remove()">&times;</button>
    `;
    
    // Add some basic styles if not already defined
    if (!document.querySelector('style[data-notification-styles]')) {
        const style = document.createElement('style');
        style.setAttribute('data-notification-styles', 'true');
        style.textContent = `
            .notification {
                position: fixed;
                top: 20px;
                right: 20px;
                background: #fff;
                border-left: 4px solid #007bff;
                padding: 15px 20px;
                border-radius: 4px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                z-index: 10000;
                opacity: 0;
                transform: translateX(100%);
                transition: all 0.3s ease;
                max-width: 400px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .notification.show {
                opacity: 1;
                transform: translateX(0);
            }
            .notification-success { border-left-color: #28a745; }
            .notification-error { border-left-color: #dc3545; }
            .notification-close {
                background: none;
                border: none;
                font-size: 18px;
                cursor: pointer;
                margin-left: auto;
            }
        `;
        document.head.appendChild(style);
    }
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.classList.add('show');
    }, 100);
    
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Export functions for debugging
window.debugTraining = {
    loadDatasetInfo,
    enableOrDisableStartTraining,
    getFilename: () => filename,
    getDatasetInfo: () => datasetInfo,
    getSelectedValues: () => ({
        target: selectedTarget,
        taskType: selectedTaskType,
        method: selectedMethod,
        unsupervisedType: selectedUnsupervisedType,
        unsupervisedMethod: selectedUnsupervisedMethod
    }),
    testDownload: (modelId) => downloadModel(modelId),
    testAnalyze: (modelId) => analyzeModel(modelId)
};