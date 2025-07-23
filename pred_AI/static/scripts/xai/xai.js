// Global variables
let analysisCharts = [];
let selectedModels = [];
let globalModelData = {};
let currentComparisonProject = null;

// ===================== FIXED ANALYZE MODEL FUNCTION =====================
function analyzeModel(projectId, modelId) {
    console.log('[DEBUG] Analyzing model:', projectId, modelId);
    showNotification('Loading model analysis...', 'info');
    
    // FIX: Use the correct route based on your analysis_routes.py
    // The route in analysis_routes.py is defined as @analysis_bp.route('/<model_id>')
    // And it's registered with prefix '/xai/analysis'
    // So the full path should be '/xai/analysis/model_id'
    const analysisUrl = `/xai/analysis/${modelId}`;
    
    console.log('[DEBUG] Redirecting to:', analysisUrl);
    
    // Optional: Check if the route exists first
    fetch(`/xai/api/models/${modelId}/analysis`, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                // Route exists, proceed with navigation
                window.location.href = analysisUrl;
            } else {
                // Route doesn't exist, try alternative
                console.log('[DEBUG] Primary route failed, trying alternative...');
                // Try the XAI route instead
                window.location.href = `/xai/analysis/${modelId}`;
            }
        })
        .catch(error => {
            console.log('[DEBUG] Route check failed, proceeding anyway:', error);
            // If check fails, try the route anyway
            window.location.href = analysisUrl;
        });
}

// ===================== PROJECT MANAGEMENT =====================
function toggleProjectExpansion(projectId) {
    const projectContent = document.getElementById(`project-content-${projectId}`);
    const expandIcon = document.getElementById(`expand-icon-${projectId}`);
    const projectCard = document.querySelector(`[data-project-id="${projectId}"]`);
    
    if (!projectContent || !expandIcon) return;
    
    const isExpanded = projectContent.style.display === 'block';
    
    if (isExpanded) {
        projectContent.style.display = 'none';
        expandIcon.innerHTML = '<i class="fas fa-chevron-right"></i>';
        projectCard.classList.remove('expanded');
        const comparisonSection = document.getElementById(`project-comparison-${projectId}`);
        if (comparisonSection) comparisonSection.style.display = 'none';
    } else {
        projectContent.style.display = 'block';
        expandIcon.innerHTML = '<i class="fas fa-chevron-down"></i>';
        projectCard.classList.add('expanded');
        
        projectContent.style.opacity = '0';
        projectContent.style.transform = 'translateY(-10px)';
        setTimeout(() => {
            projectContent.style.transition = 'all 0.3s ease';
            projectContent.style.opacity = '1';
            projectContent.style.transform = 'translateY(0)';
        }, 10);
    }
}

function showCreateProjectModal() {
    const modal = document.getElementById('create-project-modal');
    if (modal) {
        const nameInput = document.getElementById('project-name');
        const descInput = document.getElementById('project-description');
        if (nameInput) nameInput.value = '';
        if (descInput) descInput.value = '';
        
        modal.classList.add('show');
        modal.style.display = 'flex';
        modal.style.visibility = 'visible';
        modal.style.opacity = '1';
        
        setTimeout(() => {
            if (nameInput) nameInput.focus();
        }, 100);
    }
}

function hideCreateProjectModal() {
    const modal = document.getElementById('create-project-modal');
    if (modal) {
        modal.classList.remove('show');
        modal.style.display = 'none';
        modal.style.visibility = 'hidden';
        modal.style.opacity = '0';
    }
}

function createProject() {
    const name = document.getElementById('project-name').value.trim();
    const description = document.getElementById('project-description').value.trim();
    
    if (!name) {
        alert('Project name is required');
        document.getElementById('project-name').focus();
        return;
    }
    
    const createBtn = document.querySelector('.modal-footer .btn-primary');
    createBtn.disabled = true;
    createBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating...';
    
    fetch('/xai/api/projects', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, description: description })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            hideCreateProjectModal();
            showNotification('Project created successfully!', 'success');
            setTimeout(() => location.reload(), 1000);
        } else {
            showNotification('Failed to create project: ' + data.message, 'error');
        }
    })
    .catch(error => {
        showNotification('Error creating project: ' + error.message, 'error');
    })
    .finally(() => {
        createBtn.disabled = false;
        createBtn.innerHTML = '<i class="fas fa-plus"></i> Create Project';
    });
}

function deleteProject(projectId) {
    if (!confirm('Are you sure you want to delete this project and all its models?')) return;
    
    showLoadingOverlay('Deleting project...');
    
    fetch(`/xai/api/projects/${projectId}`, { method: 'DELETE' })
    .then(response => response.json())
    .then(data => {
        hideLoadingOverlay();
        if (data.success) {
            const projectCard = document.querySelector(`[data-project-id="${projectId}"]`);
            if (projectCard) {
                projectCard.style.animation = 'fadeOut 0.3s ease-out';
                setTimeout(() => {
                    projectCard.remove();
                    const remainingProjects = document.querySelectorAll('.project-card');
                    if (remainingProjects.length === 0) location.reload();
                }, 300);
            }
            showNotification('Project deleted successfully', 'success');
        } else {
            showNotification('Failed to delete project: ' + data.message, 'error');
        }
    })
    .catch(error => {
        hideLoadingOverlay();
        showNotification('Error deleting project: ' + error.message, 'error');
    });
}

// ===================== COMPARISON FUNCTIONS =====================
function enableProjectComparison(projectId) {
    console.log('[DEBUG] Enabling comparison for project:', projectId);
    
    const projectContent = document.getElementById(`project-content-${projectId}`);
    if (!projectContent || projectContent.style.display === 'none') {
        toggleProjectExpansion(projectId);
    }
    
    currentComparisonProject = projectId;
    
    const comparisonSection = document.getElementById(`project-comparison-${projectId}`);
    if (comparisonSection) {
        comparisonSection.style.display = 'block';
        
        const modelASelect = document.getElementById(`project-model-a-${projectId}`);
        const modelBSelect = document.getElementById(`project-model-b-${projectId}`);
        if (modelASelect) modelASelect.value = '';
        if (modelBSelect) modelBSelect.value = '';
        
        const resultsDiv = document.getElementById(`project-comparison-results-${projectId}`);
        if (resultsDiv) resultsDiv.style.display = 'none';
        
        setTimeout(() => {
            comparisonSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 300);
    }
}

function hideProjectComparison(projectId) {
    const comparisonSection = document.getElementById(`project-comparison-${projectId}`);
    if (comparisonSection) comparisonSection.style.display = 'none';
    
    const resultsDiv = document.getElementById(`project-comparison-results-${projectId}`);
    if (resultsDiv) resultsDiv.style.display = 'none';
    
    currentComparisonProject = null;
}

// ===================== ENHANCED COMPARISON UPDATE FUNCTION =====================
function updateProjectComparison(projectId) {
    console.log('[DEBUG] Updating comparison for project:', projectId);
    
    const modelASelect = document.getElementById(`project-model-a-${projectId}`);
    const modelBSelect = document.getElementById(`project-model-b-${projectId}`);
    
    if (!modelASelect || !modelBSelect) {
        console.error('[ERROR] Model select elements not found');
        return;
    }
    
    const modelA = modelASelect.value;
    const modelB = modelBSelect.value;
    
    console.log('[DEBUG] Selected models:', modelA, 'vs', modelB);
    
    const resultsDiv = document.getElementById(`project-comparison-results-${projectId}`);
    
    if (!modelA || !modelB || modelA === modelB) {
        if (resultsDiv) resultsDiv.style.display = 'none';
        return;
    }
    
    // Show enhanced loading state
    showLoadingOverlay('Loading model comparison...');
    
    const apiUrl = `/xai/api/projects/${projectId}/compare/${modelA}/${modelB}`;
    console.log('[DEBUG] Making API call to:', apiUrl);
    
    fetch(apiUrl)
        .then(response => response.json())
        .then(data => {
            hideLoadingOverlay();
            console.log('[DEBUG] API response data:', data);
            
            if (data.success) {
                renderEnhancedModelComparison(projectId, data);
            } else {
                showNotification('Error comparing models: ' + data.message, 'error');
            }
        })
        .catch(error => {
            hideLoadingOverlay();
            console.error('[ERROR] API call failed:', error);
            showNotification('Error: ' + error.message, 'error');
        });
}

// ===================== ENHANCED COMPARISON RENDERING =====================
function renderEnhancedModelComparison(projectId, comparisonData) {
    console.log('[DEBUG] Rendering enhanced comparison for project:', projectId);
    
    const resultsDiv = document.getElementById(`project-comparison-results-${projectId}`);
    if (!resultsDiv) {
        console.error('[ERROR] Results div not found');
        return;
    }
    
    const modelA = comparisonData.model_a;
    const modelB = comparisonData.model_b;
    
    // Show loading state first
    resultsDiv.innerHTML = `
        <div class="comparison-loading">
            <div class="spinner"></div>
            <p>Analyzing model performance...</p>
        </div>
    `;
    resultsDiv.style.display = 'block';
    
    // Generate enhanced comparison HTML
    setTimeout(() => {
        const html = generateEnhancedComparisonHTML(projectId, comparisonData, modelA, modelB);
        resultsDiv.innerHTML = html;
        
        // CRITICAL: Apply enhanced styling after rendering
        setTimeout(() => {
            applyEnhancedComparisonStyling(projectId, comparisonData);
            addEnhancedInteractivity();
            
            resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }, 100);
        
    }, 1000);
}

function generateEnhancedComparisonHTML(projectId, comparisonData, modelA, modelB) {
    return `
        <div class="comparison-matrix-clean enhanced-comparison">
            <div class="comparison-header">
                <h3>
                    <i class="fas fa-balance-scale"></i>
                    Enhanced Model Performance Comparison
                </h3>
            </div>
            
            <!-- Enhanced Model Information Cards -->
            <div class="model-comparison-grid">
                <div class="model-comparison-card model-a enhanced-card">
                    <div class="model-header">
                        <div class="model-label">Model A</div>
                        <div class="model-name">${modelA.algorithm || 'Unknown Algorithm'}</div>
                    </div>
                    <div class="model-id enhanced-id" title="Click to copy">${truncateModelId(modelA.id)}</div>
                    <div class="model-meta">
                        <span class="problem-type enhanced-badge">${modelA.problem_type}</span>
                        <span class="created-date enhanced-badge">${formatDate(modelA.created_at)}</span>
                    </div>
                </div>
                
                <div class="vs-divider enhanced-vs">
                    <div class="vs-circle enhanced-vs-circle">
                        <span>VS</span>
                    </div>
                </div>
                
                <div class="model-comparison-card model-b enhanced-card">
                    <div class="model-header">
                        <div class="model-label">Model B</div>
                        <div class="model-name">${modelB.algorithm || 'Unknown Algorithm'}</div>
                    </div>
                    <div class="model-id enhanced-id" title="Click to copy">${truncateModelId(modelB.id)}</div>
                    <div class="model-meta">
                        <span class="problem-type enhanced-badge">${modelB.problem_type}</span>
                        <span class="created-date enhanced-badge">${formatDate(modelB.created_at)}</span>
                    </div>
                </div>
            </div>
            
            <!-- Enhanced Metrics Table -->
            <div class="metrics-comparison enhanced-metrics">
                <div class="metrics-table-container enhanced-table">
                    <table class="clean-metrics-table enhanced-metrics-table" id="enhanced-metrics-table-${projectId}">
                        <thead>
                            <tr>
                                <th>Performance Metric</th>
                                <th>Model A</th>
                                <th>Model B</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${generateEnhancedMetricRows(modelA, modelB, comparisonData)}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Enhanced Winner Section -->
            ${generateEnhancedWinnerSection(comparisonData, modelA, modelB)}
        </div>
    `;
}

function generateEnhancedMetricRows(modelA, modelB, comparisonData) {
    const metricsA = modelA.metrics?.test || {};
    const metricsB = modelB.metrics?.test || {};
    
    // For unsupervised models, also check direct properties
    if (Object.keys(metricsA).length === 0 && modelA.problem_type !== 'classification' && modelA.problem_type !== 'regression') {
        // Check for direct metric properties in results
        if (modelA.silhouette_score !== undefined) metricsA.silhouette_score = modelA.silhouette_score;
        if (modelA.n_clusters !== undefined) metricsA.n_clusters = modelA.n_clusters;
        if (modelA.variance_explained !== undefined) metricsA.variance_explained = modelA.variance_explained;
        if (modelA.anomaly_ratio !== undefined) metricsA.outlier_ratio = modelA.anomaly_ratio;
        if (modelA.rules_count !== undefined) metricsA.rules_count = modelA.rules_count;
    }
    
    if (Object.keys(metricsB).length === 0 && modelB.problem_type !== 'classification' && modelB.problem_type !== 'regression') {
        // Check for direct metric properties in results
        if (modelB.silhouette_score !== undefined) metricsB.silhouette_score = modelB.silhouette_score;
        if (modelB.n_clusters !== undefined) metricsB.n_clusters = modelB.n_clusters;
        if (modelB.variance_explained !== undefined) metricsB.variance_explained = modelB.variance_explained;
        if (modelB.anomaly_ratio !== undefined) metricsB.outlier_ratio = modelB.anomaly_ratio;
        if (modelB.rules_count !== undefined) metricsB.rules_count = modelB.rules_count;
    }
    
    const essentialMetrics = getEssentialMetrics(modelA.problem_type);
    
    let tableRows = '';
    essentialMetrics.forEach((metric, index) => {
        if (metricsA[metric] !== undefined || metricsB[metric] !== undefined) {
            const valueA = metricsA[metric];
            const valueB = metricsB[metric];
            const formattedA = formatMetricValue(valueA, metric);
            const formattedB = formatMetricValue(valueB, metric);
            
            const comparison = compareMetricValues(metric, valueA, valueB);
            
            tableRows += `
                <tr class="metric-row enhanced-metric-row" data-metric="${metric}">
                    <td class="metric-name enhanced-metric-name">${formatMetricName(metric)}</td>
                    <td class="metric-value enhanced-metric-value ${comparison === 'A' ? 'enhanced-winner' : 'enhanced-loser'}" 
                        data-value="${valueA}" data-model="A">${formattedA}</td>
                    <td class="metric-value enhanced-metric-value ${comparison === 'B' ? 'enhanced-winner' : 'enhanced-loser'}" 
                        data-value="${valueB}" data-model="B">${formattedB}</td>
                </tr>
            `;
        }
    });

    return tableRows;
}

function generateEnhancedWinnerSection(comparisonData, modelA, modelB) {
    const primaryMetric = modelA.problem_type === 'classification' ? 'Accuracy' : 'R² Score';
    const difference = comparisonData.score_difference || 0;
    const significanceLevel = getSignificanceLevel(difference);
    
    let winnerContent = '';
    if (comparisonData.winner === 'A') {
        winnerContent = generateWinnerContent(modelA, modelB, comparisonData, 'A', primaryMetric, significanceLevel);
    } else if (comparisonData.winner === 'B') {
        winnerContent = generateWinnerContent(modelB, modelA, comparisonData, 'B', primaryMetric, significanceLevel);
    } else {
        winnerContent = generateTieContent(modelA, modelB, comparisonData, primaryMetric);
    }

    return `
        <div class="winner-section enhanced-winner-section">
            ${winnerContent}
        </div>
    `;
}

function generateWinnerContent(winnerModel, loserModel, comparisonData, winner, primaryMetric, significanceLevel) {
    const winnerScore = winner === 'A' ? comparisonData.score_a : comparisonData.score_b;
    const loserScore = winner === 'A' ? comparisonData.score_b : comparisonData.score_a;
    
    return `
        <div class="winner-announcement enhanced-winner-announcement">
            <div class="winner-badge-clean enhanced-winner-badge">
                <i class="fas fa-trophy"></i>
                Winner: ${winnerModel.algorithm || winnerModel.id}
            </div>
            <div class="performance-summary enhanced-performance-summary">
                <span class="score-label enhanced-score-label">${primaryMetric} Performance</span>
                <div class="score-comparison enhanced-score-comparison">
                    <span class="loser-score enhanced-loser-score">${formatPercentage(loserScore)}</span>
                    <span class="vs-text enhanced-vs-text">vs</span>
                    <span class="winner-score enhanced-winner-score">${formatPercentage(winnerScore)}</span>
                </div>
                <div class="difference-indicator enhanced-difference-indicator ${significanceLevel.toLowerCase()}">
                    +${formatPercentage(comparisonData.score_difference)} (${significanceLevel})
                </div>
            </div>
        </div>
    `;
}

function generateTieContent(modelA, modelB, comparisonData, primaryMetric) {
    return `
        <div class="tie-announcement enhanced-tie-announcement">
            <div class="tie-badge-clean enhanced-tie-badge">
                <i class="fas fa-equals"></i>
                Performance Tie
            </div>
            <div class="performance-summary enhanced-performance-summary">
                <span class="score-label enhanced-score-label">${primaryMetric} Performance</span>
                <div class="score-comparison enhanced-score-comparison">
                    <span class="tie-score enhanced-tie-score">${formatPercentage(comparisonData.score_a)}</span>
                    <span class="vs-text enhanced-vs-text">≈</span>
                    <span class="tie-score enhanced-tie-score">${formatPercentage(comparisonData.score_b)}</span>
                </div>
                <div class="difference-indicator enhanced-difference-indicator marginal">
                    Δ${formatPercentage(comparisonData.score_difference)} (Marginal)
                </div>
            </div>
        </div>
    `;
}

// ===================== ENHANCED STYLING APPLICATION =====================
function applyEnhancedComparisonStyling(projectId, comparisonData) {
    console.log('[DEBUG] Applying enhanced styling...');
    
    // Add enhanced CSS dynamically
    addEnhancedComparisonCSS();
    
    // Apply winner styling to metric values
    const table = document.getElementById(`enhanced-metrics-table-${projectId}`);
    if (table) {
        const winnerCells = table.querySelectorAll('.enhanced-winner');
        const loserCells = table.querySelectorAll('.enhanced-loser');
        
        winnerCells.forEach(cell => {
            cell.style.cssText = `
                background: linear-gradient(135deg, #d1fae5, #a7f3d0) !important;
                color: #065f46 !important;
                border: 3px solid #10b981 !important;
                box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3) !important;
                transform: scale(1.1) !important;
                animation: glow 2s infinite !important;
                position: relative !important;
                font-weight: 900 !important;
                font-size: 1.25rem !important;
                text-align: center !important;
                padding: 1rem !important;
                border-radius: 12px !important;
                transition: all 0.4s ease !important;
            `;
            
            // Add trophy icon
            if (!cell.querySelector('.trophy-icon')) {
                const trophy = document.createElement('span');
                trophy.className = 'trophy-icon';
                trophy.innerHTML = '🏆';
                trophy.style.cssText = `
                    position: absolute;
                    right: 0.5rem;
                    top: 50%;
                    transform: translateY(-50%);
                    font-size: 1rem;
                    animation: sparkle 2s infinite;
                `;
                cell.appendChild(trophy);
            }
        });
        
        loserCells.forEach(cell => {
            cell.style.cssText = `
                background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important;
                color: #6b7280 !important;
                border: 2px solid #e2e8f0 !important;
                font-weight: 700 !important;
                font-size: 1.1rem !important;
                text-align: center !important;
                padding: 1rem !important;
                border-radius: 12px !important;
                transition: all 0.4s ease !important;
            `;
        });
    }
    
    // Apply enhanced styling to other elements
    const enhancedElements = document.querySelectorAll('.enhanced-comparison *');
    enhancedElements.forEach(element => {
        if (element.classList.contains('enhanced-winner-badge')) {
            element.style.animation = 'bounce 2s infinite';
        }
        if (element.classList.contains('enhanced-winner-score')) {
            element.style.animation = 'pulse 2s infinite';
        }
        if (element.classList.contains('enhanced-vs-circle')) {
            element.style.animation = 'pulse 3s infinite';
        }
    });
    
    console.log('[DEBUG] Enhanced styling applied successfully');
}

function addEnhancedComparisonCSS() {
    // Check if enhanced CSS is already added
    if (document.getElementById('enhanced-comparison-css')) return;
    
    const style = document.createElement('style');
    style.id = 'enhanced-comparison-css';
    style.textContent = `
        .enhanced-comparison {
            position: relative;
            z-index: 10;
        }
        
        .enhanced-metrics-table .enhanced-winner {
            background: linear-gradient(135deg, #d1fae5, #a7f3d0) !important;
            color: #065f46 !important;
            border: 3px solid #10b981 !important;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3) !important;
            transform: scale(1.1) !important;
            animation: glow 2s infinite !important;
            position: relative !important;
        }
        
        .enhanced-metrics-table .enhanced-winner::after {
            content: '🏆';
            position: absolute;
            right: 0.5rem;
            top: 50%;
            transform: translateY(-50%);
            font-size: 1rem;
            animation: sparkle 2s infinite;
        }
        
        .enhanced-metrics-table .enhanced-loser {
            background: linear-gradient(135deg, #f8fafc, #e2e8f0) !important;
            color: #6b7280 !important;
            border: 2px solid #e2e8f0 !important;
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3); }
            50% { box-shadow: 0 12px 32px rgba(16, 185, 129, 0.5); }
        }
        
        @keyframes sparkle {
            0%, 100% { opacity: 1; transform: translateY(-50%) scale(1); }
            50% { opacity: 0.7; transform: translateY(-50%) scale(1.3); }
        }
        
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-10px); }
            60% { transform: translateY(-5px); }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.05); opacity: 0.8; }
        }
        
        .enhanced-winner-badge {
            animation: bounce 2s infinite !important;
        }
        
        .enhanced-winner-score {
            animation: pulse 2s infinite !important;
        }
        
        .enhanced-vs-circle {
            animation: pulse 3s infinite !important;
        }
    `;
    document.head.appendChild(style);
}

function addEnhancedInteractivity() {
    // Add click-to-copy for model IDs
    document.querySelectorAll('.enhanced-id').forEach(element => {
        element.addEventListener('click', function() {
            navigator.clipboard.writeText(this.textContent).then(() => {
                showNotification('Model ID copied to clipboard!', 'success');
            });
        });
        element.style.cursor = 'pointer';
    });
    
    // Enhanced hover effects for winner metrics
    document.querySelectorAll('.enhanced-winner').forEach(element => {
        element.addEventListener('mouseenter', function() {
            this.style.transform = 'scale(1.15) !important';
            this.style.zIndex = '100';
        });
        
        element.addEventListener('mouseleave', function() {
            this.style.transform = 'scale(1.1) !important';
            this.style.zIndex = '10';
        });
    });
}

// ===================== UTILITY FUNCTIONS =====================
function getEssentialMetrics(problemType) {
    const essentialMetricsMap = {
        'classification': ['accuracy', 'precision', 'recall', 'f1'],
        'regression': ['r2', 'rmse', 'mae'],
        'clustering': ['silhouette_score', 'n_clusters', 'calinski_harabasz_score'],
        'anomaly': ['outlier_ratio', 'anomalies_detected'],
        'anomaly_detection': ['outlier_ratio', 'anomalies_detected'],
        'dimensionality_reduction': ['variance_explained', 'kl_divergence', 'n_components'],
        'association': ['rules_count', 'max_lift', 'frequent_itemsets_count'],
        'association_rules': ['rules_count', 'max_lift', 'frequent_itemsets_count'],
        'unsupervised': ['completed', 'n_clusters', 'outlier_ratio']
    };
    return essentialMetricsMap[problemType] || ['accuracy', 'precision', 'recall', 'f1'];
}

function formatMetricValue(value, metricName) {
    if (value === undefined || value === null) return '—';
    
    // Handle different metric types
    if (metricName === 'completed') {
        return '✓ Done';
    }
    
    if (typeof value === 'number') {
        // Percentage metrics (0-1 scale)
        if (['accuracy', 'precision', 'recall', 'f1', 'r2', 'silhouette_score', 'variance_explained', 'outlier_ratio'].includes(metricName)) {
            return (value * 100).toFixed(1) + '%';
        }
        // Count metrics
        if (['n_clusters', 'anomalies_detected', 'rules_count', 'frequent_itemsets_count', 'n_components'].includes(metricName)) {
            return Math.round(value).toString();
        }
        // Decimal metrics
        return value.toFixed(3);
    }
    return value.toString();
}

function formatMetricName(metric) {
    const nameMap = {
        'accuracy': 'Accuracy',
        'precision': 'Precision', 
        'recall': 'Recall',
        'f1': 'F1 Score',
        'r2': 'R² Score',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'silhouette_score': 'Silhouette Score',
        'n_clusters': 'Clusters Found',
        'calinski_harabasz_score': 'Calinski-Harabasz Score',
        'davies_bouldin_score': 'Davies-Bouldin Score',
        'outlier_ratio': 'Outlier Ratio',
        'anomalies_detected': 'Anomalies Found',
        'variance_explained': 'Variance Explained',
        'kl_divergence': 'KL Divergence',
        'n_components': 'Components',
        'rules_count': 'Rules Found',
        'max_lift': 'Max Lift',
        'frequent_itemsets_count': 'Frequent Itemsets',
        'completed': 'Status'
    };
    return nameMap[metric] || metric.charAt(0).toUpperCase() + metric.slice(1).replace(/_/g, ' ');
}

function compareMetricValues(metric, valueA, valueB) {
    if (valueA === undefined || valueA === null || valueB === undefined || valueB === null) {
        return 'tie';
    }
    
    const higherIsBetter = [
        'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc', 'r2',
        'silhouette_score', 'calinski_harabasz_score', 'variance_explained',
        'n_clusters', 'anomalies_detected', 'rules_count', 'frequent_itemsets_count',
        'max_lift', 'n_components'
    ];
    const lowerIsBetter = ['rmse', 'mae', 'mse', 'log_loss', 'davies_bouldin_score', 'kl_divergence'];
    const specialCases = ['completed', 'outlier_ratio']; // Context-dependent
    
    const diff = Math.abs(valueA - valueB);
    if (diff < 0.001) return 'tie';
    
    if (higherIsBetter.includes(metric)) {
        return valueA > valueB ? 'A' : 'B';
    } else if (lowerIsBetter.includes(metric)) {
        return valueA < valueB ? 'A' : 'B';
    } else if (specialCases.includes(metric)) {
        // For these, we'll default to higher is better
        return valueA > valueB ? 'A' : 'B';
    } else {
        // Default case
        return valueA > valueB ? 'A' : 'B';
    }
}

function truncateModelId(modelId) {
    if (!modelId) return 'Unknown';
    if (modelId.length > 30) {
        return modelId.substring(0, 15) + '...' + modelId.substring(modelId.length - 10);
    }
    return modelId;
}

function formatDate(dateString) {
    if (!dateString || dateString === 'Unknown') return 'Unknown';
    try {
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            month: 'short',
            day: 'numeric'
        });
    } catch (e) {
        return 'Unknown';
    }
}

function formatPercentage(value) {
    if (value === undefined || value === null) return '—';
    return (value * 100).toFixed(1) + '%';
}

function getSignificanceLevel(difference) {
    if (difference > 0.15) return 'Significant';
    if (difference > 0.08) return 'Moderate'; 
    if (difference > 0.03) return 'Marginal';
    return 'Minimal';
}

// ===================== DOWNLOAD FUNCTIONS =====================
function downloadBestModel(projectId, modelId) {
    console.log('[DEBUG] Downloading best model:', projectId, modelId);
    showNotification('Downloading best model (.pkl)...', 'info');
    
    const downloadUrl = `/xai/api/models/${modelId}/download/best`;
    
    fetch(downloadUrl, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                window.location.href = downloadUrl;
                showNotification('Download initiated successfully', 'success');
            } else {
                throw new Error(`Server responded with status: ${response.status}`);
            }
        })
        .catch(error => {
            console.error('[ERROR] Download failed:', error);
            showNotification('Download failed: ' + error.message, 'error');
        });
}

function downloadAllModels(projectId, modelId) {
    console.log('[DEBUG] Downloading all models:', projectId, modelId);
    showNotification('Downloading all models (.zip)...', 'info');
    
    const downloadUrl = `/xai/api/models/${modelId}/download/all`;
    
    fetch(downloadUrl, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                window.location.href = downloadUrl;
                showNotification('Download initiated successfully', 'success');
            } else {
                throw new Error(`Server responded with status: ${response.status}`);
            }
        })
        .catch(error => {
            console.error('[ERROR] Download failed:', error);
            showNotification('Download failed: ' + error.message, 'error');
        });
}

function downloadResults(projectId, modelId) {
    console.log('[DEBUG] Downloading results:', projectId, modelId);
    showNotification('Downloading results (.json)...', 'info');
    
    const downloadUrl = `/xai/api/models/${modelId}/download/results`;
    
    fetch(downloadUrl, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                window.location.href = downloadUrl;
                showNotification('Download initiated successfully', 'success');
            } else {
                throw new Error(`Server responded with status: ${response.status}`);
            }
        })
        .catch(error => {
            console.error('[ERROR] Download failed:', error);
            showNotification('Download failed: ' + error.message, 'error');
        });
}

function downloadModel(modelId) {
    console.log('[DEBUG] Legacy downloadModel called:', modelId);
    showNotification('Downloading model (.pkl)...', 'info');

    const downloadUrl = `/xai/api/models/${modelId}/download/best`;

    fetch(downloadUrl, { method: 'HEAD' })
        .then(response => {
            if (response.ok) {
                window.location.href = downloadUrl;
                showNotification('Download initiated successfully', 'success');
            } else {
                throw new Error(`Server responded with status: ${response.status}`);
            }
        })
        .catch(error => {
            console.error('[ERROR] Download failed:', error);
            showNotification('Download failed: ' + error.message, 'error');
        });
}

function deleteModel(projectId, modelId) {
    if (!confirm(`Are you sure you want to delete model "${modelId}"? This action cannot be undone.`)) {
        return;
    }
    
    showLoadingOverlay('Deleting model...');
    
    fetch(`/xai/api/models/${modelId}`, { method: 'DELETE' })
    .then(response => response.json())
    .then(data => {
        hideLoadingOverlay();
        if (data.success) {
            const modelCard = document.querySelector(`[data-model-id="${modelId}"]`);
            if (modelCard) {
                modelCard.style.animation = 'fadeOut 0.3s ease-out';
                setTimeout(() => {
                    modelCard.remove();
                    const remainingModels = document.querySelectorAll(`[data-project-id="${projectId}"] .model-card`);
                    if (remainingModels.length === 0) {
                        location.reload();
                    }
                }, 300);
            }
            showNotification('Model deleted successfully', 'success');
        } else {
            showNotification('Failed to delete model: ' + data.message, 'error');
        }
    })
    .catch(error => {
        hideLoadingOverlay();
        showNotification('Error deleting model: ' + error.message, 'error');
    });
}

// ===================== HELPER FUNCTIONS =====================
function showLoadingOverlay(message = 'Loading...') {
    const existing = document.getElementById('loading-overlay');
    if (existing) existing.remove();
    
    const overlay = document.createElement('div');
    overlay.id = 'loading-overlay';
    overlay.className = 'loading-overlay';
    overlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.7);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        backdrop-filter: blur(4px);
    `;
    overlay.innerHTML = `
        <div class="loading-content" style="
            background: white;
            padding: 2rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        ">
            <div class="loading-spinner" style="
                width: 48px;
                height: 48px;
                border: 4px solid #e5e7eb;
                border-top: 4px solid #6366f1;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 1rem;
            "></div>
            <div class="loading-text" style="
                color: #374151;
                font-weight: 600;
                font-size: 1.1rem;
            ">${message}</div>
        </div>
    `;
    document.body.appendChild(overlay);
    
    // Add spinner animation
    const spinnerStyle = document.createElement('style');
    spinnerStyle.textContent = `
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    `;
    document.head.appendChild(spinnerStyle);
}

function hideLoadingOverlay() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.remove();
        }, 300);
    }
}

function showNotification(message, type = 'info') {
    console.log(`[NOTIFICATION] ${type.toUpperCase()}: ${message}`);
    
    document.querySelectorAll('.notification').forEach(n => n.remove());
    
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 2rem;
        border-radius: 12px;
        color: white;
        font-weight: 600;
        z-index: 10000;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transform: translateX(100%);
        transition: transform 0.3s ease;
        max-width: 400px;
        word-wrap: break-word;
    `;
    
    const icon = getNotificationIcon(type);
    const bgColor = getNotificationColor(type);
    notification.style.background = bgColor;
    
    notification.innerHTML = `
        <i class="fas fa-${icon}"></i>
        <span>${message}</span>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
    }, 100);
    
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, 400);
    }, 4000);
}

function getNotificationIcon(type) {
    switch (type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        case 'info': 
        default: return 'info-circle';
    }
}

function getNotificationColor(type) {
    switch (type) {
        case 'success': return 'linear-gradient(135deg, #10b981, #059669)';
        case 'error': return 'linear-gradient(135deg, #ef4444, #dc2626)';
        case 'warning': return 'linear-gradient(135deg, #f59e0b, #d97706)';
        case 'info': 
        default: return 'linear-gradient(135deg, #3b82f6, #1d4ed8)';
    }
}

function closeModal() {
    document.querySelectorAll('.modal-overlay').forEach(modal => {
        modal.remove();
    });
}

// ===================== PROJECT SEARCH =====================
function initializeProjectSearch() {
    const searchInput = document.getElementById('project-search');
    if (!searchInput) return;
    
    console.log('[DEBUG] Initializing project search');
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase().trim();
        const projectCards = document.querySelectorAll('.project-card');
        
        projectCards.forEach(card => {
            const projectName = card.querySelector('.project-title').textContent.toLowerCase();
            const projectDescription = card.querySelector('.project-description').textContent.toLowerCase();
            const modelCount = card.querySelector('.model-count').textContent.toLowerCase();
            
            const isMatch = projectName.includes(searchTerm) || 
                           projectDescription.includes(searchTerm) || 
                           modelCount.includes(searchTerm);
            
            if (isMatch || searchTerm === '') {
                card.style.display = 'block';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            } else {
                card.style.opacity = '0';
                card.style.transform = 'translateY(-10px)';
                setTimeout(() => {
                    if (!card.querySelector('.project-title').textContent.toLowerCase().includes(searchInput.value.toLowerCase())) {
                        card.style.display = 'none';
                    }
                }, 200);
            }
        });
        
        const visibleProjects = Array.from(projectCards).filter(card => 
            card.style.display !== 'none'
        );
        
        const existingNoResults = document.querySelector('.no-search-results');
        if (visibleProjects.length === 0 && searchTerm !== '') {
            if (!existingNoResults) {
                const noResultsDiv = document.createElement('div');
                noResultsDiv.className = 'no-search-results';
                noResultsDiv.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-search"></i>
                        <h3>No Projects Found</h3>
                        <p>No projects match your search for "${searchTerm}"</p>
                        <button class="btn-primary" onclick="document.getElementById('project-search').value = ''; document.getElementById('project-search').dispatchEvent(new Event('input'));">
                            <i class="fas fa-times"></i> Clear Search
                        </button>
                    </div>
                `;
                document.querySelector('.projects-container').appendChild(noResultsDiv);
            }
        } else if (existingNoResults) {
            existingNoResults.remove();
        }
    });
    
    document.addEventListener('keydown', function(event) {
        if ((event.ctrlKey || event.metaKey) && event.key === 'f') {
            event.preventDefault();
            searchInput.focus();
            searchInput.select();
        }
        
        if (event.key === 'Escape' && document.activeElement === searchInput) {
            searchInput.value = '';
            searchInput.dispatchEvent(new Event('input'));
            searchInput.blur();
        }
    });
}

// ===================== INITIALIZATION =====================
document.addEventListener('DOMContentLoaded', function() {
    console.log('[DEBUG] Enhanced XAI page loaded');
    
    if (document.getElementById('project-search')) {
        initializeProjectSearch();
    }
    
    // Make functions globally available for debugging
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        console.log('[DEBUG] Development mode detected');
        window.analyzeModel = analyzeModel;
        window.downloadBestModel = downloadBestModel;
        window.downloadAllModels = downloadAllModels;
        window.downloadResults = downloadResults;
        window.updateProjectComparison = updateProjectComparison;
        window.enableProjectComparison = enableProjectComparison;
        window.hideProjectComparison = hideProjectComparison;
    }
});

// Close modal on Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});

// Close dropdown when clicking outside
document.addEventListener('click', function(event) {
    if (!event.target.closest('.download-dropdown')) {
        document.querySelectorAll('.download-menu').forEach(menu => {
            menu.classList.remove('show');
        });
        
        const backdrop = document.querySelector('.download-menu-backdrop');
        if (backdrop) {
            backdrop.remove();
        }
    }
});