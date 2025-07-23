// Global variables
let analysisData = null;
let modelId = window.modelId || null;

// Chart.js chart instances
let liftChart = null, rocChart = null, prChart = null, residualChart = null, qqChart = null, confusionMatrixChart = null;

// ===================== MAIN LOADING FUNCTION =====================

async function loadAnalysis(id) {
    modelId = id || modelId;
    showLoadingState();

    try {
        const apiUrl = `/xai/api/${modelId}/data`;
        const response = await fetch(apiUrl, { method: 'GET', headers: { 'Accept': 'application/json' } });
        if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        const data = await response.json();
        if (!data.success) throw new Error(data.message || 'API returned error');
        analysisData = data.data;

        // Fill header info
        document.getElementById('model-name').textContent = `Model: ${analysisData.model_id || 'N/A'}`;
        document.getElementById('created-date').textContent = `Created: ${analysisData.created_at || 'N/A'}`;
        document.getElementById('problem-type').textContent = analysisData.problem_type || 'N/A';

        // Show/hide sections based on problem type
        if (analysisData.problem_type && analysisData.problem_type.toLowerCase().includes('regression')) {
            showSection('regression-section');
            hideSection('classification-section');
            fillRegressionMetrics(analysisData);
        } else {
            showSection('classification-section');
            hideSection('regression-section');
            fillClassificationMetrics(analysisData);
        }

        // Show lift/cumulative gain chart if available
        if (analysisData.lift_chart && analysisData.lift_chart.percent_samples && analysisData.lift_chart.percent_samples.length > 0) {
            showSection('lift-section');
            plotLiftChart(analysisData.lift_chart);
        } else {
            hideSection('lift-section');
        }

        // Show XAI section if available
        if (analysisData.xai) showSection('xai-section');
        else hideSection('xai-section');

        showAnalysisContent();
    } catch (error) {
        showErrorState(error.message);
    }
}

// ===================== SECTION TOGGLING =====================

function showSection(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = '';
}
function hideSection(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = 'none';
}
function showLoadingState() {
    // Optionally implement a loading spinner
}
function showAnalysisContent() {
    // Optionally implement content fade-in
}
function showErrorState(msg) {
    alert("Analysis Error: " + msg);
}

// ===================== REGRESSION METRICS =====================

function fillRegressionMetrics(data) {
    const m = data.metrics || {};
    setMetric('mae-value', m.mae);
    setMetric('mse-value', m.mse);
    setMetric('rmse-value', m.rmse);
    setMetric('r2-value', m.r2);
    setMetric('adjr2-value', m.adjusted_r2);
    setMetric('mape-value', m.mape);
    setMetric('medae-value', m.median_ae);
    setMetric('explvar-value', m.explained_variance);

    // Residual plot
    if (data.y_pred && data.residuals) plotResiduals(data.y_pred, data.residuals);
    // QQ plot
    if (data.residuals) plotQQ(data.residuals);
}
// ===================== CLASSIFICATION METRICS =====================

function fillClassificationMetrics(data) {
    const m = data.metrics || {};
    setMetric('accuracy-value', m.accuracy);
    setMetric('precision-value', m.precision);
    setMetric('recall-value', m.recall);
    setMetric('f1-value', m.f1);
    setMetric('mcc-value', m.mcc);
    setMetric('kappa-value', m.cohen_kappa);
    setMetric('logloss-value', m.log_loss);
    setMetric('brier-value', m.brier_score);

    if (data.roc_curve && data.roc_curve.auc !== undefined) {
        setMetric('roc-auc-value', data.roc_curve.auc);
    }
    if (data.pr_curve && data.pr_curve.auc !== undefined) {
        setMetric('pr-auc-value', data.pr_curve.auc);
    }

    // Confusion matrix
    if (data.confusion_matrix) {
        // Try to get class labels from backend
        let labels = null;
        if (data.class_labels) labels = data.class_labels;
        plotConfusionMatrix(data.confusion_matrix, labels);
    }
    // ROC curve
    if (data.roc_curve) plotROCCurve(data.roc_curve);
    // PR curve
    if (data.pr_curve) plotPRCurve(data.pr_curve);

    // Classification report
    if (m.classification_report) {
        showSection('classification-report-section');
        renderClassificationReport(m.classification_report);
    } else {
        hideSection('classification-report-section');
    }
}

// Helper to render the classification report as a table
function renderClassificationReport(report) {
    const container = document.getElementById('classification-report-table');
    if (!container) return;
    let html = '<table class="preview-table"><thead><tr><th>Class</th>';
    const keys = Object.keys(report[Object.keys(report)[0]]);
    for (const k of keys) html += `<th>${k}</th>`;
    html += '</tr></thead><tbody>';
    for (const label in report) {
        if (typeof report[label] !== 'object') continue;
        html += `<tr><td>${label}</td>`;
        for (const k of keys) {
            let val = report[label][k];
            if (typeof val === 'number') val = val.toFixed(3);
            html += `<td>${val}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;
}

// ===================== LIFT CHART =====================

function plotLiftChart(liftData) {
    if (!liftData || !liftData.percent_samples || !liftData.percent_positives) return;
    const ctx = document.getElementById('lift-chart').getContext('2d');
    if (liftChart) liftChart.destroy();
    liftChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: liftData.percent_samples,
            datasets: [
                { label: 'Model', data: liftData.percent_positives, borderColor: '#4f46e5', fill: false },
                { label: 'Random', data: liftData.percent_samples, borderColor: '#aaa', borderDash: [5,5], fill: false }
            ]
        },
        options: {
            plugins: { legend: { display: true } },
            scales: { x: { title: { display: true, text: '% of Samples' } }, y: { title: { display: true, text: '% of Positives Captured' } } }
        }
    });
}

// ===================== REGRESSION PLOTS =====================

function plotResiduals(y_pred, residuals) {
    const ctx = document.getElementById('residual-plot').getContext('2d');
    if (residualChart) residualChart.destroy();
    residualChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Residuals',
                data: y_pred.map((x, i) => ({ x, y: residuals[i] })),
                backgroundColor: '#4f46e5'
            }]
        },
        options: {
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Predicted' } },
                y: { title: { display: true, text: 'Residuals' } }
            }
        }
    });
}

function plotQQ(residuals) {
    // For demo: plot sorted residuals vs normal quantiles
    const ctx = document.getElementById('qq-plot').getContext('2d');
    if (qqChart) qqChart.destroy();
    const sorted = [...residuals].sort((a, b) => a - b);
    const n = sorted.length;
    // jStat is required for normal quantiles
    const quantiles = Array.from({length: n}, (_, i) => jStat.normal.inv((i + 0.5) / n, 0, 1));
    qqChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'QQ Plot',
                data: quantiles.map((q, i) => ({ x: q, y: sorted[i] })),
                backgroundColor: '#4f46e5'
            }]
        },
        options: {
            plugins: { legend: { display: false } },
            scales: {
                x: { title: { display: true, text: 'Theoretical Quantiles' } },
                y: { title: { display: true, text: 'Sample Quantiles' } }
            }
        }
    });
}

// ===================== CLASSIFICATION PLOTS =====================

function plotConfusionMatrix(cm, labels) {
    // cm: 2D array (matrix), labels: array of class names (optional)
    if (!labels && analysisData && analysisData.metrics && analysisData.metrics.classification_report) {
        labels = Object.keys(analysisData.metrics.classification_report)
            .filter(k => !['accuracy', 'macro avg', 'weighted avg'].includes(k));
    }
    if (!labels || labels.length !== cm.length) {
        labels = Array.from({length: cm.length}, (_, i) => `${i}`);
    }

    const container = document.getElementById('confusion-matrix-table');
    if (!container) return;

    // Find min/max for color scaling
    let min = Math.min(...cm.flat());
    let max = Math.max(...cm.flat());

    // Matplotlib Blues colormap function
    function getMatplotlibBlue(value, min, max) {
        if (min === max) return '#f7fbff';
        const normalized = (value - min) / (max - min);
        
        // Clean blues progression
        if (normalized < 0.1) return '#f7fbff';
        if (normalized < 0.2) return '#deebf7';
        if (normalized < 0.3) return '#c6dbef';
        if (normalized < 0.4) return '#9ecae1';
        if (normalized < 0.5) return '#6baed6';
        if (normalized < 0.6) return '#4292c6';
        if (normalized < 0.7) return '#2171b5';
        if (normalized < 0.8) return '#08519c';
        return '#08306b';
    }

    const n = cm.length;
    
    // Create clean HTML table layout
    let html = `
        <div class="cm-matplotlib-container">
            <div class="cm-matplotlib-title">Confusion Matrix</div>
            
            <div class="cm-matplotlib-content">
                <div class="cm-matplotlib-main">
                    <!-- Predicted label -->
                    <div class="cm-predicted-label">Predicted label</div>
                    
                    <!-- Matrix table -->
                    <div class="cm-table-wrapper">
                        <table class="cm-matplotlib-table">
                            <thead>
                                <tr>
                                    <th class="cm-corner-cell"></th>
                                    ${labels.map(label => `<th class="cm-header-cell">${label}</th>`).join('')}
                                </tr>
                            </thead>
                            <tbody>
                                ${cm.map((row, i) => `
                                    <tr>
                                        <th class="cm-row-header">${labels[i]}</th>
                                        ${row.map((value) => {
                                            const bgColor = getMatplotlibBlue(value, min, max);
                                            const textColor = value > (max - min) * 0.5 + min ? '#ffffff' : '#000000';
                                            return `<td class="cm-data-cell" style="background-color: ${bgColor}; color: ${textColor};">
                                                        <span class="cm-cell-value">${value}</span>
                                                    </td>`;
                                        }).join('')}
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                        
                        <!-- True label (rotated) -->
                        <div class="cm-true-label">True label</div>
                    </div>
                </div>
                
                <!-- Colorbar -->
                <div class="cm-colorbar-container">
                    <div class="cm-colorbar-wrapper">
                        <div class="cm-colorbar-scale">
                            ${Array.from({length: 50}, (_, i) => {
                                const value = min + (max - min) * ((49 - i) / 49); // Reverse order for top-to-bottom
                                const color = getMatplotlibBlue(value, min, max);
                                return `<div class="cm-colorbar-segment" style="background-color: ${color};"></div>`;
                            }).join('')}
                        </div>
                        <div class="cm-colorbar-labels">
                            <span class="cm-colorbar-max">${max}</span>
                            <span class="cm-colorbar-min">${min}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    container.innerHTML = html;
}

function plotROCCurve(roc) {
    const ctx = document.getElementById('roc-chart').getContext('2d');
    if (rocChart) rocChart.destroy();
    
    // Calculate AUC if not provided
    const auc = roc.auc || calculateAUC(roc.fpr, roc.tpr);
    
    rocChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `ROC curve (AUC = ${auc.toFixed(2)})`,
                    data: roc.fpr.map((fpr, i) => ({ x: fpr, y: roc.tpr[i] })),
                    borderColor: '#1f77b4', // Matplotlib default blue
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0,
                    order: 1
                },
                {
                    label: '',
                    data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
                    borderColor: '#000000',
                    backgroundColor: 'transparent',
                    borderWidth: 1,
                    borderDash: [8, 4],
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 0,
                    tension: 0,
                    order: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 0.85,
            layout: {
                padding: {
                    left: 10,
                    right: 10,
                    top: 10,
                    bottom: 10
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'ROC Curve',
                    font: {
                        size: 14,
                        weight: 'normal',
                        family: 'DejaVu Sans, Arial, sans-serif'
                    },
                    color: '#000000',
                    padding: {
                        top: 5,
                        bottom: 15
                    }
                },
                legend: {
                    display: true,
                    position: 'upper right',
                    align: 'start',
                    labels: {
                        color: '#000000',
                        font: {
                            size: 10,
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        padding: 8,
                        usePointStyle: false,
                        boxWidth: 20,
                        boxHeight: 2,
                        filter: function(legendItem, chartData) {
                            return legendItem.text !== '';
                        }
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(255, 255, 225, 0.9)',
                    titleColor: '#000',
                    bodyColor: '#000',
                    borderColor: '#666',
                    borderWidth: 1,
                    titleFont: {
                        size: 10
                    },
                    bodyFont: {
                        size: 10
                    },
                    displayColors: false,
                    callbacks: {
                        title: function() { return ''; },
                        label: function(context) {
                            return `(${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'False Positive Rate',
                        font: {
                            size: 11,
                            weight: 'normal',
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        color: '#000000',
                        padding: {
                            top: 8
                        }
                    },
                    ticks: {
                        color: '#000000',
                        font: {
                            size: 9,
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        stepSize: 0.2,
                        padding: 5,
                        callback: function(value) {
                            return Number(value).toFixed(1);
                        }
                    },
                    grid: {
                        color: '#b0b0b0',
                        lineWidth: 0.8,
                        drawTicks: true,
                        tickLength: 4
                    },
                    border: {
                        color: '#000000',
                        width: 0.8
                    }
                },
                y: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'True Positive Rate',
                        font: {
                            size: 11,
                            weight: 'normal',
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        color: '#000000',
                        padding: {
                            bottom: 8
                        }
                    },
                    ticks: {
                        color: '#000000',
                        font: {
                            size: 9,
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        stepSize: 0.2,
                        padding: 5,
                        callback: function(value) {
                            return Number(value).toFixed(1);
                        }
                    },
                    grid: {
                        color: '#b0b0b0',
                        lineWidth: 0.8,
                        drawTicks: true,
                        tickLength: 4
                    },
                    border: {
                        color: '#000000',
                        width: 0.8
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'nearest'
            }
        }
    });
}

// Helper function to calculate AUC using trapezoidal rule
function calculateAUC(fpr, tpr) {
    let auc = 0;
    for (let i = 1; i < fpr.length; i++) {
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2;
    }
    return auc;
}

function plotPRCurve(pr) {
    const ctx = document.getElementById('pr-curve').getContext('2d');
    if (prChart) prChart.destroy();
    
    // Calculate AUC if not provided
    const auc = pr.auc || calculatePRAUC(pr.recall, pr.precision);
    
    prChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [
                {
                    label: `Precision-Recall curve (AUC = ${auc.toFixed(2)})`,
                    data: pr.recall.map((recall, i) => ({ x: recall, y: pr.precision[i] })),
                    borderColor: '#ff7f0e', // Matplotlib default orange
                    backgroundColor: 'transparent',
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 3,
                    tension: 0,
                    order: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            aspectRatio: 0.85,
            layout: {
                padding: {
                    left: 10,
                    right: 10,
                    top: 10,
                    bottom: 10
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: 'Precision-Recall Curve',
                    font: {
                        size: 14,
                        weight: 'normal',
                        family: 'DejaVu Sans, Arial, sans-serif'
                    },
                    color: '#000000',
                    padding: {
                        top: 5,
                        bottom: 15
                    }
                },
                legend: {
                    display: true,
                    position: 'upper right',
                    align: 'start',
                    labels: {
                        color: '#000000',
                        font: {
                            size: 10,
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        padding: 8,
                        usePointStyle: false,
                        boxWidth: 20,
                        boxHeight: 2
                    }
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: 'rgba(255, 255, 225, 0.9)',
                    titleColor: '#000',
                    bodyColor: '#000',
                    borderColor: '#666',
                    borderWidth: 1,
                    titleFont: {
                        size: 10
                    },
                    bodyFont: {
                        size: 10
                    },
                    displayColors: false,
                    callbacks: {
                        title: function() { return ''; },
                        label: function(context) {
                            return `(${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    position: 'bottom',
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Recall',
                        font: {
                            size: 11,
                            weight: 'normal',
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        color: '#000000',
                        padding: {
                            top: 8
                        }
                    },
                    ticks: {
                        color: '#000000',
                        font: {
                            size: 9,
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        stepSize: 0.2,
                        padding: 5,
                        callback: function(value) {
                            return Number(value).toFixed(1);
                        }
                    },
                    grid: {
                        color: '#b0b0b0',
                        lineWidth: 0.8,
                        drawTicks: true,
                        tickLength: 4
                    },
                    border: {
                        color: '#000000',
                        width: 0.8
                    }
                },
                y: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Precision',
                        font: {
                            size: 11,
                            weight: 'normal',
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        color: '#000000',
                        padding: {
                            bottom: 8
                        }
                    },
                    ticks: {
                        color: '#000000',
                        font: {
                            size: 9,
                            family: 'DejaVu Sans, Arial, sans-serif'
                        },
                        stepSize: 0.2,
                        padding: 5,
                        callback: function(value) {
                            return Number(value).toFixed(1);
                        }
                    },
                    grid: {
                        color: '#b0b0b0',
                        lineWidth: 0.8,
                        drawTicks: true,
                        tickLength: 4
                    },
                    border: {
                        color: '#000000',
                        width: 0.8
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'nearest'
            }
        }
    });
}

// Helper function to calculate PR AUC using trapezoidal rule
function calculatePRAUC(recall, precision) {
    let auc = 0;
    for (let i = 1; i < recall.length; i++) {
        auc += (recall[i] - recall[i-1]) * (precision[i] + precision[i-1]) / 2;
    }
    return Math.abs(auc); // Ensure positive AUC
}

// ===================== UTILS =====================

function setMetric(id, value) {
    const el = document.getElementById(id);
    if (el) {
        if (typeof value === "number") {
            el.textContent = value.toFixed(3);
        } else if (value !== undefined && value !== null) {
            el.textContent = value;
        } else {
            el.textContent = '--';
        }
    }
}

// ===================== EXPORT BUTTONS =====================

function exportToJSON() {
    if (!analysisData) return;
    const blob = new Blob([JSON.stringify(analysisData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `analysis_${modelId}.json`; a.click();
    URL.revokeObjectURL(url);
}
function exportToPDF() {
    window.print();
}
function printAnalysis() {
    window.print();
}

// ===================== INIT =====================

window.addEventListener('DOMContentLoaded', () => {
    loadAnalysis(modelId);
});