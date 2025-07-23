// Get data passed from template
const pageData = window.pageData || {};
const currentDomain = pageData.currentDomain || 'connection';
let currentCorrelations = null;

console.log('Correlation analysis loaded with domain:', currentDomain);

// Handle correlation form submission
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('correlation-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const columns = Array.from(document.querySelectorAll('.corr-column-checkbox:checked')).map(cb => cb.value);
            const method = document.getElementById('method-select').value;
            const messageDiv = document.getElementById('correlation-message');
            const submitBtn = this.querySelector('button[type="submit"]');
            
            console.log('Correlation form submitted with:', { columns, method });
            
            if (columns.length < 2) {
                messageDiv.innerHTML = '❌ Please select at least 2 columns.';
                messageDiv.style.color = '#dc2626';
                return;
            }
            
            // Show loading state
            messageDiv.innerHTML = '🔄 Computing correlations...';
            messageDiv.style.color = '#007bff';
            submitBtn.disabled = true;
            submitBtn.textContent = 'Computing...';
            
            fetch(`/${currentDomain}/compute_correlations`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    columns: columns,
                    method: method
                })
            })
            .then(response => {
                console.log('Correlation response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Correlation data received:', data);
                submitBtn.disabled = false;
                submitBtn.textContent = '📊 Compute Correlations';
                
                if (data.success) {
                    currentCorrelations = data;
                    showResults(data);
                    messageDiv.innerHTML = '✅ Correlation analysis completed!';
                    messageDiv.style.color = '#10b981';
                } else {
                    messageDiv.innerHTML = `❌ ${data.message}`;
                    messageDiv.style.color = '#dc2626';
                    hideResults();
                }
            })
            .catch(error => {
                console.error('Correlation error:', error);
                submitBtn.disabled = false;
                submitBtn.textContent = '📊 Compute Correlations';
                messageDiv.innerHTML = `❌ Error: ${error.message}`;
                messageDiv.style.color = '#dc2626';
                hideResults();
            });
        });
    }
});

function showResults(data) {
    const resultsSection = document.getElementById('results-section');
    const methodInfo = document.getElementById('method-info');
    const matrixTable = document.getElementById('matrix-table');
    const heatmapDiv = document.getElementById('heatmap');
    
    if (resultsSection) resultsSection.style.display = 'block';
    
    // Set method description
    const methodDescriptions = {
        'pearson': 'Pearson correlation measures linear relationships between variables. It assumes normally distributed data and is sensitive to outliers.',
        'spearman': 'Spearman correlation measures monotonic relationships (whether linear or not) using ranked data. It is non-parametric and works with ordinal data.',
        'kendall': 'Kendall correlation measures ordinal association between variables. It is robust with small datasets and many tied ranks.'
    };
    
    if (document.getElementById('method-description')) {
        document.getElementById('method-description').innerHTML = `
            <strong>Method:</strong> ${data.method.toUpperCase()}<br>
            ${methodDescriptions[data.method.toLowerCase()] || ''}
        `;
    }
    
    // Create correlation matrix table
    const columns = Object.keys(data.correlations);
    let html = '<thead><tr><th></th>';
    
    // Header row
    columns.forEach(col => {
        html += `<th style="padding:8px 12px; background:#f8f9fa; border:1px solid #e2e8f0; text-align:center;">${col}</th>`;
    });
    html += '</tr></thead><tbody>';
    
    // Data rows
    columns.forEach(col1 => {
        html += `<tr><td style="padding:8px 12px; background:#f8f9fa; border:1px solid #e2e8f0; font-weight:500;">${col1}</td>`;
        columns.forEach(col2 => {
            const value = data.correlations[col1][col2];
            const absValue = Math.abs(value);
            let bgColor = '#ffffff';
            let textColor = '#333';
            
            if (absValue >= 0.8) {
                bgColor = value > 0 ? 'rgba(0,100,0,0.3)' : 'rgba(139,0,0,0.3)';
            } else if (absValue >= 0.6) {
                bgColor = value > 0 ? 'rgba(0,128,0,0.2)' : 'rgba(178,34,34,0.2)';
            } else if (absValue >= 0.4) {
                bgColor = value > 0 ? 'rgba(144,238,144,0.2)' : 'rgba(255,182,193,0.2)';
            }
            
            if (col1 === col2) {
                bgColor = '#f8f9fa';
                textColor = '#999';
            }
            
            html += `<td style="padding:8px 12px; border:1px solid #e2e8f0; text-align:center; background:${bgColor}; color:${textColor};">${value.toFixed(3)}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody>';
    
    if (matrixTable) matrixTable.innerHTML = html;
    
    // Create heatmap visualization (using simple HTML/CSS)
    if (heatmapDiv) {
        heatmapDiv.innerHTML = `
            <div style="display:inline-block; border:1px solid #e2e8f0; padding:10px; background:white;">
                <div style="display:grid; grid-template-columns:repeat(${columns.length}, 40px); gap:2px;">
                    ${columns.map(col1 => 
                        columns.map(col2 => {
                            const value = data.correlations[col1][col2];
                            const absValue = Math.abs(value);
                            let bgColor = '#ffffff';
                            
                            if (col1 === col2) {
                                bgColor = '#f8f9fa';
                            } else if (absValue >= 0.8) {
                                bgColor = value > 0 ? '#4CAF50' : '#F44336';
                            } else if (absValue >= 0.6) {
                                bgColor = value > 0 ? '#8BC34A' : '#FF9800';
                            } else if (absValue >= 0.4) {
                                bgColor = value > 0 ? '#CDDC39' : '#FFC107';
                            } else if (absValue >= 0.2) {
                                bgColor = value > 0 ? '#E6EE9C' : '#FFE082';
                            }
                            
                            return `<div title="${col1} vs ${col2}: ${value.toFixed(3)}" 
                                    style="width:40px; height:40px; background:${bgColor}; display:flex; align-items:center; justify-content:center; cursor:pointer; font-size:0.7em;">
                                    ${col1 === col2 ? '—' : value.toFixed(2)}
                                   </div>`;
                        }).join('')
                    ).join('')}
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:5px; font-size:0.8em;">
                    <span style="color:#F44336;">-1.0</span>
                    <span style="color:#FF9800;">-0.5</span>
                    <span>0.0</span>
                    <span style="color:#8BC34A;">0.5</span>
                    <span style="color:#4CAF50;">1.0</span>
                </div>
            </div>
        `;
    }
    
    // Scroll to results
    if (resultsSection) resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) resultsSection.style.display = 'none';
    currentCorrelations = null;
}

// Navigation Functions
function goBack() {
    window.location.href = `/${currentDomain}`;
}

function goToDataOps() {
    window.location.href = `/${currentDomain}/data_ops`;
}

function goToOutlierAnalysis() {
    window.location.href = `/${currentDomain}/outlier_analysis`;
}

// Select All Columns logic for correlation analysis
document.addEventListener('DOMContentLoaded', function() {
    const selectAll = document.getElementById('select-all-corr-columns');
    const checkboxes = Array.from(document.querySelectorAll('.corr-column-checkbox'));

    if (selectAll && checkboxes.length > 0) {
        selectAll.addEventListener('change', function() {
            checkboxes.forEach(cb => cb.checked = selectAll.checked);
        });

        checkboxes.forEach(cb => {
            cb.addEventListener('change', function() {
                const total = checkboxes.length;
                const checked = checkboxes.filter(c => c.checked).length;
                if (checked === 0) {
                    selectAll.checked = false;
                    selectAll.indeterminate = false;
                } else if (checked === total) {
                    selectAll.checked = true;
                    selectAll.indeterminate = false;
                } else {
                    selectAll.checked = false;
                    selectAll.indeterminate = true;
                }
            });
        });
    }
});