// Get data from window.pageData instead of template variables
const pageData = window.pageData || {};
const currentDomain = pageData.currentDomain || 'connection';
const domainDisplay = pageData.domainDisplay || '';
let currentAnalysis = null;

console.log('Outlier analysis loaded with domain:', currentDomain);

// Load columns overview on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, loading columns overview');
    loadColumnsOverview();
});

function loadColumnsOverview() {
    console.log('Loading columns overview from:', `/${currentDomain}/outlier_summary`);
    
    fetch(`/${currentDomain}/outlier_summary`, {
        method: 'GET'
    })
    .then(response => {
        console.log('Overview response status:', response.status);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Overview data received:', data);
        const overviewDiv = document.getElementById('columns-overview');
        
        if (data.success && data.column_summaries) {
            const summaries = data.column_summaries;
            const numColumns = Object.keys(summaries).length;
            
            if (numColumns === 0) {
                overviewDiv.innerHTML = '<p style="color:#666;">No numeric columns found in the dataset.</p>';
                return;
            }
            
            let html = `<p style="margin-bottom:1rem; color:#666;">Found <strong>${numColumns}</strong> numeric columns in your dataset:</p>`;
            html += '<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr)); gap:10px;">';
            
            for (const [column, summary] of Object.entries(summaries)) {
                const outlierColor = summary.total_outliers > 0 ? '#e74c3c' : '#10b981';
                const outlierIcon = summary.total_outliers > 0 ? '⚠️' : '✅';
                
                html += `
                    <div style="padding:0.8rem; background:#f8f9fa; border-radius:6px; border:1px solid #e9ecef;">
                        <strong style="color:#2d3748;">${column}</strong><br>
                        <small style="color:${outlierColor};">
                            ${outlierIcon} ${summary.total_outliers} outliers (${summary.outlier_percentage}%)
                        </small>
                    </div>
                `;
            }
            
            html += '</div>';
            overviewDiv.innerHTML = html;
        } else {
            overviewDiv.innerHTML = '<p style="color:#dc2626;">Error loading column overview.</p>';
        }
    })
    .catch(error => {
        console.error('Error loading overview:', error);
        document.getElementById('columns-overview').innerHTML = '<p style="color:#dc2626;">Error loading column overview.</p>';
    });
}

// Handle outlier detection form
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('outlier-detection-form');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const column = document.getElementById('column-select').value;
            const method = document.getElementById('method-select').value;
            const messageDiv = document.getElementById('detection-message');
            const submitBtn = this.querySelector('button[type="submit"]');
            
            console.log('Form submitted with:', { column, method });
            
            if (!column) {
                messageDiv.innerHTML = '❌ Please select a column to analyze.';
                messageDiv.style.color = '#dc2626';
                return;
            }
            
            // Show loading state
            messageDiv.innerHTML = '🔄 Analyzing outliers...';
            messageDiv.style.color = '#007bff';
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';
            
            fetch(`/${currentDomain}/detect_outliers`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    column: column,
                    method: method
                })
            })
            .then(response => {
                console.log('Detection response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                console.log('Detection data received:', data);
                submitBtn.disabled = false;
                submitBtn.textContent = '🔍 Analyze Outliers';
                
                if (data.success) {
                    currentAnalysis = data;
                    showResults(data);
                    messageDiv.innerHTML = '✅ Outlier analysis completed!';
                    messageDiv.style.color = '#10b981';
                } else {
                    messageDiv.innerHTML = `❌ ${data.message}`;
                    messageDiv.style.color = '#dc2626';
                    hideResults();
                }
            })
            .catch(error => {
                console.error('Detection error:', error);
                submitBtn.disabled = false;
                submitBtn.textContent = '🔍 Analyze Outliers';
                messageDiv.innerHTML = `❌ Error: ${error.message}`;
                messageDiv.style.color = '#dc2626';
                hideResults();
            });
        });
    }
});

function showResults(data) {
    const resultsSection = document.getElementById('results-section');
    const summaryDiv = document.getElementById('outlier-summary');
    const plotDiv = document.getElementById('outlier-plot');
    
    if (resultsSection) resultsSection.style.display = 'block';
    
    // Create summary
    const summary = data.outlier_summary;
    let summaryHtml = `
        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:15px; margin-bottom:1rem;">
            <div>
                <strong style="color:#e74c3c;">📊 Analysis Results</strong><br>
                <span style="font-size:0.9rem; color:#666;">Column: <strong>${data.column}</strong></span><br>
                <span style="font-size:0.9rem; color:#666;">Method: <strong>${data.method}</strong></span>
            </div>
            <div>
                <strong style="color:#e74c3c;">🔢 Outlier Count</strong><br>
                <span style="font-size:1.2rem; font-weight:bold; color:#dc2626;">${summary.total_outliers}</span> outliers<br>
                <span style="font-size:0.9rem; color:#666;">${summary.outlier_percentage.toFixed(2)}% of data</span>
            </div>
        </div>
    `;
    
    if (summary.sample_outliers && summary.sample_outliers.length > 0) {
        summaryHtml += `
            <div style="margin-top:1rem; padding:0.8rem; background:#ffffff; border-radius:6px; border:1px solid #e5e7eb;">
                <strong style="color:#374151;">Sample Outlier Values:</strong><br>
                <span style="font-family:monospace; color:#dc2626;">${summary.sample_outliers.join(', ')}</span>
                ${summary.showing_sample ? '<span style="color:#666; font-size:0.9rem;"> (showing first 10)</span>' : ''}
            </div>
        `;
    }
    
    if (summaryDiv) summaryDiv.innerHTML = summaryHtml;
    
    // Show plot
    if (data.plot_image && plotDiv) {
        plotDiv.innerHTML = `
            <h4 style="margin-bottom:1rem; color:#374151;">📈 Outlier Visualization</h4>
            <img src="data:image/png;base64,${data.plot_image}" 
                 style="max-width:100%; height:auto; border-radius:8px; box-shadow:0 4px 6px rgba(0,0,0,0.1);" 
                 alt="Outlier Detection Plot">
        `;
    }
    
    // Scroll to results
    if (resultsSection) resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function hideResults() {
    const resultsSection = document.getElementById('results-section');
    if (resultsSection) resultsSection.style.display = 'none';
    currentAnalysis = null;
}

// Handle outlier removal
document.addEventListener('DOMContentLoaded', function() {
    const removeBtn = document.getElementById('remove-outliers-btn');
    if (removeBtn) {
        removeBtn.addEventListener('click', function() {
            if (!currentAnalysis) {
                alert('No outlier analysis available. Please run detection first.');
                return;
            }
            
            const confirmRemoval = confirm(
                `Are you sure you want to remove ${currentAnalysis.outlier_summary.total_outliers} outlier(s) from column "${currentAnalysis.column}"?\n\n` +
                `This action cannot be undone and will permanently modify your dataset.`
            );
            
            if (!confirmRemoval) {
                return;
            }
            
            const btn = this;
            const originalText = btn.textContent;
            const messageDiv = document.getElementById('removal-message');
            
            // Show loading state
            btn.disabled = true;
            btn.textContent = 'Removing...';
            if (messageDiv) {
                messageDiv.innerHTML = '🔄 Removing outliers from dataset...';
                messageDiv.style.color = '#007bff';
            }
            
            fetch(`/${currentDomain}/remove_outliers`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    column: currentAnalysis.column,
                    method: currentAnalysis.method
                })
            })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                return response.json();
            })
            .then(data => {
                btn.disabled = false;
                btn.textContent = originalText;
                
                if (data.success && messageDiv) {
                    messageDiv.innerHTML = `
                        ✅ <strong>Outliers removed successfully!</strong><br>
                        📊 Rows: ${data.original_rows} → ${data.final_rows} (${data.removed_rows} removed)<br>
                        ✨ Dataset has been updated. You can now proceed to data cleaning operations.
                    `;
                    messageDiv.style.color = '#10b981';
                    
                    // Hide the action buttons and show continue section
                    const outlierActions = document.getElementById('outlier-actions');
                    const continueSection = document.getElementById('continue-section');
                    if (outlierActions) outlierActions.style.display = 'none';
                    if (continueSection) continueSection.style.display = 'block';
                    
                    // Update overview
                    loadColumnsOverview();
                    
                } else if (messageDiv) {
                    messageDiv.innerHTML = `❌ Error removing outliers: ${data.message}`;
                    messageDiv.style.color = '#dc2626';
                }
            })
            .catch(error => {
                console.error('Removal error:', error);
                btn.disabled = false;
                btn.textContent = originalText;
                const messageDiv = document.getElementById('removal-message');
                if (messageDiv) {
                    messageDiv.innerHTML = `❌ Error: ${error.message}`;
                    messageDiv.style.color = '#dc2626';
                }
            });
        });
    }
});

// Handle keep outliers
document.addEventListener('DOMContentLoaded', function() {
    const keepBtn = document.getElementById('keep-outliers-btn');
    if (keepBtn) {
        keepBtn.addEventListener('click', function() {
            const messageDiv = document.getElementById('removal-message');
            if (messageDiv) {
                messageDiv.innerHTML = '✅ Outliers kept in dataset. You can proceed to data cleaning operations.';
                messageDiv.style.color = '#10b981';
            }
            
            // Hide the action buttons and show continue section
            const outlierActions = document.getElementById('outlier-actions');
            const continueSection = document.getElementById('continue-section');
            if (outlierActions) outlierActions.style.display = 'none';
            if (continueSection) continueSection.style.display = 'block';
        });
    }
});

// Navigation Functions
function goBack() {
    window.location.href = `/${currentDomain}`;
}

function goToDataOps() {
    window.location.href = `/${currentDomain}/data_ops`;
}

function goToCorrelationAnalysis() {
    window.location.href = `/${currentDomain}/correlation_analysis`;
}