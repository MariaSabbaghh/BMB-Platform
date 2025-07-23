console.log("🚀 Loading data_ops.js - Enhanced version with Group By Notification");

// Enhanced data retrieval with debugging
function getPageData() {
    console.log("🔍 Getting page data...");
    
    // Check if pageData exists on window
    if (window.pageData) {
        console.log("✅ Found window.pageData:", window.pageData);
        return window.pageData;
    }
    
    // Fallback: try to get from script tag
    const scriptTag = document.querySelector('script[type="application/json"]#page-data');
    if (scriptTag) {
        try {
            const data = JSON.parse(scriptTag.textContent);
            console.log("✅ Found data in script tag:", data);
            return data;
        } catch (e) {
            console.error("❌ Error parsing script tag data:", e);
        }
    }
    
    // Last resort: empty object
    console.warn("⚠️ No page data found, using defaults");
    return {};
}

// Get data with enhanced error handling
const pageData = getPageData();
const initialColumns = Array.isArray(pageData.columns) ? pageData.columns : [];
const initialPreviewRows = Array.isArray(pageData.preview_rows) ? pageData.preview_rows : [];
const currentDomain = pageData.currentDomain || pageData.domain || 'connection';
const domainDisplay = pageData.domainDisplay || pageData.domain_display || 'Connection';

// Debug data
console.log('📋 Processed page data:', { 
    currentDomain, 
    initialColumns: initialColumns.length, 
    initialPreviewRows: initialPreviewRows.length,
    filename: pageData.filename,
    columnsArray: initialColumns,
    samplePreviewRow: initialPreviewRows[0] || 'No rows'
});

// Initial render when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded, rendering preview");
    renderPreviewTable(initialColumns, initialPreviewRows);
    
    // Debug checks
    console.log("Preview container exists:", !!document.getElementById("preview-table"));
    console.log("Preview rows sample:", initialPreviewRows.slice(0, 2));
});

// Global variables to track current state after operations
window.currentColumns = [...initialColumns];
window.currentPreviewRows = [...initialPreviewRows];

// Enhanced table rendering with better error handling
function renderTable(columns, previewRows, containerId, tableStyle = "") {
    console.log(`🎨 renderTable for ${containerId}:`, {
        columnsCount: columns ? columns.length : 0,
        rowsCount: previewRows ? previewRows.length : 0,
        hasColumns: !!columns,
        hasRows: !!previewRows
    });
    
    const container = document.getElementById(containerId);
    if (!container) {
        console.error(`❌ Container '${containerId}' not found!`);
        return;
    }
    
    // Validate inputs
    if (!columns || !Array.isArray(columns) || columns.length === 0) {
        console.warn("⚠️ No valid columns provided");
        container.innerHTML = `
            <div style='padding: 2rem; text-align: center; color: #666; border: 2px dashed #ccc; border-radius: 8px;'>
                <h4>📊 No Columns Available</h4>
                <p>No column information found in the data.</p>
                <p><strong>Debug Info:</strong> Columns: ${columns ? columns.length : 'null/undefined'}</p>
                <button onclick="window.showDebugInfo()" style="margin-top: 10px; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    🔍 Show Debug Info
                </button>
            </div>
        `;
        return;
    }
    
    if (!previewRows || !Array.isArray(previewRows) || previewRows.length === 0) {
        console.warn("⚠️ No valid preview rows provided");
        container.innerHTML = `
            <div style='padding: 2rem; text-align: center; color: #666; border: 2px dashed #ffc107; border-radius: 8px; background: #fff9e6;'>
                <h4>📋 No Data Rows</h4>
                <p>The file appears to contain only headers or is empty.</p>
                <p><strong>Found Columns:</strong> ${columns.join(', ')}</p>
                <p><strong>Rows:</strong> ${previewRows ? previewRows.length : 'null/undefined'}</p>
                <div style="margin-top: 15px;">
                    <button onclick="window.showDebugInfo()" style="margin-right: 10px; padding: 8px 16px; background: #17a2b8; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        🔍 Debug Info
                    </button>
                    <button onclick="window.retryFileLoad()" style="padding: 8px 16px; background: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer;">
                        🔄 Retry Load
                    </button>
                </div>
            </div>
        `;
        return;
    }
    
    console.log("✅ Rendering table with valid data");
    
    try {
        let html = `<table style="width:100%; border-collapse:collapse; background:white; ${tableStyle}">
            <thead>
                <tr>`;
        
        columns.forEach(col => {
            html += `<th style="border-bottom:2px solid #e2e8f0; padding:8px 10px; color:#6a0dad; font-weight:600; text-align: left;">${escapeHtml(String(col))}</th>`;
        });
        
        html += `</tr></thead><tbody>`;
        
        previewRows.forEach((row, rowIndex) => {
            if (!row || typeof row !== 'object') {
                console.warn(`⚠️ Invalid row at index ${rowIndex}:`, row);
                return;
            }
            
            html += `<tr style="${rowIndex % 2 === 0 ? 'background-color: #f8f9fa;' : ''}">`;
            
            columns.forEach(col => {
                let cell = row[col];
                let cellHtml = '';
                
                try {
                    if (cell === null || cell === undefined) {
                        cellHtml = "<em style='color:#dc2626; font-style: italic; font-weight:500;'>NULL</em>";
                    } else if (cell === '') {
                        cellHtml = "<em style='color:#dc2626; font-style: italic; font-weight:500;'>EMPTY</em>";
                    } else if (typeof cell === 'string' && cell.trim() === '') {
                        cellHtml = "<em style='color:#dc2626; font-style: italic; font-weight:500;'>WHITESPACE</em>";
                    } else if (typeof cell === 'string' && (cell.toLowerCase() === 'nan' || cell.toLowerCase() === 'na')) {
                        cellHtml = "<em style='color:#dc2626; font-style: italic; font-weight:500;'>NaN</em>";
                    } else {
                        const cellStr = String(cell);
                        if (Number.isInteger(cell) || (!isNaN(cell) && !isNaN(parseFloat(cell)) && isFinite(cell))) {
                            cellHtml = `<span style="color:#059669; font-weight:500;">${escapeHtml(cellStr)}</span>`;
                        } else {
                            cellHtml = escapeHtml(cellStr);
                        }
                    }
                } catch (e) {
                    console.error(`Error processing cell [${rowIndex}][${col}]:`, e);
                    cellHtml = `<em style="color:#dc2626;">Error: ${escapeHtml(String(cell))}</em>`;
                }
                
                html += `<td style="border-bottom:1px solid #f1f1f1; padding:7px 10px; color:#444; text-align: left;">${cellHtml}</td>`;
            });
            html += `</tr>`;
        });
        
        html += `</tbody></table>`;
        container.innerHTML = html;
        
        console.log(`✅ Successfully rendered table in ${containerId}`);
        
    } catch (error) {
        console.error("💥 Error rendering table:", error);
        container.innerHTML = `
            <div style='padding: 2rem; text-align: center; color: #dc2626; border: 2px solid #dc2626; border-radius: 8px; background: #fef2f2;'>
                <h4>❌ Rendering Error</h4>
                <p><strong>Error:</strong> ${error.message}</p>
                <button onclick="window.showDebugInfo()" style="margin-top: 10px; padding: 8px 16px; background: #dc2626; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    🔍 Show Debug Info
                </button>
            </div>
        `;
    }
}

function renderPreviewTable(columns, previewRows) {
    console.log("🎯 renderPreviewTable called:", {
        columns: columns ? columns.length : 0,
        rows: previewRows ? previewRows.length : 0
    });
    
    const rowCount = previewRows && Array.isArray(previewRows) ? previewRows.length : 0;
    const colCount = columns && Array.isArray(columns) ? columns.length : 0;
    
    // Update counters
    const rowCountEl = document.getElementById("row-count");
    const colCountEl = document.getElementById("col-count");
    
    if (rowCountEl) {
        rowCountEl.textContent = rowCount;
        console.log(`✅ Updated row count: ${rowCount}`);
    } else {
        console.warn("⚠️ row-count element not found");
    }
    
    if (colCountEl) {
        colCountEl.textContent = colCount;
        console.log(`✅ Updated col count: ${colCount}`);
    } else {
        console.warn("⚠️ col-count element not found");
    }
    
    // Render the main table
    renderTable(columns, previewRows, "preview-table");
    
    // Calculate missing values
    let totalMissing = 0;
    if (previewRows && previewRows.length > 0 && columns && columns.length > 0) {
        try {
            previewRows.forEach(row => {
                if (row && typeof row === 'object') {
                    columns.forEach(col => {
                        const cell = row[col];
                        if (cell === null || cell === undefined || cell === '' || 
                            (typeof cell === 'string' && (cell.trim() === '' || cell.toLowerCase() === 'nan' || cell.toLowerCase() === 'na'))) {
                            totalMissing++;
                        }
                    });
                }
            });
        } catch (e) {
            console.error("Error calculating missing values:", e);
        }
    }
    
    // Update data info
    const dataInfo = document.getElementById("data-info");
    if (dataInfo) {
        dataInfo.innerHTML = `
            Rows: <span id="row-count">${rowCount}</span> | 
            Columns: <span id="col-count">${colCount}</span> | 
            Missing values: <span style="color: ${totalMissing > 0 ? '#dc2626' : '#059669'}; font-weight: 500;">${totalMissing}</span>
        `;
        console.log(`✅ Updated data info - Missing: ${totalMissing}`);
    } else {
        console.warn("⚠️ data-info element not found");
    }
}

function showCleanedResults(data) {
    console.log("🎉 Showing cleaned results:", data);
    
    const resultsSection = document.getElementById("results-section");
    const resultsSummary = document.getElementById("results-summary");
    const cleanedTable = document.getElementById("cleaned-table");
    const postSaveFilename = document.getElementById("post-save-filename");
    const postSaveMessage = document.getElementById("post-save-message");
    
    // Show results section
    if (resultsSection) {
        resultsSection.style.display = "block";
    } else {
        console.error("❌ results-section not found");
        return;
    }
    
    // Set up post-operation save filename with processed suffix (only in normal mode)
    if (postSaveFilename && (!window.editMode || !window.editMode.isEditing)) {
        const currentFilename = pageData.filename || "processed_data.csv";
        const nameWithoutExt = currentFilename.replace(/\.[^/.]+$/, "");
        const ext = currentFilename.split('.').pop();
        postSaveFilename.value = `${nameWithoutExt}_processed.${ext}`;
        
        // Show info message about saving location
        if (postSaveMessage) {
            postSaveMessage.innerHTML = "💡 Save your processed data to the cleaned_data folder for future reference";
            postSaveMessage.style.color = "#666";
        }
    }
    
    // Update save section for edit mode
    updateSaveSection();
    
    // Create summary
    let summaryHtml = `<strong>✅ Data Cleaning Complete!</strong><br>`;
    if (data.original_shape && data.final_shape) {
        const rowChange = data.final_shape[0] - data.original_shape[0];
        const rowChangeText = rowChange === 0 ? "No change" : (rowChange > 0 ? `+${rowChange} rows` : `${Math.abs(rowChange)} rows removed`);
        summaryHtml += `📊 <strong>Rows:</strong> ${data.original_shape[0]} → ${data.final_shape[0]} (${rowChangeText})<br>`;
    }
    
    if (data.operation_details && data.operation_details.length > 0) {
        summaryHtml += `⚙️ <strong>Applied operations:</strong><br>`;
        data.operation_details.slice(0, 3).forEach(detail => {
            summaryHtml += `&nbsp;&nbsp;• ${detail}<br>`;
        });
        if (data.operation_details.length > 3) {
            summaryHtml += `&nbsp;&nbsp;• ... and ${data.operation_details.length - 3} more operations`;
        }
    }
    
    if (resultsSummary) {
        resultsSummary.innerHTML = summaryHtml;
    }
    
    // Render cleaned data table
    if (cleanedTable && data.columns && data.preview_rows) {
        renderTable(data.columns, data.preview_rows, "cleaned-table", "border:none;");
    }
    
    // Update preview title
    const previewTitle = document.getElementById("preview-title");
    if (previewTitle) {
        previewTitle.textContent = "Original Data (Before Cleaning)";
    }
    
    // Scroll to results
    if (resultsSection) {
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function escapeHtml(text) {
    if (typeof text !== 'string') {
        text = String(text);
    }
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, function(m) { return map[m]; });
}

// Debug functions
window.showDebugInfo = function() {
    const info = {
        pageData: pageData,
        initialColumns: initialColumns,
        initialPreviewRows: initialPreviewRows.slice(0, 2), // Only show first 2 rows
        currentColumns: window.currentColumns,
        currentPreviewRows: window.currentPreviewRows.slice(0, 2),
        sessionInfo: {
            currentDomain: currentDomain,
            filename: pageData.filename
        },
        elements: {
            previewTable: !!document.getElementById("preview-table"),
            dataInfo: !!document.getElementById("data-info"),
            rowCount: !!document.getElementById("row-count"),
            colCount: !!document.getElementById("col-count"),
            operationsList: !!document.getElementById("operations-list"),
            addOperationBtn: !!document.getElementById("add-operation-btn")
        }
    };
    
    console.log("🐛 Complete Debug Info:", info);
    
    // Show debug modal
    const debugModal = document.createElement('div');
    debugModal.style.cssText = `
        position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
        background: white; border: 2px solid #007bff; border-radius: 12px;
        padding: 20px; max-width: 80%; max-height: 80%; overflow: auto;
        z-index: 9999; box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    `;
    
    debugModal.innerHTML = `
        <h3 style="margin-top: 0; color: #007bff;">🔍 Debug Information</h3>
        <div style="margin-bottom: 15px;">
            <strong>Columns:</strong> ${initialColumns.length} found<br>
            <strong>Preview Rows:</strong> ${initialPreviewRows.length} found<br>
            <strong>Current Domain:</strong> ${currentDomain}<br>
            <strong>Filename:</strong> ${pageData.filename || 'Not set'}
        </div>
        <div style="margin-bottom: 15px;">
            <strong>Elements Status:</strong><br>
            ${Object.entries(info.elements).map(([key, exists]) => 
                `• ${key}: ${exists ? '✅' : '❌'}`).join('<br>')}
        </div>
        <div style="margin-bottom: 15px;">
            <strong>Sample Data:</strong><br>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 4px; font-size: 12px; max-height: 200px; overflow: auto;">${JSON.stringify(info, null, 2)}</pre>
        </div>
        <button onclick="this.parentElement.remove()" style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
            Close
        </button>
    `;
    
    document.body.appendChild(debugModal);
};

window.retryFileLoad = function() {
    console.log("🔄 Retrying file load...");
    window.location.reload();
};

// === OPERATIONS MANAGEMENT ===

const operationsList = document.getElementById("operations-list");
const addOperationBtn = document.getElementById("add-operation-btn");
let operationCount = 0;

function handleSelectAllToggle(selectAllCheckbox, columnCheckboxes) {
    const isChecked = selectAllCheckbox.checked;
    columnCheckboxes.forEach(checkbox => {
        checkbox.checked = isChecked;
    });
}

function handleColumnCheckboxChange(selectAllCheckbox, columnCheckboxes) {
    const totalCheckboxes = columnCheckboxes.length;
    const checkedCheckboxes = columnCheckboxes.filter(cb => cb.checked).length;
    
    if (checkedCheckboxes === 0) {
        selectAllCheckbox.checked = false;
        selectAllCheckbox.indeterminate = false;
    } else if (checkedCheckboxes === totalCheckboxes) {
        selectAllCheckbox.checked = true;
        selectAllCheckbox.indeterminate = false;
    } else {
        selectAllCheckbox.checked = false;
        selectAllCheckbox.indeterminate = true;
    }
}

// Updated createOperationBlock function with notification under Function section
function createOperationBlock() {
    const opId = `operation-${operationCount++}`;
    const block = document.createElement("div");
    block.className = "operation-block";
    block.style.cssText = "border:1px solid #e2e8f0; border-radius:8px; padding:1rem; margin-bottom:1rem; background:#f9f9fc;";
    block.dataset.id = opId;

    const currentColumns = window.currentColumns || initialColumns;

    if (currentColumns.length === 0) {
        block.innerHTML = `
            <div style="padding: 1rem; text-align: center; color: #dc2626; border: 1px solid #dc2626; border-radius: 4px; background: #fef2f2;">
                ❌ No columns available for operations. Please ensure your file is properly loaded.
                <button onclick="window.retryFileLoad()" style="margin-left: 10px; padding: 4px 8px; background: #dc2626; color: white; border: none; border-radius: 4px; cursor: pointer;">
                    Retry
                </button>
            </div>
        `;
        return block;
    }

    block.innerHTML = `
        <div style="display:flex; gap:20px; flex-wrap:wrap; align-items:flex-start;">
            <div style="flex:1; min-width:180px;">
                <label style="font-weight:500; color:#6a0dad;">Function<br>
                    <select class="function-select" name="${opId}-function" required style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                        <option value="">Select function</option>
                        <option value="handle_missing_values">Handle Missing Values</option>
                        <option value="fix_data_types">Fix Data Types</option>
                        <option value="encode_categorical">Encode Categorical Column</option>
                        <option value="group_by">Group By Column</option>
                        <option value="delete_duplicates">Delete Duplicates</option>
                    </select>
                </label>
                <small style="color:#888; display:block; margin-top:4px;">
                    💡 For outlier removal, use <a href="#" onclick="goToOutlierAnalysis()" style="color:#e74c3c; text-decoration:underline;">Outlier Analysis</a>
                </small>
                
                <!-- Group By Function Notice - Shows when Group By is selected -->
                <div class="group-by-function-notice" id="group-by-function-notice-${opId}" style="display: none; margin-top: 12px;">
                    <div style="
                        background: linear-gradient(135deg, #f0f8ff 0%, #e6f3ff 100%);
                        border: 2px solid #4a90e2;
                        border-radius: 8px;
                        padding: 12px 16px;
                        box-shadow: 0 2px 8px rgba(74, 144, 226, 0.15);
                        position: relative;
                        overflow: hidden;
                    ">
                        <!-- Notice header with icon -->
                        <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                            <div style="
                                background: #4a90e2;
                                color: white;
                                border-radius: 50%;
                                width: 20px;
                                height: 20px;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                font-size: 12px;
                                font-weight: bold;
                            ">!</div>
                            <strong style="color: #2c5282; font-size: 0.95rem;">Machine Learning Application Notice</strong>
                        </div>
                        
                        <!-- Notice content -->
                        <div style="color: #2c5282; font-size: 0.9rem; line-height: 1.4;">
                            <p style="margin: 0 0 8px 0;">
                                <strong>Group By operations</strong> are commonly used for <strong>Association Rule Learning</strong> in unsupervised machine learning.
                            </p>
                            <div style="background: rgba(74, 144, 226, 0.1); padding: 8px; border-radius: 4px; margin: 8px 0;">
                                <div style="font-size: 0.85rem;">
                                    <strong>🎯 Use Cases:</strong> Market basket analysis, recommendation systems, pattern discovery
                                </div>
                            </div>
                            <div style="font-size: 0.85rem; color: #4a5568;">
                                <strong>💡 Next Step:</strong> Consider using the grouped data for association training in the unsupervised learning section
                            </div>
                        </div>
                        
                        <!-- Subtle accent border -->
                        <div style="
                            position: absolute;
                            top: 0;
                            left: 0;
                            right: 0;
                            height: 3px;
                            background: linear-gradient(90deg, #4a90e2, #667eea, #764ba2);
                        "></div>
                    </div>
                </div>
            </div>
            
            <div class="method-area" style="flex:1.5; min-width:250px;">
                <!-- Will be populated based on function selection -->
            </div>
            
            <div class="columns-area" style="flex:1.5; min-width:200px; display:none;">
                <label style="font-weight:500; color:#6a0dad;">Columns<br>
                    <div style="border:1px solid #e2e8f0; border-radius:6px; padding:6px; background:#fff; max-height:120px; overflow-y:auto;">
                        <!-- Select All Option -->
                        <div style="border-bottom:1px solid #e2e8f0; padding-bottom:4px; margin-bottom:4px; background:#f8f9fa;">
                            <label style="display:block; font-weight:600; color:#6a0dad;">
                                <input type="checkbox" class="select-all-checkbox" style="margin-right:6px;">
                                Select All Columns
                            </label>
                        </div>
                        <!-- Individual Column Checkboxes -->
                        ${currentColumns.map(col => `
                            <label style="display:block; margin-bottom:2px; font-weight:400;">
                                <input type="checkbox" class="column-checkbox" name="${opId}-columns" value="${escapeHtml(String(col))}" style="margin-right:6px;">
                                ${escapeHtml(String(col))}
                            </label>
                        `).join("")}
                    </div>
                    <small style="color:#888;">(Check one or more columns)</small>
                </label>
            </div>
            
            <div style="align-self:flex-start;">
                <button type="button" class="remove-operation-btn btn-secondary" style="margin-top:22px;">Remove</button>
            </div>
        </div>
    `;

    // Set up select all functionality
    const selectAllCheckbox = block.querySelector(".select-all-checkbox");
    const columnCheckboxes = Array.from(block.querySelectorAll(".column-checkbox"));
    
    if (selectAllCheckbox) {
        selectAllCheckbox.addEventListener("change", function() {
            handleSelectAllToggle(this, columnCheckboxes);
        });
    }
    
    columnCheckboxes.forEach(checkbox => {
        checkbox.addEventListener("change", function() {
            handleColumnCheckboxChange(selectAllCheckbox, columnCheckboxes);
        });
    });

    // Method area logic
    const funcSelect = block.querySelector(".function-select");
    const methodArea = block.querySelector(".method-area");
    const columnsArea = block.querySelector(".columns-area");
    const functionNotice = block.querySelector(`#group-by-function-notice-${opId}`);

    function updateMethodArea() {
        let html = "";
        const selectedFunction = funcSelect.value;
        
        // Show/hide the Group By function notice
        if (functionNotice) {
            if (selectedFunction === "group_by") {
                console.log("🔔 Showing Group By function notice");
                functionNotice.style.display = "block";
                functionNotice.style.animation = "slideInDown 0.3s ease-out";
            } else {
                console.log("🔔 Hiding Group By function notice");
                functionNotice.style.display = "none";
            }
        }
        
        if (selectedFunction === "handle_missing_values") {
            html = `<label style="font-weight:500; color:#6a0dad;">Method<br>
                <select class="option-select" name="${opId}-method" style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                    <option value="drop">Drop rows</option>
                    <option value="mean">Fill with mean</option>
                    <option value="median">Fill with median</option>
                    <option value="mode">Fill with mode</option>
                    <option value="forward_fill">Forward fill</option>
                    <option value="backward_fill">Backward fill</option>
                </select>
            </label>`;
            
            columnsArea.style.display = "block";
            
        } else if (selectedFunction === "fix_data_types") {
            html = `<label style="font-weight:500; color:#6a0dad;">Data Type<br>
                <select class="dtype-select" name="${opId}-dtype" style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                    <option value="int">Integer</option>
                    <option value="float">Float</option>
                    <option value="str">String</option>
                    <option value="bool">Boolean</option>
                    <option value="category">Category</option>
                </select>
            </label>`;
            
            columnsArea.style.display = "block";
            
        } else if (selectedFunction === "encode_categorical") {
            html = `<label style="font-weight:500; color:#6a0dad;">Encoding Method<br>
                <select class="option-select" name="${opId}-method" style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                    <option value="onehot">One-Hot Encoding</option>
                    <option value="label">Label Encoding</option>
                    <option value="ordinal">Ordinal Encoding</option>
                </select>
            </label>`;
            
            columnsArea.style.display = "block";
            
        } else if (selectedFunction === "group_by") {
            html = `
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <label style="font-weight:500; color:#6a0dad;">Group By Column<br>
                        <select class="group-by-select" name="${opId}-group-by-column" style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                            <option value="">Select column to group by</option>
                            ${currentColumns.map(col => `<option value="${escapeHtml(String(col))}">${escapeHtml(String(col))}</option>`).join("")}
                        </select>
                    </label>
                    
                    <label style="font-weight:500; color:#6a0dad;">Aggregate Column<br>
                        <select class="aggregate-select" name="${opId}-aggregate-column" style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                            <option value="">Select column to aggregate</option>
                            ${currentColumns.map(col => `<option value="${escapeHtml(String(col))}">${escapeHtml(String(col))}</option>`).join("")}
                        </select>
                    </label>
                    <label style="font-weight:500; color:#6a0dad;">Aggregation Method<br>
                        <select class="aggregation-method-select" name="${opId}-aggregation-method" style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                            <option value="list">Create List (comma-separated)</option>
                            <option value="unique_list">Create Unique List</option>
                            <option value="count">Count Items</option>
                            <option value="sum">Sum (numeric only)</option>
                            <option value="mean">Average (numeric only)</option>
                            <option value="first">First Value</option>
                            <option value="last">Last Value</option>
                        </select>
                    </label>
                </div>
            `;
            
            columnsArea.style.display = "none";
            
        } else if (selectedFunction === "delete_duplicates") {
            html = `
                <div style="display: flex; flex-direction: column; gap: 10px;">
                    <label style="font-weight:500; color:#6a0dad;">Duplicate Check Method<br>
                        <select class="duplicates-method-select" name="${opId}-duplicates-method" style="width:100%; padding:8px; border-radius:6px; border:1px solid #e2e8f0;">
                            <option value="all_columns">All Columns (exact row match)</option>
                            <option value="selected_columns">Selected Columns Only</option>
                        </select>
                    </label>
                    <div class="duplicates-keep-options" style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                        <label style="font-weight:500; color:#6a0dad; margin-right: 15px;">Keep:</label>
                        <label style="display: flex; align-items: center; gap: 5px; margin-right: 10px;">
                            <input type="radio" name="${opId}-keep-duplicates" value="first" checked> First Occurrence
                        </label>
                        <label style="display: flex; align-items: center; gap: 5px; margin-right: 10px;">
                            <input type="radio" name="${opId}-keep-duplicates" value="last"> Last Occurrence
                        </label>
                        <label style="display: flex; align-items: center; gap: 5px;">
                            <input type="radio" name="${opId}-keep-duplicates" value="none"> None (Delete All Duplicates)
                        </label>
                    </div>
                </div>
            `;
            
            // Show columns area only if "Selected Columns Only" is chosen (handled in event listener)
            columnsArea.style.display = "none";
        }

        if (methodArea) {
            methodArea.innerHTML = html;
        }
        
        // Set up event listeners for dynamic behavior
        if (selectedFunction === "delete_duplicates") {
            const methodSelect = block.querySelector(".duplicates-method-select");
            if (methodSelect) {
                methodSelect.addEventListener("change", function() {
                    columnsArea.style.display = this.value === "selected_columns" ? "block" : "none";
                });
            }
        }
    }

    if (funcSelect) {
        funcSelect.addEventListener("change", updateMethodArea);
        updateMethodArea();
    }

    // Remove operation
    const removeBtn = block.querySelector(".remove-operation-btn");
    if (removeBtn) {
        removeBtn.addEventListener("click", () => {
            block.remove();
        });
    }

    return block;
}

// Initialize operations when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log("🔧 Initializing operations...");
    
    // Add first operation by default
    if (operationsList) {
        console.log("✅ Adding initial operation block");
        operationsList.appendChild(createOperationBlock());
    } else {
        console.error("❌ operations-list element not found");
    }

    // Add operation button
    if (addOperationBtn) {
        console.log("✅ Setting up add operation button");
        addOperationBtn.addEventListener("click", () => {
            console.log("➕ Adding new operation block");
            operationsList.appendChild(createOperationBlock());
        });
    } else {
        console.error("❌ add-operation-btn element not found");
    }
});

// === FORM SUBMISSION ===

document.addEventListener('DOMContentLoaded', function() {
    const dataOpsForm = document.getElementById("data-ops-form");
    if (!dataOpsForm) {
        console.error("❌ data-ops-form not found");
        return;
    }
    
    console.log("✅ Setting up form submission");
    
    dataOpsForm.addEventListener("submit", function(e) {
        e.preventDefault();
        console.log("📝 Form submitted - processing operations");
        
        const resultMessage = document.getElementById("result-message");
        const submitBtn = this.querySelector('button[type="submit"]');
        
        // Show loading state
        if (resultMessage) {
            resultMessage.textContent = "🔄 Processing operations...";
            resultMessage.style.color = "#007bff";
        }
        
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.textContent = "Processing...";
        }
        
        const ops = [];
        let hasErrors = false;
        
        // Collect operations from form
        document.querySelectorAll(".operation-block").forEach((block, index) => {
            console.log(`Processing operation block ${index + 1}`);
            
            const func = block.querySelector(".function-select")?.value;
            if (!func) {
                console.log(`Skipping empty operation block ${index + 1}`);
                return; // Skip empty operations
            }
            
            let params = { function: func };
            
            if (func === "delete_duplicates") {
                // Handle delete duplicates operation
                const method = block.querySelector(".duplicates-method-select")?.value || "all_columns";
                const keepRadios = block.querySelectorAll(`input[name$="-keep-duplicates"]:checked`);
                const keep = keepRadios.length > 0 ? keepRadios[0].value : "first";
                
                params.method = method;
                params.keep = keep;
                
                // If method is "selected_columns", get the selected columns
                if (method === "selected_columns") {
                    const columns = Array.from(block.querySelectorAll('.column-checkbox:checked')).map(cb => cb.value);
                    if (columns.length === 0) {
                        console.error(`Delete duplicates with selected columns method requires at least one column in block ${index + 1}`);
                        if (resultMessage) {
                            resultMessage.innerHTML = "❌ Delete duplicates with 'Selected Columns Only' method requires at least one column to be selected.";
                            resultMessage.style.color = "#dc2626";
                        }
                        hasErrors = true;
                        return;
                    }
                    params.columns = columns;
                } else {
                    // For "all_columns" method, we don't need specific columns
                    params.columns = [];
                }
                
                console.log(`Added delete duplicates operation: method='${method}', keep='${keep}', columns=${JSON.stringify(params.columns)}`);
                
            } else if (func === "group_by") {
                // Handle group by operation
                const groupByCol = block.querySelector(".group-by-select")?.value;
                const aggregateCol = block.querySelector(".aggregate-select")?.value;
                const aggregationMethod = block.querySelector(".aggregation-method-select")?.value || "list";
                
                if (!groupByCol) {
                    console.error(`Group by operation missing Group By Column in block ${index + 1}`);
                    if (resultMessage) {
                        resultMessage.innerHTML = "❌ Group by operation requires 'Group By Column' to be selected.";
                        resultMessage.style.color = "#dc2626";
                    }
                    hasErrors = true;
                    return;
                }
                
                if (!aggregateCol) {
                    console.error(`Group by operation missing Aggregate Column in block ${index + 1}`);
                    if (resultMessage) {
                        resultMessage.innerHTML = "❌ Group by operation requires 'Aggregate Column' to be selected.";
                        resultMessage.style.color = "#dc2626";
                    }
                    hasErrors = true;
                    return;
                }
                
                params.columns = [groupByCol]; // Required for the processing loop
                params.group_by_column = groupByCol;
                params.aggregate_column = aggregateCol;
                params.aggregation_method = aggregationMethod;
                
                console.log(`Added group by operation: Group by '${groupByCol}', aggregate '${aggregateCol}' using '${aggregationMethod}'`);
                
            } else {
                // Handle regular operations that need column selection
                const columns = Array.from(block.querySelectorAll('.column-checkbox:checked')).map(cb => cb.value);
                if (columns.length === 0) {
                    console.error(`No columns selected for operation ${index + 1}`);
                    if (resultMessage) {
                        resultMessage.innerHTML = "❌ Please select at least one column for each operation (except Delete Duplicates with 'All Columns' method).";
                        resultMessage.style.color = "#dc2626";
                    }
                    hasErrors = true;
                    return;
                }
                
                params.columns = columns;
                
                if (func === "handle_missing_values") {
                    const method = block.querySelector(".option-select")?.value;
                    if (method) {
                        params.method = method;
                    }
                } else if (func === "fix_data_types") {
                    const dtype = block.querySelector(".dtype-select")?.value;
                    if (dtype) {
                        params.dtype = dtype;
                    }
                } else if (func === "encode_categorical") {
                    const method = block.querySelector(".option-select")?.value;
                    if (method) {
                        params.method = method;
                    }
                }
                
                console.log(`Added operation: ${func} on columns: ${columns.join(', ')}`);
            }
            
            ops.push(params);
        });
        
        if (hasErrors) {
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = "Apply All Operations";
            }
            return;
        }
        
        if (ops.length === 0) {
            console.warn("No operations to apply");
            if (resultMessage) {
                resultMessage.innerHTML = "❌ No operations to apply.";
                resultMessage.style.color = "#dc2626";
            }
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = "Apply All Operations";
            }
            return;
        }
        
        console.log(`Sending ${ops.length} operations to server:`, ops);
        
        // Send operations to server
        fetch(`/${currentDomain}/apply_function`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ 
                operations: ops
            })
        })
        .then(response => {
            console.log(`Server response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("✅ Server response:", data);
            
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = "Apply All Operations";
            }
            
            if (data.success) {
                // Show visual results table instead of just a message
                showCleanedResults(data);
                
                let successMessage = "✅ Operations completed! Check the cleaned data results above.";
                successMessage += "<br>📁 Your original file remains unchanged. Use 'Save Processed Data' to save the cleaned version.";
                
                if (resultMessage) {
                    resultMessage.innerHTML = successMessage;
                    resultMessage.style.color = "#059669";
                }
                
                // Update the global variables so subsequent operations use the new data
                if (data.preview_rows && data.columns) {
                    window.currentColumns = data.columns;
                    window.currentPreviewRows = data.preview_rows;
                    console.log("✅ Updated current data state");
                }
                
                // Clear the operations form for next use
                document.querySelectorAll(".operation-block").forEach(block => block.remove());
                if (operationsList) {
                    operationsList.appendChild(createOperationBlock());
                }
                
            } else {
                console.error("❌ Server returned error:", data.message);
                if (resultMessage) {
                    resultMessage.innerHTML = `❌ ${data.message || "Operation failed"}`;
                    resultMessage.style.color = "#dc2626";
                }
            }
        })
        .catch(error => {
            console.error("💥 Request failed:", error);
            
            if (submitBtn) {
                submitBtn.disabled = false;
                submitBtn.textContent = "Apply All Operations";
            }
            
            if (resultMessage) {
                resultMessage.innerHTML = `❌ Error: ${error.message}`;
                resultMessage.style.color = "#dc2626";
            }
        });
    });
});

// === SAVE FUNCTIONALITY ===

document.addEventListener('click', function(e) {
    if (e.target && e.target.id === 'post-save-btn') {
        console.log("💾 Save button clicked");
        
        const filename = document.getElementById("post-save-filename")?.value?.trim();
        
        // In edit mode, we'll use the original filename and update the version
        let saveData = {};
        
        if (window.editMode && window.editMode.isEditing) {
            // Edit mode - update existing version
            saveData = {
                filename: window.editMode.originalFilename,
                edit_version: window.editMode.version,
                original_filename: window.editMode.originalFilename,
                domain: currentDomain
            };
            
            // Update button text for edit mode
            e.target.textContent = "Updating Version...";
        } else {
            // Normal mode - create new file
            if (!filename) {
                alert("Please enter a filename.");
                return;
            }
            saveData = { 
                filename: filename,
                domain: currentDomain
            };
            e.target.textContent = "Saving...";
        }
        
        e.target.disabled = true;
        
        fetch("/connection/save_processed_file", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(saveData)
        })
        .then(response => response.json())
        .then(data => {
            console.log("💾 Save response:", data);
            
            e.target.disabled = false;
            
            const postSaveMessage = document.getElementById("post-save-message");
            if (data.success) {
                if (window.editMode && window.editMode.isEditing) {
                    // Edit mode success
                    e.target.textContent = "✅ Version Updated";
                    e.target.style.background = "#28a745";
                    if (postSaveMessage) {
                        postSaveMessage.innerHTML = `✅ Version ${data.version_updated} updated successfully in: <strong>${data.saved_path}</strong><br>
                            <small><a href="/catalog" style="color: #007bff;">← Back to Catalog</a> to see the updated version</small>`;
                        postSaveMessage.style.color = "#059669";
                    }
                    
                    // Add success animation
                    e.target.style.transform = "scale(1.05)";
                    setTimeout(() => {
                        e.target.style.transform = "scale(1)";
                    }, 200);
                } else {
                    // Normal mode success
                    e.target.textContent = "💾 Save Processed Data";
                    if (postSaveMessage) {
                        postSaveMessage.innerHTML = `✅ File saved successfully to: <strong>${data.saved_path}</strong>`;
                        postSaveMessage.style.color = "#059669";
                    }
                }
            } else {
                e.target.textContent = window.editMode && window.editMode.isEditing ? "Update Version" : "💾 Save Processed Data";
                if (postSaveMessage) {
                    postSaveMessage.innerHTML = `❌ Error saving file: ${data.message}`;
                    postSaveMessage.style.color = "#dc2626";
                }
            }
        })
        .catch(error => {
            console.error("💥 Save error:", error);
            
            e.target.disabled = false;
            e.target.textContent = window.editMode && window.editMode.isEditing ? "Update Version" : "💾 Save Processed Data";
            const postSaveMessage = document.getElementById("post-save-message");
            if (postSaveMessage) {
                postSaveMessage.innerHTML = `❌ Error saving file: ${error.message}`;
                postSaveMessage.style.color = "#dc2626";
            }
        });
    }
    
    // Download CSV Button Handler
    if (e.target && e.target.id === 'download-csv-btn') {
        console.log("📥 Download button clicked");
        
        e.target.disabled = true;
        e.target.textContent = "Downloading...";
        
        // Get the current filename and create download link
        const currentFilename = pageData.filename || "data.csv";
        const downloadUrl = `/uploads/${encodeURIComponent(currentFilename)}`;
        
        // Create temporary download link
        const link = document.createElement('a');
        link.href = downloadUrl;
        link.download = currentFilename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        setTimeout(() => {
            e.target.disabled = false;
            e.target.textContent = "📥 Download CSV";
        }, 1000);
        
        const postSaveMessage = document.getElementById("post-save-message");
        if (postSaveMessage) {
            postSaveMessage.innerHTML = `✅ Download started for "${currentFilename}"`;
            postSaveMessage.style.color = "#059669";
        }
    }
});

// === TOGGLE FUNCTIONALITY ===

document.addEventListener('DOMContentLoaded', function() {
    const toggleBtn = document.getElementById("toggle-original");
    if (toggleBtn) {
        console.log("✅ Setting up toggle button");
        
        toggleBtn.addEventListener("click", function() {
            const container = document.getElementById("preview-table-container");
            const btn = this;
            
            if (container.style.display === "none") {
                container.style.display = "block";
                btn.textContent = "Hide Original";
            } else {
                container.style.display = "none";
                btn.textContent = "Show Original";
            }
        });
    }
});

// === SAVE SECTION UPDATES ===

function updateSaveSection() {
    const postOperationSave = document.getElementById("post-operation-save");
    if (postOperationSave && window.editMode && window.editMode.isEditing) {
        const saveHeader = postOperationSave.querySelector("h4");
        const saveButton = document.getElementById("post-save-btn");
        const filenameInput = document.getElementById("post-save-filename");
        const helpText = postOperationSave.querySelector("small");
        
        if (saveHeader) {
            saveHeader.innerHTML = `<i class="fas fa-save"></i> Update Version ${window.editMode.version}`;
        }
        
        if (saveButton) {
            saveButton.textContent = "🔄 Update Version";
            saveButton.style.background = "#6f42c1";
            saveButton.style.borderColor = "#6f42c1";
        }
        
        if (filenameInput) {
            filenameInput.style.display = "none";
            if (filenameInput.previousElementSibling) {
                filenameInput.previousElementSibling.style.display = "none"; // Hide label text
            }
        }
        
        if (helpText) {
            helpText.innerHTML = `
                📁 This will update version ${window.editMode.version} of: <strong>${window.editMode.originalFilename}</strong><br>
                ✅ The version will be updated in place - no new version will be created
            `;
        }
    }
}

// === NAVIGATION FUNCTIONS ===

function goBack() {
    window.history.back();
}

function goToOutlierAnalysis() {
    window.location.href = `/${currentDomain}/outlier_analysis`;
}

function goToCorrelationAnalysis() {
    window.location.href = `/${currentDomain}/correlation_analysis`;
}

// === STATISTICS FILTERING ===

document.addEventListener("DOMContentLoaded", function () {
    console.log("📊 Setting up statistics filtering");
    
    const columnCheckboxes = document.querySelectorAll(".column-checkbox");
    const metricCheckboxes = document.querySelectorAll(".metric-checkbox");
    const selectAllColumns = document.getElementById("select-all-columns");
    const selectAllMetrics = document.getElementById("select-all-metrics");

    function updateVisibleRows() {
        const selectedCols = Array.from(columnCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);
        const selectedMetrics = Array.from(metricCheckboxes)
            .filter(cb => cb.checked)
            .map(cb => cb.value);

        document.querySelectorAll(".stats-row").forEach(row => {
            const col = row.getAttribute("data-col");
            const metric = row.getAttribute("data-metric");
            const show = selectedCols.includes(col) && selectedMetrics.includes(metric);
            row.style.display = show ? "" : "none";
        });
    }

    if (selectAllColumns) {
        selectAllColumns.addEventListener("change", function () {
            columnCheckboxes.forEach(cb => {
                cb.checked = this.checked;
            });
            updateVisibleRows();
        });
    }

    if (selectAllMetrics) {
        selectAllMetrics.addEventListener("change", function () {
            metricCheckboxes.forEach(cb => {
                cb.checked = this.checked;
            });
            updateVisibleRows();
        });
    }

    columnCheckboxes.forEach(cb => cb.addEventListener("change", updateVisibleRows));
    metricCheckboxes.forEach(cb => cb.addEventListener("change", updateVisibleRows));

    updateVisibleRows(); // Initial filter
});

// === DEBUG FUNCTIONS FOR GROUP BY NOTIFICATION ===

// Function to manually add notifications to existing Group By operations
function addGroupByNotifications() {
    console.log("🔧 Manually adding Group By notifications...");
    
    const operationBlocks = document.querySelectorAll('.operation-block');
    let addedCount = 0;
    
    operationBlocks.forEach((block, index) => {
        const funcSelect = block.querySelector('.function-select');
        if (funcSelect && funcSelect.value === 'group_by') {
            const groupBySelect = block.querySelector('.group-by-select');
            if (groupBySelect) {
                // Check if notification already exists
                let notification = block.querySelector('.group-by-notification');
                if (!notification) {
                    // Create and add notification
                    notification = document.createElement('div');
                    notification.className = 'group-by-notification';
                    notification.style.display = 'none';
                    notification.innerHTML = `
                        <div style="display: flex; align-items: flex-start; gap: 12px;">
                            <div class="ml-tip-icon">🤖</div>
                            <div>
                                <strong style="color: #6f42c1;">Machine Learning Tip:</strong><br>
                                Group By operations are commonly used for <strong>Association Rule Learning</strong> in unsupervised machine learning.
                            </div>
                        </div>
                    `;
                    
                    // Insert after the dropdown
                    groupBySelect.parentNode.insertBefore(notification, groupBySelect.nextSibling);
                    
                    // Add event listener
                    groupBySelect.addEventListener('change', function() {
                        if (this.value) {
                            notification.style.display = 'block';
                            console.log(`🤖 Showing notification for block ${index}`);
                        } else {
                            notification.style.display = 'none';
                            console.log(`🤖 Hiding notification for block ${index}`);
                        }
                    });
                    
                    addedCount++;
                    console.log(`✅ Added notification to block ${index}`);
                }
            }
        }
    });
    
    console.log(`✅ Added ${addedCount} Group By notifications`);
    return addedCount;
}

// Make debug functions available globally
window.addGroupByNotifications = addGroupByNotifications;

console.log("✅ data_ops.js loaded successfully with Group By notifications!");
console.log("🔧 Debug function available: addGroupByNotifications()");