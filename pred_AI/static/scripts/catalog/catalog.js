// Global state
let activeInfo = {}; // Track which info sections are open
let activeFeatures = {}; // Track which features sections are open
let activeVersions = {}; // Track which version sections are open
let loadedInfo = {}; // Cache loaded info
let loadedFeatures = {}; // Cache loaded features
let loadedVersions = {}; // Cache loaded versions

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('Catalog page loaded, initializing...');
    
    // Check if we have dataset items
    const datasetItems = document.querySelectorAll('.dataset-item');
    console.log(`Found ${datasetItems.length} dataset items`);
    
    // Load quick stats for all datasets
    datasetItems.forEach((item, index) => {
        const filename = item.dataset.filename;
        console.log(`Loading stats for dataset ${index + 1}: ${filename}`);
        loadQuickStats(filename);
    });
    
    if (datasetItems.length === 0) {
        console.log('No datasets found - showing empty state');
    }
});

function loadQuickStats(filename) {
    console.log(`Loading quick stats for: ${filename}`);
    const quickStatsDiv = document.getElementById(`quick-stats-${filename}`);
    
    if (!quickStatsDiv) {
        console.error(`Quick stats element not found for filename: ${filename}`);
        return;
    }
    
    console.log(`Making request to: /catalog/info/${filename}`);
    
    fetch(`/catalog/info/${filename}`)
        .then(response => {
            console.log(`Response status for ${filename}:`, response.status);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Data received for ${filename}:`, data);
            
            if (data.success && quickStatsDiv) {
                const info = data.info;
                quickStatsDiv.innerHTML = `
                    <span class="quick-stat">
                        <i class="fas fa-table-cells"></i> ${info.rows.toLocaleString()} rows
                    </span>
                    <span class="quick-stat">
                        <i class="fas fa-columns"></i> ${info.columns} features
                    </span>
                    <span class="quick-stat">
                        <i class="fas fa-weight-hanging"></i> ${info.file_size}
                    </span>
                `;
                console.log(`Successfully loaded stats for ${filename}`);
            } else if (quickStatsDiv) {
                console.error(`Error in response for ${filename}:`, data);
                quickStatsDiv.innerHTML = '<span class="quick-stat-error">Error loading stats</span>';
            }
        })
        .catch(error => {
            console.error(`Error loading quick stats for ${filename}:`, error);
            if (quickStatsDiv) {
                quickStatsDiv.innerHTML = '<span class="quick-stat-error">Error loading stats</span>';
            }
        });
}

function toggleMenu(filename) {
    const menu = document.getElementById(`menu-${filename}`);
    const isVisible = menu.style.display === 'block';
    
    // Close all other menus
    document.querySelectorAll('.dropdown-menu').forEach(otherMenu => {
        otherMenu.style.display = 'none';
    });
    
    // Toggle current menu
    menu.style.display = isVisible ? 'none' : 'block';
}

function closeAllSections(filename) {
    // Close preview with null checks
    const previewDiv = document.getElementById(`preview-${filename}`);
    if (previewDiv && previewDiv.style && previewDiv.style.display === "block") {
        previewDiv.style.display = "none";
        const previewBtn = document.querySelector(`[data-filename="${filename}"] .preview-btn`);
        if (previewBtn) {
            const spanEl = previewBtn.querySelector('span');
            const iconEl = previewBtn.querySelector('i');
            if (spanEl) spanEl.textContent = "View";
            if (iconEl) iconEl.className = "fas fa-eye";
        }
    }
    
    // Close info with null checks
    if (activeInfo[filename]) {
        const infoSection = document.getElementById(`info-section-${filename}`);
        const infoBtn = document.querySelector(`[data-filename="${filename}"] .info-btn`);
        if (infoSection) infoSection.classList.remove('active');
        if (infoBtn) infoBtn.classList.remove('active');
        activeInfo[filename] = false;
    }
    
    // Close features with null checks
    if (activeFeatures[filename]) {
        const featuresSection = document.getElementById(`features-section-${filename}`);
        const featuresBtn = document.querySelector(`[data-filename="${filename}"] .features-btn`);
        if (featuresSection) featuresSection.classList.remove('active');
        if (featuresBtn) featuresBtn.classList.remove('active');
        activeFeatures[filename] = false;
    }
    
    // Close versions with null checks
    if (activeVersions[filename]) {
        const versionSection = document.getElementById(`version-section-${filename}`);
        const versionBtn = document.querySelector(`[data-filename="${filename}"] .version-btn`);
        if (versionSection) versionSection.classList.remove('active');
        if (versionBtn) versionBtn.classList.remove('active');
        activeVersions[filename] = false;
    }
}

function toggleInfo(filename) {
    console.log(`Toggling info for: ${filename}`);
    const section = document.getElementById(`info-section-${filename}`);
    const button = document.querySelector(`[data-filename="${filename}"] .info-btn`);
    
    if (!section || !button) {
        console.error(`Info elements not found for ${filename}`);
        return;
    }
    
    // Close other sections
    closeAllSections(filename);
    
    if (activeInfo[filename]) {
        // Close info section
        section.classList.remove('active');
        button.classList.remove('active');
        activeInfo[filename] = false;
        console.log(`Closed info for ${filename}`);
    } else {
        // Open this info section
        section.classList.add('active');
        button.classList.add('active');
        activeInfo[filename] = true;
        console.log(`Opened info for ${filename}`);
        
        // Load content if not already loaded
        if (!loadedInfo[filename]) {
            console.log(`Loading info content for ${filename}`);
            loadDatasetInfo(filename);
        }
    }
}

function toggleFeatures(filename) {
    console.log(`Toggling features for: ${filename}`);
    const section = document.getElementById(`features-section-${filename}`);
    const button = document.querySelector(`[data-filename="${filename}"] .features-btn`);
    
    if (!section || !button) {
        console.error(`Features elements not found for ${filename}`);
        return;
    }
    
    // Close other sections
    closeAllSections(filename);
    
    if (activeFeatures[filename]) {
        // Close features section
        section.classList.remove('active');
        button.classList.remove('active');
        activeFeatures[filename] = false;
    } else {
        // Open this features section
        section.classList.add('active');
        button.classList.add('active');
        activeFeatures[filename] = true;
        
        // Load content if not already loaded
        if (!loadedFeatures[filename]) {
            loadFeaturesList(filename);
        }
    }
}

function toggleVersionHistory(filename) {
    console.log(`Toggling version history for: ${filename}`);
    const section = document.getElementById(`version-section-${filename}`);
    const button = document.querySelector(`[data-filename="${filename}"] .version-btn`);
    
    if (!section || !button) {
        console.error(`Version elements not found for ${filename}`);
        return;
    }
    
    // Close other sections
    closeAllSections(filename);
    
    if (activeVersions[filename]) {
        // Close version section
        section.classList.remove('active');
        button.classList.remove('active');
        activeVersions[filename] = false;
    } else {
        // Open this version section
        section.classList.add('active');
        button.classList.add('active');
        activeVersions[filename] = true;
        
        // Load content if not already loaded
        if (!loadedVersions[filename]) {
            loadVersionHistory(filename);
        }
    }
}

function viewFile(btn, filename) {
    console.log(`Viewing file: ${filename}`);
    const previewDiv = document.getElementById('preview-' + filename);
    const btnText = btn.querySelector('span');
    const btnIcon = btn.querySelector('i');

    // If preview is open, hide it and return
    if (previewDiv.style.display === "block") {
        previewDiv.style.display = "none";
        previewDiv.innerHTML = "";
        btnText.textContent = "View";
        btnIcon.className = "fas fa-eye";
        return;
    }

    // Otherwise, close other sections except preview
    closeAllSections(filename, 'preview');

    // Close any other open previews
    document.querySelectorAll('.file-preview').forEach(div => {
        if (div !== previewDiv) {
            div.style.display = "none";
            div.innerHTML = "";
        }
    });
    document.querySelectorAll('.preview-btn span').forEach(span => {
        if (span !== btnText) {
            span.textContent = "View";
        }
    });
    document.querySelectorAll('.preview-btn i').forEach(icon => {
        if (icon !== btnIcon) {
            icon.className = "fas fa-eye";
        }
    });

    previewDiv.innerHTML = "<em>Loading preview...</em>";
    previewDiv.style.display = "block";
    btnText.textContent = "Hide";
    btnIcon.className = "fas fa-eye-slash";

    // Default: first page, 10 rows
    loadSheet(filename, 0, 10);
}

function loadSheet(filename, start, length) {
    console.log(`Loading sheet for ${filename}, start: ${start}, length: ${length}`);
    const previewDiv = document.getElementById('preview-' + filename);
    
    // First, get column types
    fetch(`/catalog/info/${filename}`)
        .then(r => r.json())
        .then(infoData => {
            if (infoData.success) {
                const info = infoData.info;
                
                // Then get the preview data
                fetch(`/catalog/view/${filename}?start=${start}&length=${length}`)
                    .then(r => r.json())
                    .then(data => {
                        if (data.success) {
                            let totalSheets = Math.ceil(data.total / length);
                            let currentSheet = Math.floor(start / length) + 1;
                            
                            // Create enhanced table with column types
                            const tableHTML = enhanceTableWithTypes(data.preview, info);
                            
                            let controls = `
                                <div class="sheet-controls">
                                    <div class="sheet-controls-left">
                                        <label for="sheet-length-${filename}">
                                            <i class="fas fa-table"></i>
                                            <span>Rows per sheet:</span>
                                        </label>
                                        <select id="sheet-length-${filename}">
                                            <option value="10"${length==10?' selected':''}>10</option>
                                            <option value="25"${length==25?' selected':''}>25</option>
                                            <option value="50"${length==50?' selected':''}>50</option>
                                            <option value="100"${length==100?' selected':''}>100</option>
                                        </select>
                                    </div>
                                    <div class="sheet-controls-right">
                                        <button id="sheet-prev-${filename}" ${start==0?'disabled':''} title="Previous sheet">
                                            <i class="fas fa-chevron-left"></i>
                                        </button>
                                        <span class="sheet-page-info">
                                            Sheet <b>${currentSheet}</b> of <b>${totalSheets}</b>
                                        </span>
                                        <button id="sheet-next-${filename}" ${(start+length)>=data.total?'disabled':''} title="Next sheet">
                                            <i class="fas fa-chevron-right"></i>
                                        </button>
                                    </div>
                                </div>
                            `;
                            
                            previewDiv.innerHTML = `
                                <div class="sheet-preview-panel">
                                    ${controls}
                                    <div class="sheet-table-wrap">
                                        ${tableHTML}
                                    </div>
                                </div>
                            `;

                            // Add event listeners for controls
                            document.getElementById(`sheet-length-${filename}`).onchange = function() {
                                loadSheet(filename, 0, parseInt(this.value));
                            };
                            document.getElementById(`sheet-prev-${filename}`).onclick = function() {
                                loadSheet(filename, Math.max(0, start - length), length);
                            };
                            document.getElementById(`sheet-next-${filename}`).onclick = function() {
                                loadSheet(filename, start + length, length);
                            };
                        } else {
                            console.error(`Error loading preview for ${filename}:`, data);
                            previewDiv.innerHTML = "<div class='table-error'>Error loading preview.</div>";
                        }
                    })
                    .catch(error => {
                        console.error('Error loading sheet data:', error);
                        previewDiv.innerHTML = "<div class='table-error'>Error loading preview data.</div>";
                    });
            } else {
                console.warn(`Could not load column info for ${filename}, using fallback`);
                // Fallback to original preview without types
                fetch(`/catalog/view/${filename}?start=${start}&length=${length}`)
                    .then(r => r.json())
                    .then(data => {
                        if (data.success) {
                            let totalSheets = Math.ceil(data.total / length);
                            let currentSheet = Math.floor(start / length) + 1;
                            
                            let controls = `
                                <div class="sheet-controls">
                                    <div class="sheet-controls-left">
                                        <label for="sheet-length-${filename}">
                                            <i class="fas fa-table"></i>
                                            <span>Rows per sheet:</span>
                                        </label>
                                        <select id="sheet-length-${filename}">
                                            <option value="10"${length==10?' selected':''}>10</option>
                                            <option value="25"${length==25?' selected':''}>25</option>
                                            <option value="50"${length==50?' selected':''}>50</option>
                                            <option value="100"${length==100?' selected':''}>100</option>
                                        </select>
                                    </div>
                                    <div class="sheet-controls-right">
                                        <button id="sheet-prev-${filename}" ${start==0?'disabled':''} title="Previous sheet">
                                            <i class="fas fa-chevron-left"></i>
                                        </button>
                                        <span class="sheet-page-info">
                                            Sheet <b>${currentSheet}</b> of <b>${totalSheets}</b>
                                        </span>
                                        <button id="sheet-next-${filename}" ${(start+length)>=data.total?'disabled':''} title="Next sheet">
                                            <i class="fas fa-chevron-right"></i>
                                        </button>
                                    </div>
                                </div>
                            `;
                            
                            // Wrap the basic table in scrollable container
                            const wrappedTable = `
                                <div class="sheet-table-container">
                                    ${data.preview}
                                </div>
                            `;
                            
                            previewDiv.innerHTML = `
                                <div class="sheet-preview-panel">
                                    ${controls}
                                    <div class="sheet-table-wrap">
                                        ${wrappedTable}
                                    </div>
                                </div>
                            `;

                            // Add event listeners for controls
                            document.getElementById(`sheet-length-${filename}`).onchange = function() {
                                loadSheet(filename, 0, parseInt(this.value));
                            };
                            document.getElementById(`sheet-prev-${filename}`).onclick = function() {
                                loadSheet(filename, Math.max(0, start - length), length);
                            };
                            document.getElementById(`sheet-next-${filename}`).onclick = function() {
                                loadSheet(filename, start + length, length);
                            };
                        } else {
                            previewDiv.innerHTML = "<div class='table-error'>Error loading preview.</div>";
                        }
                    })
                    .catch(error => {
                        console.error('Error loading fallback data:', error);
                        previewDiv.innerHTML = "<div class='table-error'>Error loading preview data.</div>";
                    });
            }
        })
        .catch(error => {
            console.error('Error loading column info:', error);
            previewDiv.innerHTML = "<div class='table-error'>Error loading column information.</div>";
        });
}

function enhanceTableWithTypes(originalTableHTML, info) {
    // Create a temporary DOM element to parse the HTML
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = originalTableHTML;
    
    const table = tempDiv.querySelector('table');
    if (!table) return originalTableHTML;
    
    // Only process the first row (header row)
    const headerRow = table.querySelector('thead tr') || table.querySelector('tr:first-child');
    if (!headerRow) return originalTableHTML;
    
    const headers = headerRow.querySelectorAll('th, td');
    
    // Create column type mapping
    const columnTypes = {};
    info.numeric_features.forEach(col => columnTypes[col] = 'numeric');
    info.categorical_features.forEach(col => columnTypes[col] = 'categorical');
    info.datetime_features.forEach(col => columnTypes[col] = 'datetime');
    
    // Add type information to headers
    headers.forEach(header => {
        const columnName = header.textContent.trim();
        let columnType = columnTypes[columnName];
        
        // Handle unknown types by inferring from column name patterns
        if (!columnType) {
            columnType = inferColumnType(columnName);
        }
        
        // Add type indicator
        const typeIcon = getTypeIcon(columnType);
        const typeLabel = getTypeLabel(columnType);
        
        header.innerHTML = `
            <div class="column-header">
                <div class="column-name">${columnName}</div>
                <div class="column-type">
                    <i class="${typeIcon}"></i>
                    <span>${typeLabel}</span>
                </div>
            </div>
        `;
        header.classList.add(`column-${columnType}`);
    });
    
    // Wrap the table in a scrollable container
    const scrollableContainer = document.createElement('div');
    scrollableContainer.className = 'sheet-table-container';
    
    // Add scroll hints
    const leftHint = document.createElement('div');
    leftHint.className = 'scroll-hint left';
    leftHint.textContent = '← Scroll left';
    
    const rightHint = document.createElement('div');
    rightHint.className = 'scroll-hint right';
    rightHint.textContent = 'Scroll right →';
    
    scrollableContainer.appendChild(leftHint);
    scrollableContainer.appendChild(rightHint);
    scrollableContainer.appendChild(table);
    
    // Add scroll event listeners to show/hide hints
    scrollableContainer.addEventListener('scroll', function() {
        const { scrollLeft, scrollWidth, clientWidth } = this;
        
        // Show/hide left hint
        if (scrollLeft > 0) {
            leftHint.style.display = 'none';
        } else {
            leftHint.style.display = 'block';
        }
        
        // Show/hide right hint
        if (scrollLeft + clientWidth >= scrollWidth - 10) {
            rightHint.style.display = 'none';
        } else {
            rightHint.style.display = 'block';
        }
    });
    
    // Initial hint visibility
    setTimeout(() => {
        const { scrollWidth, clientWidth } = scrollableContainer;
        if (scrollWidth > clientWidth) {
            rightHint.style.display = 'block';
        }
    }, 100);
    
    tempDiv.innerHTML = '';
    tempDiv.appendChild(scrollableContainer);
    
    return tempDiv.innerHTML;
}

function inferColumnType(columnName) {
    const lowerName = columnName.toLowerCase();
    
    // Check for common numeric patterns
    if (lowerName.includes('id') || lowerName.includes('count') || lowerName.includes('qty') || 
        lowerName.includes('price') || lowerName.includes('amount') || lowerName.includes('freq') ||
        lowerName.includes('avg') || lowerName.includes('sum') || lowerName.includes('total') ||
        lowerName.includes('days') || lowerName.includes('hours') || lowerName.includes('minutes') ||
        lowerName.includes('score') || lowerName.includes('rate') || lowerName.includes('percent')) {
        return 'numeric';
    }
    
    // Check for common date patterns
    if (lowerName.includes('date') || lowerName.includes('time') || lowerName.includes('created') ||
        lowerName.includes('updated') || lowerName.includes('modified') || lowerName.includes('timestamp')) {
        return 'datetime';
    }
    
    // Default to categorical for text-like columns
    if (lowerName.includes('name') || lowerName.includes('title') || lowerName.includes('description') ||
        lowerName.includes('category') || lowerName.includes('type') || lowerName.includes('status')) {
        return 'categorical';
    }
    
    // Final fallback to categorical
    return 'categorical';
}

function getTypeIcon(type) {
    switch(type) {
        case 'numeric': return 'fas fa-hashtag';
        case 'categorical': return 'fas fa-tags';
        case 'datetime': return 'fas fa-calendar';
        default: return 'fas fa-question';
    }
}

function getTypeLabel(type) {
    switch(type) {
        case 'numeric': return 'NUMBER';
        case 'categorical': return 'TEXT';
        case 'datetime': return 'DATE';
        default: return 'TEXT'; // Default unknown to TEXT instead of UNKNOWN
    }
}

// NEW: Load Features List Function
function loadFeaturesList(filename) {
    console.log(`Loading features list for: ${filename}`);
    const contentDiv = document.getElementById(`features-content-${filename}`);
    
    fetch(`/catalog/features/${filename}`)
        .then(r => {
            console.log(`Features response status for ${filename}:`, r.status);
            return r.json();
        })
        .then(data => {
            console.log(`Features data for ${filename}:`, data);
            if (data.success) {
                loadedFeatures[filename] = data.features;
                renderFeaturesList(filename, data.features);
            } else {
                contentDiv.innerHTML = '<div class="error-state">Error loading feature information.</div>';
            }
        })
        .catch(error => {
            console.error('Error loading features:', error);
            contentDiv.innerHTML = '<div class="error-state">Error loading feature information.</div>';
        });
}

function renderFeaturesList(filename, features) {
    const contentDiv = document.getElementById(`features-content-${filename}`);

    const numericFeatures = features.filter(f => f.type === 'Numeric');
    const categoricalFeatures = features.filter(f => f.type === 'Categorical');
    const datetimeFeatures = features.filter(f => f.type === 'Date/Time');

    const filterTabs = `
        <div class="feature-filter-tabs">
            <button class="filter-tab active" onclick="filterFeatures('${filename}', 'all')">
                <i class="fas fa-list"></i> All Features
            </button>
            <button class="filter-tab" onclick="filterFeatures('${filename}', 'Numeric')">
                <i class="fas fa-hashtag"></i> Numeric
            </button>
            <button class="filter-tab" onclick="filterFeatures('${filename}', 'Categorical')">
                <i class="fas fa-tags"></i> Categorical
            </button>
            <button class="filter-tab" onclick="filterFeatures('${filename}', 'Date/Time')">
                <i class="fas fa-calendar"></i> Date/Time
            </button>
        </div>
    `;

    const featuresHTML = features.map(feature => {
        const qualityColor = feature.quality_score >= 80 ? 'good' :
                             feature.quality_score >= 60 ? 'warning' : 'poor';

        let statsHTML = '';
        if (feature.type === 'Numeric' && feature.stats) {
            statsHTML = `
                <div class="feature-stats">
                    ${feature.stats.min !== null ? `<span class="stat">Min: ${feature.stats.min}</span>` : ''}
                    ${feature.stats.max !== null ? `<span class="stat">Max: ${feature.stats.max}</span>` : ''}
                    ${feature.stats.mean !== null ? `<span class="stat">Mean: ${feature.stats.mean}</span>` : ''}
                    ${feature.stats.std !== null ? `<span class="stat">Std: ${feature.stats.std}</span>` : ''}
                </div>
            `;
        } else if (feature.type === 'Categorical' && feature.stats) {
            statsHTML = `
                <div class="feature-stats">
                    <span class="stat">Unique: ${feature.stats.unique_values}</span>
                    ${feature.stats.most_frequent ? `<span class="stat">Most frequent: ${feature.stats.most_frequent}</span>` : ''}
                </div>
            `;
        } else if (feature.type === 'Date/Time' && feature.stats) {
            statsHTML = `
                <div class="feature-stats">
                    ${feature.stats.earliest ? `<span class="stat">From: ${feature.stats.earliest}</span>` : ''}
                    ${feature.stats.latest ? `<span class="stat">To: ${feature.stats.latest}</span>` : ''}
                </div>
            `;
        }

        return `
        <div class="feature-item" data-type="${feature.type}">
            <div class="feature-header">
                <div class="feature-info">
                    <div class="feature-name-section">
                        <i class="${feature.type_icon}"></i>
                        <span class="feature-name">${feature.name}</span>
                        <span class="feature-type-badge ${feature.type.toLowerCase().replace('/', '-')}">${feature.type}</span>
                    </div>
                    <div class="feature-quality">
                        <div class="quality-score ${qualityColor}">
                            <span class="score-value">${feature.quality_score}</span>
                            <span class="score-label">Quality</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="feature-details">
                ${statsHTML}
                <div class="feature-issues">
                    ${feature.quality_issues.map(issue => `<span class="issue-tag ${issue === 'Good quality' ? 'good' : 'warning'}">${issue}</span>`).join('')}
                </div>
            </div>
        </div>
        `;
    }).join('');

    contentDiv.innerHTML = `
        <div class="features-list">
            <div class="features-header">
                <h4><i class="fas fa-list"></i> Features Overview</h4>
                <p class="features-subtitle">Detailed analysis of features in this dataset</p>
            </div>
            ${filterTabs}
            <div class="features-container" id="features-container-${filename}">
                ${featuresHTML}
            </div>
        </div>
    `;
}

function filterFeatures(filename, type) {
    const container = document.getElementById(`features-container-${filename}`);
    const tabs = document.querySelectorAll(`[data-filename="${filename}"] .filter-tab`);
    
    // Update active tab
    tabs.forEach(tab => tab.classList.remove('active'));
    event.target.classList.add('active');
    
    // Filter features
    const featureItems = container.querySelectorAll('.feature-item');
    featureItems.forEach(item => {
        if (type === 'all' || item.dataset.type === type) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

function loadVersionHistory(filename) {
    console.log(`Loading version history for: ${filename}`);
    const contentDiv = document.getElementById(`version-content-${filename}`);
    
    // Fetch real version data from backend
    fetch(`/catalog/versions/${filename}`)
        .then(r => {
            console.log(`Version response status for ${filename}:`, r.status);
            return r.json();
        })
        .then(data => {
            console.log(`Version data for ${filename}:`, data);
            if (data.success && data.versions) {
                loadedVersions[filename] = true;
                renderVersionHistory(filename, data.versions);
            } else {
                contentDiv.innerHTML = '<div class="error-state">No version history available for this file.</div>';
            }
        })
        .catch(error => {
            console.error('Error loading version history:', error);
            contentDiv.innerHTML = '<div class="error-state">Error loading version history.</div>';
        });
}

function goBackToCleaning(filename, version) {
    console.log(`Going back to cleaning for ${filename}, version ${version}`);
    
    if (!confirm(`Go back to cleaning for version ${version}? This will load the current data for editing. When you save, it will update version ${version} instead of creating a new version.`)) {
        return;
    }
    
    // Safer way to get the button element
    const btn = event && event.target ? event.target : null;
    if (!btn) {
        console.error('Could not find button element');
        alert('Error: Could not find button element');
        return;
    }
    
    const originalText = btn.textContent || btn.innerText || 'Go Back to Cleaning';
    btn.disabled = true;
    btn.textContent = 'Loading...';
    
    // Set session data to indicate we're editing this version
    fetch(`/catalog/prepare_cleaning/${filename}/${version}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ 
            edit_mode: true,
            edit_version: version
        })
    })
    .then(r => {
        console.log(`Prepare cleaning response status:`, r.status);
        return r.json();
    })
    .then(data => {
        console.log(`Prepare cleaning response:`, data);
        if (btn) {
            btn.disabled = false;
            btn.textContent = originalText;
        }
        
        if (data.success) {
            // Redirect to cleaning with special parameter
            window.location.href = `/connection/data_ops?edit_version=${version}&filename=${encodeURIComponent(filename)}`;
        } else {
            alert(`❌ ${data.message}`);
        }
    })
    .catch(error => {
        console.error('Error in goBackToCleaning:', error);
        if (btn) {
            btn.disabled = false;
            btn.textContent = originalText;
        }
        alert(`❌ Error: ${error.message}`);
    });
}

function renderVersionHistory(filename, versions) {
    const contentDiv = document.getElementById(`version-content-${filename}`);
    
    if (!versions || versions.length === 0) {
        contentDiv.innerHTML = '<div class="error-state">No version history available.</div>';
        return;
    }
    
    const versionsHTML = versions.map(version => {
        const statusClass = version.status === 'current' ? 'current' : 
                           version.status === 'original' ? 'original' : 'previous';
        
        return `
        <div class="version-item ${statusClass}" data-version="${version.version}">
            <div class="version-header">
                <div class="version-info">
                    <div class="version-number">
                        <i class="fas fa-code-branch"></i>
                        Version ${version.version}
                        ${version.status === 'current' ? '<span class="current-badge">Current</span>' : ''}
                        ${version.status === 'original' ? '<span class="original-badge">Original</span>' : ''}
                    </div>
                    <div class="version-meta">
                        <span class="version-date">
                            <i class="fas fa-clock"></i> ${version.date}
                        </span>
                        <span class="version-user">
                            <i class="fas fa-user"></i> ${version.user}
                        </span>
                    </div>
                </div>
                <div class="version-actions">
                    ${version.version > 1 ? `
                        <button class="version-action-btn cleaning-btn" onclick="goBackToCleaning('${filename}', '${version.version}')">
                            <i class="fas fa-broom"></i> Go Back to Cleaning
                        </button>
                    ` : ''}
                    ${version.status !== 'current' ? `
                        <button class="version-action-btn restore-btn" onclick="restoreVersion('${filename}', '${version.version}')">
                            <i class="fas fa-undo"></i> Make Current
                        </button>
                    ` : ''}
                    <button class="version-action-btn download-btn" onclick="downloadVersion('${filename}', '${version.version}')">
                        <i class="fas fa-download"></i> Download
                    </button>
                </div>
            </div>
            <div class="version-details">
                <div class="version-stats">
                    <span class="version-stat">
                        <i class="fas fa-table-cells"></i> ${version.rows.toLocaleString()} rows
                    </span>
                    <span class="version-stat">
                        <i class="fas fa-columns"></i> ${version.columns} columns
                    </span>
                    <span class="version-stat">
                        <i class="fas fa-weight-hanging"></i> ${version.size}
                    </span>
                </div>
                <div class="version-changes">${version.changes}</div>
                ${version.status === 'current' ? '<div class="current-note"><i class="fas fa-star"></i> This version is used for training and analysis</div>' : ''}
            </div>
        </div>
    `;
    }).join('');
    
    const currentVersion = versions.find(v => v.status === 'current');
    const totalVersions = versions.length;
    
    contentDiv.innerHTML = `
        <div class="version-history">
            <div class="version-history-header">
                <h4><i class="fas fa-history"></i> Version History</h4>
                <p class="version-subtitle">
                    ${totalVersions} version${totalVersions !== 1 ? 's' : ''} tracked. 
                    ${currentVersion ? `Current: v${currentVersion.version}` : 'No current version'}
                </p>
            </div>
            <div class="versions-list">
                ${versionsHTML}
            </div>
        </div>
    `;
}

function restoreVersion(filename, version) {
    if (!confirm(`Are you sure you want to make version ${version} the current version for "${filename}"? This will create a new version and make it the default for training.`)) return;
    
    const btn = event.target;
    const originalText = btn.textContent;
    btn.disabled = true;
    btn.textContent = 'Making Current...';
    
    fetch(`/catalog/restore_version/${filename}/${version}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ user: "user" })
    })
    .then(r => r.json())
    .then(data => {
        btn.disabled = false;
        btn.textContent = originalText;
        
        if (data.success) {
            alert(`✅ ${data.message}\n\nThis version is now the current version used for training.`);
            // Reload version history to show the updated status
            loadedVersions[filename] = false;
            loadVersionHistory(filename);
        } else {
            alert(`❌ ${data.message}`);
        }
    })
    .catch(error => {
        btn.disabled = false;
        btn.textContent = originalText;
        alert(`❌ Error: ${error.message}`);
    });
}

function downloadVersion(filename, version) {
    // Create download link for specific version
    const downloadUrl = `/catalog/download_version/${filename}/${version}`;
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.download = `${filename}_v${version}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function createVersion(filename) {
    const changes = prompt("Enter a description for this version:", "Manual version creation");
    if (!changes) return;
    
    fetch(`/catalog/create_version/${filename}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ 
            changes: changes,
            user: "user"
        })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            alert(`✅ ${data.message}`);
            // Reload version history if it's open
            if (loadedVersions[filename]) {
                loadedVersions[filename] = false;
                loadVersionHistory(filename);
            }
        } else {
            alert(`❌ ${data.message}`);
        }
    })
    .catch(error => {
        alert(`❌ Error: ${error.message}`);
    });
    
    // Close menu
    document.getElementById(`menu-${filename}`).style.display = 'none';
}

function loadDatasetInfo(filename) {
    console.log(`Loading dataset info for: ${filename}`);
    const contentDiv = document.getElementById(`info-content-${filename}`);
    
    fetch(`/catalog/info/${filename}`)
        .then(r => {
            console.log(`Info response status for ${filename}:`, r.status);
            return r.json();
        })
        .then(data => {
            console.log(`Info data for ${filename}:`, data);
            if (data.success) {
                loadedInfo[filename] = data.info;
                renderDatasetInfo(filename, data.info);
            } else {
                contentDiv.innerHTML = '<div class="error-state">Error loading dataset information.</div>';
            }
        })
        .catch(error => {
            console.error('Error loading dataset info:', error);
            contentDiv.innerHTML = '<div class="error-state">Error loading dataset information.</div>';
        });
}

function renderDatasetInfo(filename, info) {
    const contentDiv = document.getElementById(`info-content-${filename}`);
    
    const missingDataHtml = info.missing_data.length > 0 ? 
        info.missing_data.slice(0, 5).map(col => `
            <div class="missing-item">
                <span class="missing-column">${col.column}</span>
                <span class="missing-count">${col.missing_count} (${col.missing_percentage}%)</span>
            </div>
        `).join('') : '<div class="no-missing">✓ No missing data found</div>';
    
    contentDiv.innerHTML = `
        <div class="info-grid">
            <div class="info-card">
                <div class="info-card-header">
                    <i class="fas fa-chart-bar"></i>
                    <h4>Dataset Overview</h4>
                </div>
                <div class="info-stats">
                    <div class="stat-row">
                        <span class="stat-label">Total Rows:</span>
                        <span class="stat-value">${info.rows.toLocaleString()}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Total Features:</span>
                        <span class="stat-value">${info.columns}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">File Size:</span>
                        <span class="stat-value">${info.file_size}</span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Memory Usage:</span>
                        <span class="stat-value">${info.memory_usage}</span>
                    </div>
                </div>
            </div>
            
            <div class="info-card">
                <div class="info-card-header">
                    <i class="fas fa-columns"></i>
                    <h4>Feature Types</h4>
                </div>
                <div class="feature-breakdown">
                    <div class="feature-type-item">
                        <div class="feature-type-icon numeric">
                            <i class="fas fa-hashtag"></i>
                        </div>
                        <div class="feature-type-info">
                            <div class="feature-type-name">Numeric</div>
                            <div class="feature-type-count">${info.numeric_columns} features</div>
                        </div>
                    </div>
                    <div class="feature-type-item">
                        <div class="feature-type-icon categorical">
                            <i class="fas fa-tags"></i>
                        </div>
                        <div class="feature-type-info">
                            <div class="feature-type-name">Categorical</div>
                            <div class="feature-type-count">${info.categorical_columns} features</div>
                        </div>
                    </div>
                    <div class="feature-type-item">
                        <div class="feature-type-icon datetime">
                            <i class="fas fa-calendar"></i>
                        </div>
                        <div class="feature-type-info">
                            <div class="feature-type-name">DateTime</div>
                            <div class="feature-type-count">${info.datetime_columns} features</div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="info-card">
                <div class="info-card-header">
                    <i class="fas fa-shield-alt"></i>
                    <h4>Data Quality</h4>
                </div>
                <div class="quality-indicators">
                    <div class="quality-item">
                        <div class="quality-icon ${info.total_missing > 0 ? 'warning' : 'good'}">
                            <i class="fas ${info.total_missing > 0 ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                        </div>
                        <div class="quality-info">
                            <div class="quality-label">Missing Data</div>
                            <div class="quality-value">${info.total_missing.toLocaleString()} cells (${info.missing_percentage}%)</div>
                        </div>
                    </div>
                    <div class="quality-item">
                        <div class="quality-icon ${info.duplicate_rows > 0 ? 'warning' : 'good'}">
                            <i class="fas ${info.duplicate_rows > 0 ? 'fa-copy' : 'fa-check-circle'}"></i>
                        </div>
                        <div class="quality-info">
                            <div class="quality-label">Duplicate Rows</div>
                            <div class="quality-value">${info.duplicate_rows.toLocaleString()}</div>
                        </div>
                    </div>
                </div>
            </div>
            
            ${info.missing_data.length > 0 ? `
                <div class="info-card full-width">
                    <div class="info-card-header">
                        <i class="fas fa-exclamation-circle"></i>
                        <h4>Missing Data Details</h4>
                    </div>
                    <div class="missing-data-list">
                        ${missingDataHtml}
                        ${info.missing_data.length > 5 ? `<div class="more-missing">+ ${info.missing_data.length - 5} more columns with missing data</div>` : ''}
                    </div>
                </div>
            ` : ''}
        </div>
    `;
}

function deleteFile(btn, filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"? This will also delete all version history.`)) return;
    
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Deleting...';
    
    fetch(`/catalog/delete/${filename}`, {
        method: "POST"
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            // Remove the dataset item from the DOM
            const item = document.querySelector(`[data-filename="${filename}"]`);
            item.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => {
                item.remove();
                updateDatasetCount();
            }, 300);
        } else {
            alert(data.message || "Delete failed.");
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-trash"></i> Delete';
        }
    })
    .catch(() => {
        alert("Delete failed.");
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-trash"></i> Delete';
    });
}

function updateDatasetCount() {
    const count = document.querySelectorAll('.dataset-item').length;
    const countElement = document.querySelector('.dataset-count');
    if (countElement) {
        countElement.textContent = `${count} dataset(s)`;
    }
}

// Close menus when clicking outside
document.addEventListener('click', function(e) {
    if (!e.target.closest('.dropdown-container')) {
        document.querySelectorAll('.dropdown-menu').forEach(menu => {
            menu.style.display = 'none';
        });
    }
});

// Add animations
const style = document.createElement('style');
style.textContent = `
@keyframes slideOut {
    from { opacity: 1; transform: translateX(0); }
    to { opacity: 0; transform: translateX(-100%); }
}
`;
document.head.appendChild(style);