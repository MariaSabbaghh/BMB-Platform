/**
 * Professional Object Detection Application
 * Manages use cases, model uploads, image processing, and detection workflow
 */

class ObjectDetectionApp {
    constructor() {
        // Initialize data storage (using memory instead of localStorage for Claude.ai compatibility)
        this.useCases = this.loadUseCases();
        this.currentUseCase = null;
        this.uploadedModel = null; // Stores file object
        this.uploadedModelInfo = null; // Stores info from backend upload response
        this.uploadedImage = null; // Stores file object
        this.uploadedImageInfo = null; // Stores info from backend upload response
        this.detectionResults = null;
        
        // Initialize the application
        this.initializeElements();
        this.setupEventListeners();
        this.initializeView();
    }

    /**
     * Initialize all DOM elements
     */
    initializeElements() {
        // Sections
        this.dashboard = document.getElementById('dashboard');
        this.useCasesSection = document.getElementById('use-cases-section');
        this.predefinedSection = document.getElementById('predefined-section');
        this.workflowSection = document.getElementById('workflow-section');
        this.resultsSection = document.getElementById('results-section');
        
        // Dashboard buttons
        this.createNewBtn = document.getElementById('create-new-btn');
        this.viewMyCasesBtn = document.getElementById('view-my-cases-btn');
        this.predefinedCasesBtn = document.getElementById('predefined-cases-btn');
        
        // Navigation buttons
        this.backToDashboard = document.getElementById('back-to-dashboard');
        this.backToDashboardPredefined = document.getElementById('back-to-dashboard-predefined');
        this.backToDashboardWorkflow = document.getElementById('back-to-dashboard-workflow');
        this.createNewFromList = document.getElementById('create-new-from-list');
        this.backToMain = document.getElementById('back-to-main');
        
        // Dashboard elements
        this.caseCount = document.getElementById('case-count');
        this.recentCases = document.getElementById('recent-cases');
        this.predefinedGrid = document.getElementById('predefined-grid');
        
        // Use cases
        this.useCasesGrid = document.getElementById('use-cases-grid');
        this.emptyState = document.getElementById('empty-state');
        
        // Workflow elements
        this.workflowTitle = document.getElementById('workflow-title-text');
        this.workflowSubtitle = document.getElementById('workflow-subtitle');
        this.progressSteps = document.querySelectorAll('.step');
        this.workflowSteps = document.querySelectorAll('.workflow-step');
        
        // Upload elements
        this.modelUploadArea = document.getElementById('model-upload-area');
        this.modelUpload = document.getElementById('model-upload');
        this.modelInfo = document.getElementById('model-info');
        this.imageUploadArea = document.getElementById('image-upload-area');
        this.imageUpload = document.getElementById('image-upload');
        this.imagePreview = document.getElementById('image-preview');
        
        // Detection elements
        this.startDetection = document.getElementById('start-detection');
        this.resultsContent = document.getElementById('results-content');
        this.downloadResults = document.getElementById('download-results');
        this.newDetection = document.getElementById('new-detection');
        
        // Modal elements
        this.createModal = document.getElementById('create-modal');
        this.closeModal = document.getElementById('close-modal');
        this.createForm = document.getElementById('create-form');
        this.cancelCreate = document.getElementById('cancel-create');
        this.confirmCreate = document.getElementById('confirm-create');
        this.useCaseName = document.getElementById('use-case-name');
        this.useCaseDescription = document.getElementById('use-case-description');
        
        // Loading overlay
        this.loadingOverlay = document.getElementById('loading-overlay');
    }

    /**
     * Setup all event listeners
     */
    setupEventListeners() {
        // Dashboard buttons
        this.createNewBtn?.addEventListener('click', () => this.showCreateModal());
        this.viewMyCasesBtn?.addEventListener('click', () => this.showUseCases());
        this.predefinedCasesBtn?.addEventListener('click', () => this.showPredefinedCases());
        
        // Navigation buttons
        this.backToDashboard?.addEventListener('click', () => this.showDashboard());
        this.backToDashboardPredefined?.addEventListener('click', () => this.showDashboard());
        this.backToDashboardWorkflow?.addEventListener('click', () => this.showDashboard());
        this.createNewFromList?.addEventListener('click', () => this.showCreateModal());
        this.backToMain?.addEventListener('click', () => this.showDashboard());
        
        // Modal events
        this.closeModal?.addEventListener('click', () => this.hideCreateModal());
        this.cancelCreate?.addEventListener('click', () => this.hideCreateModal());
        this.confirmCreate?.addEventListener('click', () => this.createUseCase());
        this.createForm?.addEventListener('submit', (e) => {
            e.preventDefault();
            this.createUseCase();
        });
        
        // Modal backdrop close
        this.createModal?.addEventListener('click', (e) => {
            if (e.target === this.createModal) {
                this.hideCreateModal();
            }
        });
        
        // Upload events
        this.modelUploadArea?.addEventListener('click', () => this.modelUpload?.click());
        this.imageUploadArea?.addEventListener('click', () => this.imageUpload?.click());
        this.modelUpload?.addEventListener('change', (e) => this.handleModelUpload(e));
        this.imageUpload?.addEventListener('change', (e) => this.handleImageUpload(e));
        
        // Drag and drop
        this.setupDragAndDrop(this.modelUploadArea, this.modelUpload);
        this.setupDragAndDrop(this.imageUploadArea, this.imageUpload);
        
        // Detection and results
        this.startDetection?.addEventListener('click', () => this.runDetection());
        this.downloadResults?.addEventListener('click', () => this.downloadDetectionResults());
        this.newDetection?.addEventListener('click', () => this.startNewDetection());
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.hideCreateModal();
            }
        });
    }

    /**
     * Load use cases from memory with persistence simulation
     */
    loadUseCases() {
        // Try to load from localStorage for persistence (fallback to memory for Claude.ai)
        try {
            const saved = localStorage.getItem('objectDetectionUseCases');
            if (saved) {
                return JSON.parse(saved);
            }
        } catch (e) {
            // localStorage not available, use memory storage
        }
        
        if (!window.objectDetectionUseCases) {
            window.objectDetectionUseCases = [];
        }
        return window.objectDetectionUseCases;
    }

    /**
     * Save use cases with persistence
     */
    saveUseCases() {
        // Try to save to localStorage for persistence
        try {
            localStorage.setItem('objectDetectionUseCases', JSON.stringify(this.useCases));
        } catch (e) {
            // localStorage not available, use memory storage
        }
        window.objectDetectionUseCases = this.useCases;
    }

    /**
     * Initialize the view based on existing use cases
     */
    initializeView() {
        this.showDashboard();
        this.updateDashboard();
    }

    /**
     * Show dashboard (main landing page)
     */
    showDashboard() {
        this.hideAllSections();
        this.dashboard?.classList.remove('hidden');
        this.updateDashboard();
    }

    /**
     * Update dashboard with current use case counts and recent cases
     */
    updateDashboard() {
        // Update case count
        if (this.caseCount) {
            this.caseCount.textContent = this.useCases.length;
        }
        
        // Update recent cases
        if (this.recentCases) {
            this.recentCases.innerHTML = '';
            
            if (this.useCases.length === 0) {
                this.recentCases.innerHTML = `
                    <div style="text-align: center; color: var(--primary-500); font-size: 0.875rem;">
                        No use cases created yet
                    </div>
                `;
            } else {
                // Show up to 3 most recent cases
                const recentCases = this.useCases
                    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
                    .slice(0, 3);
                
                recentCases.forEach(useCase => {
                    const caseItem = document.createElement('div');
                    caseItem.className = 'recent-case-item';
                    caseItem.innerHTML = `
                        <span class="recent-case-name">${this.escapeHtml(useCase.name)}</span>
                        <span class="recent-case-models">${useCase.models.length} model(s)</span>
                    `;
                    this.recentCases.appendChild(caseItem);
                });
            }
        }
    }

    /**
     * Show predefined use cases
     */
    showPredefinedCases() {
        this.hideAllSections();
        this.predefinedSection?.classList.remove('hidden');
        this.renderPredefinedCases();
    }

    /**
     * Render predefined use cases
     */
    renderPredefinedCases() {
        if (!this.predefinedGrid) return;
        
        const predefinedCases = [
            {
                id: 'predefined-vehicle',
                name: 'Vehicle Detection',
                description: 'Detect cars, trucks, motorcycles, and other vehicles in images',
                category: 'Transportation',
                modelSize: '45MB',
                accuracy: '92%'
            },
            {
                id: 'predefined-person',
                name: 'Person Detection',
                description: 'Identify and locate people in various settings and poses',
                category: 'Human Detection',
                modelSize: '38MB',
                accuracy: '89%'
            },
            {
                id: 'predefined-animal',
                name: 'Animal Detection',
                description: 'Recognize common animals including pets and wildlife',
                category: 'Wildlife',
                modelSize: '52MB',
                accuracy: '87%'
            },
            {
                id: 'predefined-general',
                name: 'General Object Detection',
                description: 'Multi-purpose detector for everyday objects and items',
                category: 'General Purpose',
                modelSize: '67MB',
                accuracy: '85%'
            }
        ];
        
        this.predefinedGrid.innerHTML = '';
        
        predefinedCases.forEach(predefCase => {
            const card = document.createElement('div');
            card.className = 'use-case-card predefined-card';
            card.innerHTML = `
                <div class="use-case-header">
                    <div class="use-case-name">${predefCase.name}</div>
                    <div class="predefined-badge">${predefCase.category}</div>
                </div>
                <div class="use-case-description">${predefCase.description}</div>
                <div class="predefined-stats">
                    <div class="stat-item">
                        <span class="stat-label">Model Size:</span>
                        <span class="stat-value">${predefCase.modelSize}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Accuracy:</span>
                        <span class="stat-value">${predefCase.accuracy}</span>
                    </div>
                </div>
                <div class="use-case-actions">
                    <button class="btn btn-primary btn-small use-predefined" data-id="${predefCase.id}">
                        <i class="fas fa-download"></i>
                        Use This Template
                    </button>
                </div>
            `;
            
            // Add event listener
            const useBtn = card.querySelector('.use-predefined');
            useBtn?.addEventListener('click', () => {
                this.createFromPredefined(predefCase);
            });
            
            this.predefinedGrid.appendChild(card);
        });
    }

    /**
     * Create use case from predefined template
     */
    createFromPredefined(predefinedCase) {
        const useCase = {
            id: Date.now().toString(),
            name: predefinedCase.name,
            description: predefinedCase.description,
            models: [{
                id: 'predefined-model',
                name: `${predefinedCase.name.toLowerCase().replace(' ', '_')}_model.h5`,
                size: predefinedCase.modelSize,
                isPredefined: true,
                uploadedAt: new Date().toISOString()
            }],
            isPredefined: true,
            createdAt: new Date().toISOString()
        };

        this.useCases.push(useCase);
        this.saveUseCases();
        this.showWorkflow(useCase);
    }

    /**
     * Show initial choice screen (backward compatibility)
     */
    showInitialChoice() {
        this.showDashboard();
    }

    /**
     * Show use cases section
     */
    showUseCases() {
        this.hideAllSections();
        this.useCasesSection?.classList.remove('hidden');
        this.renderUseCases();
    }

    /**
     * Show workflow section
     */
    showWorkflow(useCase) {
        this.currentUseCase = useCase;
        this.hideAllSections();
        this.workflowSection?.classList.remove('hidden');
        
        // Update title
        if (this.workflowTitle) {
            this.workflowTitle.textContent = useCase.name;
        }
        if (this.workflowSubtitle) {
            this.workflowSubtitle.textContent = `Follow the steps to complete object detection for: ${useCase.name}`;
        }
        
        // Reset workflow
        this.resetWorkflow();
    }

    /**
     * Hide all main sections
     */
    hideAllSections() {
        this.dashboard?.classList.add('hidden');
        this.useCasesSection?.classList.add('hidden');
        this.predefinedSection?.classList.add('hidden');
        this.workflowSection?.classList.add('hidden');
        this.resultsSection?.classList.add('hidden');
    }

    /**
     * Show create use case modal
     */
    showCreateModal() {
        this.createModal?.classList.remove('hidden');
        this.useCaseName?.focus();
    }

    /**
     * Hide create use case modal
     */
    hideCreateModal() {
        this.createModal?.classList.add('hidden');
        this.createForm?.reset();
    }

    /**
     * Create new use case
     */
    createUseCase() {
        const name = this.useCaseName?.value.trim();
        const description = this.useCaseDescription?.value.trim();

        if (!name) {
            alert('Please enter a use case name');
            this.useCaseName?.focus();
            return;
        }

        const useCase = {
            id: Date.now().toString(),
            name,
            description,
            models: [],
            createdAt: new Date().toISOString()
        };

        this.useCases.push(useCase);
        this.saveUseCases();
        this.hideCreateModal();
        this.updateDashboard(); // Update dashboard counts
        this.showWorkflow(useCase);
    }

    /**
     * Render use cases grid
     */
    renderUseCases() {
        if (!this.useCasesGrid) return;
        
        this.useCasesGrid.innerHTML = '';

        if (this.useCases.length === 0) {
            this.useCasesGrid.appendChild(this.emptyState);
        } else {
            this.useCases.forEach(useCase => {
                const card = this.createUseCaseCard(useCase);
                this.useCasesGrid.appendChild(card);
            });
        }
    }

    /**
     * Create use case card element
     */
    createUseCaseCard(useCase) {
        const card = document.createElement('div');
        card.className = 'use-case-card';
        card.innerHTML = `
            <div class="use-case-header">
                <div class="use-case-name">${this.escapeHtml(useCase.name)}</div>
                <div class="use-case-date">${new Date(useCase.createdAt).toLocaleDateString()}</div>
            </div>
            <div class="use-case-description">
                ${this.escapeHtml(useCase.description || 'No description provided')}
            </div>
            <div class="use-case-models">
                <div class="model-count">${useCase.models.length} model(s)</div>
            </div>
            <div class="use-case-actions">
                <button class="btn btn-primary btn-small select-use-case" data-id="${useCase.id}">
                    <i class="fas fa-play"></i>
                    Select & Run
                </button>
                <button class="btn btn-outline btn-small add-model" data-id="${useCase.id}">
                    <i class="fas fa-plus"></i>
                    Add Model
                </button>
                <button class="btn btn-outline btn-small delete-use-case" data-id="${useCase.id}">
                    <i class="fas fa-trash"></i>
                    Delete
                </button>
            </div>
        `;

        // Add event listeners
        const selectBtn = card.querySelector('.select-use-case');
        const addModelBtn = card.querySelector('.add-model');
        const deleteBtn = card.querySelector('.delete-use-case');

        selectBtn?.addEventListener('click', () => {
            const selectedUseCase = this.useCases.find(uc => uc.id === useCase.id);
            if (selectedUseCase) {
                this.showWorkflow(selectedUseCase);
            }
        });

        addModelBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            const selectedUseCase = this.useCases.find(uc => uc.id === useCase.id);
            if (selectedUseCase) {
                this.showWorkflow(selectedUseCase);
                // Focus on model upload step
                setTimeout(() => {
                    this.modelUploadArea?.scrollIntoView({ behavior: 'smooth' });
                }, 100);
            }
        });

        deleteBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.deleteUseCase(useCase.id);
        });

        return card;
    }

    /**
     * Delete use case with confirmation
     */
    deleteUseCase(useCaseId) {
        if (confirm('Are you sure you want to delete this use case? This action cannot be undone.')) {
            this.useCases = this.useCases.filter(uc => uc.id !== useCaseId);
            this.saveUseCases();
            this.renderUseCases();
            this.updateDashboard(); // Update dashboard counts
            
            // If no use cases left, show dashboard
            if (this.useCases.length === 0) {
                this.showDashboard();
            }
        }
    }

    /**
     * Reset workflow to initial state
     */
    resetWorkflow() {
        this.uploadedModel = null;
        this.uploadedModelInfo = null;
        this.uploadedImage = null;
        this.uploadedImageInfo = null;
        this.detectionResults = null;
        
        // Reset file inputs
        if (this.modelUpload) this.modelUpload.value = '';
        if (this.imageUpload) this.imageUpload.value = '';
        
        // Reset progress steps
        this.updateProgressStep(1);
        
        // Show only first step
        this.showWorkflowStep(1);
        
        // Reset upload areas
        this.resetUploadArea(this.modelUploadArea, 'model');
        this.resetUploadArea(this.imageUploadArea, 'image');
        
        // Hide info sections
        this.modelInfo?.classList.add('hidden');
        this.imagePreview?.classList.add('hidden');
        this.resultsSection?.classList.add('hidden');
    }

    /**
     * Update progress step indicator
     */
    updateProgressStep(activeStep) {
        this.progressSteps.forEach((step, index) => {
            const stepNumber = index + 1;
            step.classList.remove('active', 'completed');
            
            if (stepNumber === activeStep) {
                step.classList.add('active');
            } else if (stepNumber < activeStep) {
                step.classList.add('completed');
            }
        });
    }

    /**
     * Show specific workflow step
     */
    showWorkflowStep(stepNumber) {
        this.workflowSteps.forEach((step, index) => {
            if (index + 1 === stepNumber) {
                step.classList.remove('hidden');
            } else {
                step.classList.add('hidden');
            }
        });
    }

    /**
     * Reset upload area to initial state
     */
    resetUploadArea(area, type) {
        if (!area) return;
        
        const config = {
            model: {
                icon: 'fas fa-cloud-upload-alt',
                title: 'Upload Model File',
                description: 'Click to browse or drag and drop your model file',
                formats: ['.h5', '.pb', '.onnx', '.pt', '.pth']
            },
            image: {
                icon: 'fas fa-image',
                title: 'Upload Image',
                description: 'Click to browse or drag and drop your image',
                formats: ['JPG', 'PNG', 'GIF', 'WebP']
            }
        };

        const typeConfig = config[type];
        if (!typeConfig) return;

        area.innerHTML = `
            <div class="upload-content">
                <div class="upload-icon">
                    <i class="${typeConfig.icon}"></i>
                </div>
                <h4>${typeConfig.title}</h4>
                <p>${typeConfig.description}</p>
                <div class="supported-formats">
                    ${typeConfig.formats.map(format => `<span class="format">${format}</span>`).join('')}
                </div>
            </div>
        `;
    }

    /**
     * Setup drag and drop functionality
     */
    setupDragAndDrop(area, input) {
        if (!area || !input) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, this.preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            area.addEventListener(eventName, () => area.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            area.addEventListener(eventName, () => area.classList.remove('dragover'), false);
        });

        area.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                input.files = files;
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    }

    /**
     * Prevent default drag behaviors
     */
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    /**
     * Handle model file upload
     */
    async handleModelUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        const validExtensions = ['.h5', '.pb', '.onnx', '.pt', '.pth'];
        const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
        
        if (!validExtensions.includes(fileExtension)) {
            alert('Please upload a valid model file (.h5, .pb, .onnx, .pt, .pth)');
            return;
        }

        this.showLoading();
        try {
            const formData = new FormData();
            formData.append('model', file);
            if (this.currentUseCase) {
                formData.append('use_case_id', this.currentUseCase.id);
            }

            const response = await fetch('/object-detection/api/object-detection/upload-model', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.uploadedModel = file; // Keep the original file object
                this.uploadedModelInfo = data.file_info; // Store backend response info
                this.displayModelInfo(this.uploadedModelInfo);
                
                // Save model info to current use case (if applicable)
                if (this.currentUseCase) {
                    const modelInfoForUseCase = {
                        id: this.uploadedModelInfo.unique_filename, // Use unique filename as ID for persistence
                        name: this.uploadedModelInfo.filename,
                        size: this.uploadedModelInfo.size,
                        uploadedAt: this.uploadedModelInfo.upload_time,
                        uniqueFilename: this.uploadedModelInfo.unique_filename // Store unique filename
                    };
                    // Prevent duplicate entries if re-uploading the same model
                    const existingModelIndex = this.currentUseCase.models.findIndex(m => m.uniqueFilename === modelInfoForUseCase.uniqueFilename);
                    if (existingModelIndex === -1) {
                        this.currentUseCase.models.push(modelInfoForUseCase);
                    } else {
                        this.currentUseCase.models[existingModelIndex] = modelInfoForUseCase; // Update existing
                    }
                    this.saveUseCases();
                }
                
                // Move to next step
                this.updateProgressStep(2);
                this.showWorkflowStep(2);
            } else {
                alert(`Model upload failed: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error uploading model:', error);
            alert('An error occurred during model upload. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Display model information
     */
    displayModelInfo(fileInfo) {
        if (!this.modelInfo) return;
        
        this.modelInfo.innerHTML = `
            <h4>
                <i class="fas fa-check-circle"></i>
                Model uploaded successfully
            </h4>
            <p><strong>File:</strong> ${this.escapeHtml(fileInfo.filename)}</p>
            <p><strong>Size:</strong> ${this.formatFileSize(fileInfo.size)}</p>
            <p><strong>Type:</strong> ${fileInfo.type.toUpperCase()} Model</p>
            <p><strong>Status:</strong> Ready for detection</p>
        `;
        this.modelInfo.classList.remove('hidden');
    }

    /**
     * Handle image file upload
     */
    async handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            alert('Please upload a valid image file');
            return;
        }

        this.showLoading();
        try {
            const formData = new FormData();
            formData.append('image', file);

            const response = await fetch('/object-detection/api/object-detection/upload-image', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                this.uploadedImage = file; // Keep the original file object
                this.uploadedImageInfo = data.file_info; // Store backend response info
                this.displayImagePreview(this.uploadedImage, this.uploadedImageInfo); // Pass original file for preview
                
                // Move to next step
                this.updateProgressStep(3);
                this.showWorkflowStep(3);
            } else {
                alert(`Image upload failed: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Error uploading image:', error);
            alert('An error occurred during image upload. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Display image preview
     */
    displayImagePreview(file, fileInfo) { // Now accepts file and fileInfo
        if (!this.imagePreview) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.imagePreview.innerHTML = `
                <img src="${e.target.result}" alt="Uploaded image">
                <div class="image-info">
                    <p><strong>File:</strong> ${this.escapeHtml(fileInfo.filename)}</p>
                    <p><strong>Size:</strong> ${this.formatFileSize(fileInfo.size)}</p>
                    <p><strong>Dimensions:</strong> Loading...</p>
                </div>
            `;
            this.imagePreview.classList.remove('hidden');
            
            // Get image dimensions
            const img = this.imagePreview.querySelector('img');
            img.onload = () => {
                const dimensionsP = this.imagePreview.querySelector('p:last-child');
                if (dimensionsP) {
                    dimensionsP.innerHTML = `<strong>Dimensions:</strong> ${img.naturalWidth} × ${img.naturalHeight} pixels`;
                }
            };
        };
        reader.readAsDataURL(file);
    }

    /**
     * Run object detection
     */
    async runDetection() {
        if (!this.uploadedModelInfo || !this.uploadedImageInfo) {
            alert('Please upload both a model and an image before starting detection');
            return;
        }

        this.showLoading();
        
        try {
            const response = await fetch('/object-detection/api/object-detection/run-detection', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    model_unique_filename: this.uploadedModelInfo.unique_filename,
                    image_unique_filename: this.uploadedImageInfo.unique_filename,
                    use_case_id: this.currentUseCase ? this.currentUseCase.id : null
                })
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Create processed image with bounding boxes using the actual detected objects
                const processedImageUrl = await this.createProcessedImage(this.uploadedImage, data.detection_results.detected_objects);

                this.detectionResults = {
                    originalImage: URL.createObjectURL(this.uploadedImage),
                    processedImage: processedImageUrl,
                    detectedObjects: data.detection_results.detected_objects,
                    processingTime: data.detection_results.processing_time,
                    modelAccuracy: data.detection_results.model_accuracy,
                    timestamp: data.detection_results.timestamp
                };
                this.displayResults();
            } else {
                alert(`Detection failed: ${data.error || 'Unknown error'}`);
            }
        } catch (error) {
            console.error('Detection failed:', error);
            alert('An error occurred during detection. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    /**
     * Create processed image with bounding boxes
     */
    async createProcessedImage(imageFile, detectedObjects) {
        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();
            
            img.onload = () => {
                // Set canvas size to image size
                canvas.width = img.width;
                canvas.height = img.height;
                
                // Draw original image
                ctx.drawImage(img, 0, 0);
                
                // Draw bounding boxes
                detectedObjects.forEach((obj, index) => {
                    // Bbox format from backend is [x, y, width, height]
                    const [x, y, width, height] = obj.bbox;
                    
                    // Set bounding box style
                    ctx.strokeStyle = this.getBoundingBoxColor(index);
                    ctx.lineWidth = 3;
                    ctx.fillStyle = this.getBoundingBoxColor(index) + '20'; // 20% opacity
                    
                    // Draw bounding box
                    ctx.fillRect(x, y, width, height);
                    ctx.strokeRect(x, y, width, height);
                    
                    // Draw label
                    const label = `${obj.name} (${(obj.confidence * 100).toFixed(1)}%)`;
                    ctx.fillStyle = this.getBoundingBoxColor(index);
                    ctx.font = '16px Arial';
                    // Adjust text position to be above the box, or inside if space is limited
                    const textX = x;
                    const textY = y > 10 ? y - 5 : y + 15; // Place above if space, else inside
                    ctx.fillText(label, textX, textY);
                });
                
                // Convert canvas to blob URL
                canvas.toBlob((blob) => {
                    resolve(URL.createObjectURL(blob));
                });
            };
            
            img.src = URL.createObjectURL(imageFile);
        });
    }

    /**
     * Get color for bounding box based on index
     */
    getBoundingBoxColor(index) {
        const colors = [
            '#FF0000', // Red
            '#00FF00', // Green
            '#0000FF', // Blue
            '#FFFF00', // Yellow
            '#FF00FF', // Magenta
            '#00FFFF', // Cyan
            '#FFA500', // Orange
            '#800080', // Purple
            '#FFC0CB', // Pink
            '#A52A2A'  // Brown
        ];
        return colors[index % colors.length];
    }

    /**
     * Display detection results
     */
    displayResults() {
        if (!this.resultsContent || !this.detectionResults) return;
        
        const results = this.detectionResults;
        
        this.resultsContent.innerHTML = `
            <div class="result-images">
                <div class="result-image-container">
                    <h4>Original Image</h4>
                    <img src="${results.originalImage}" alt="Original image">
                    <p><em>Input image for detection</em></p>
                </div>
                <div class="result-image-container">
                    <h4>Detection Results</h4>
                    <img src="${results.processedImage}" alt="Detection results">
                    <p><em>Objects detected with bounding boxes and confidence scores</em></p>
                </div>
            </div>
            <div class="result-info">
                <h4>Detection Summary</h4>
                <div class="detection-summary">
                    <p><strong>Objects Found:</strong> <span>${results.detectedObjects.length}</span></p>
                    <p><strong>Processing Time:</strong> <span>${results.processingTime}</span></p>
                    <p><strong>Model Accuracy:</strong> <span>${results.modelAccuracy}</span></p>
                    <p><strong>Use Case:</strong> <span>${this.escapeHtml(this.currentUseCase?.name || 'Unknown')}</span></p>
                    <p><strong>Model:</strong> <span>${this.escapeHtml(this.uploadedModelInfo?.filename || 'Unknown')}</span></p>
                    <p><strong>Timestamp:</strong> <span>${new Date(results.timestamp).toLocaleString()}</span></p>
                </div>
                
                <div class="detected-objects">
                    <h5>Detected Objects</h5>
                    ${results.detectedObjects.map((obj, index) => `
                        <div class="object-item">
                            <div class="object-info">
                                <span class="object-name">${this.escapeHtml(obj.name)}</span>
                                <span class="object-bbox">Box: [${obj.bbox.join(', ')}]</span>
                            </div>
                            <span class="object-confidence" style="border-left: 4px solid ${this.getBoundingBoxColor(index)};">
                                ${(obj.confidence * 100).toFixed(1)}%
                            </span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `;
        
        this.resultsSection?.classList.remove('hidden');
        setTimeout(() => {
            this.resultsSection?.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }

    /**
     * Download detection results as JSON
     */
    downloadDetectionResults() {
        if (!this.detectionResults) return;
        
        const results = {
            useCase: this.currentUseCase?.name || 'Unknown',
            model: this.uploadedModelInfo?.filename || 'Unknown',
            image: this.uploadedImageInfo?.filename || 'Unknown',
            timestamp: this.detectionResults.timestamp,
            processingTime: this.detectionResults.processingTime,
            modelAccuracy: this.detectionResults.modelAccuracy,
            detectedObjects: this.detectionResults.detectedObjects.map(obj => ({
                name: obj.name,
                confidence: obj.confidence,
                boundingBox: obj.bbox
            }))
        };
        
        const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `detection_results_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    /**
     * Start new detection (reset workflow)
     */
    startNewDetection() {
        this.resetWorkflow();
        this.resultsSection?.classList.add('hidden');
        this.showWorkflowStep(1);
    }

    /**
     * Show loading overlay
     */
    showLoading() {
        this.loadingOverlay?.classList.remove('hidden');
    }

    /**
     * Hide loading overlay
     */
    hideLoading() {
        this.loadingOverlay?.classList.add('hidden');
    }

    /**
     * Format file size in human readable format
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    /**
     * Escape HTML to prevent XSS
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

/**
 * Utility functions for additional functionality
 */

/**
 * Global function to download results (can be called from other modules)
 */
function downloadDetectionResults() {
    if (window.objectDetectionApp && window.objectDetectionApp.detectionResults) {
        window.objectDetectionApp.downloadDetectionResults();
    } else {
        alert('No detection results available to download');
    }
}

/**
 * Global function to reset application state
 */
function resetObjectDetectionApp() {
    if (window.objectDetectionApp) {
        window.objectDetectionApp.showDashboard();
        window.objectDetectionApp.resetWorkflow();
    }
}

/**
 * Global function to get current use cases (for external access)
 */
function getCurrentUseCases() {
    if (window.objectDetectionApp) {
        return window.objectDetectionApp.useCases;
    }
    return [];
}

/**
 * Global function to export use cases data
 */
function exportUseCasesData() {
    const useCases = getCurrentUseCases();
    const exportData = {
        exportDate: new Date().toISOString(),
        version: '1.0',
        useCases: useCases
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `use_cases_export_${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

/**
 * Global function to import use cases data
 */
function importUseCasesData(fileInput) {
    const file = fileInput.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
        try {
            const importData = JSON.parse(e.target.result);
            if (importData.useCases && Array.isArray(importData.useCases)) {
                if (window.objectDetectionApp) {
                    // Merge imported use cases with existing ones
                    const existingIds = window.objectDetectionApp.useCases.map(uc => uc.id);
                    const newUseCases = importData.useCases.filter(uc => !existingIds.includes(uc.id));
                    
                    window.objectDetectionApp.useCases.push(...newUseCases);
                    window.objectDetectionApp.saveUseCases();
                    window.objectDetectionApp.updateDashboard();
                    window.objectDetectionApp.renderUseCases();
                    
                    alert(`Successfully imported ${newUseCases.length} new use cases`);
                }
            } else {
                alert('Invalid import file format');
            }
        } catch (error) {
            alert('Error reading import file: ' + error.message);
        }
    };
    reader.readAsText(file);
}

/**
 * Initialize application when DOM is loaded
 */
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the application
    window.objectDetectionApp = new ObjectDetectionApp();
    
    // Add global event listeners for keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + S to download results
        if ((e.ctrlKey || e.metaKey) && e.key === 's' && window.objectDetectionApp?.detectionResults) {
            e.preventDefault();
            downloadDetectionResults();
        }
        
        // Ctrl/Cmd + H to go to dashboard
        if ((e.ctrlKey || e.metaKey) && e.key === 'h') {
            e.preventDefault();
            if (window.objectDetectionApp) {
                window.objectDetectionApp.showDashboard();
            }
        }
        
        // Ctrl/Cmd + N to create new use case
        if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
            e.preventDefault();
            if (window.objectDetectionApp) {
                window.objectDetectionApp.showCreateModal();
            }
        }
    });
});

/**
 * Handle window resize for responsive design
 */
window.addEventListener('resize', () => {
    // Debounce resize events
    clearTimeout(window.resizeTimeout);
    window.resizeTimeout = setTimeout(() => {
        // Add any resize-specific logic here if needed
        if (window.objectDetectionApp && window.objectDetectionApp.detectionResults) {
            // Re-render results if they're currently displayed
            window.objectDetectionApp.displayResults();
        }
    }, 250);
});

/**
 * Handle page visibility changes (for cleanup)
 */
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Page is now hidden - cleanup any ongoing processes
        if (window.objectDetectionApp) {
            window.objectDetectionApp.hideLoading();
        }
    }
});

/**
 * Handle before page unload (for cleanup)
 */
window.addEventListener('beforeunload', (e) => {
    // Clean up any blob URLs to prevent memory leaks
    if (window.objectDetectionApp && window.objectDetectionApp.detectionResults) {
        try {
            if (window.objectDetectionApp.detectionResults.originalImage) {
                URL.revokeObjectURL(window.objectDetectionApp.detectionResults.originalImage);
            }
            if (window.objectDetectionApp.detectionResults.processedImage) {
                URL.revokeObjectURL(window.objectDetectionApp.detectionResults.processedImage);
            }
        } catch (error) {
            console.log('Cleanup error:', error);
        }
    }
});

/**
 * Export functionality for integration with other modules
 */
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        ObjectDetectionApp,
        downloadDetectionResults,
        resetObjectDetectionApp,
        getCurrentUseCases,
        exportUseCasesData,
        importUseCasesData
    };
}

// Also make the class available globally
window.ObjectDetectionApp = ObjectDetectionApp;
