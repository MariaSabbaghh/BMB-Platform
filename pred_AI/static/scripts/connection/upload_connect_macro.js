// Upload Connect Macro JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // File upload functionality
    const dropArea = document.getElementById('connection-drop-area');
    const fileInput = document.getElementById('connection-csv-input');
    const hiddenFileInput = document.getElementById('connection-csv-input-hidden');
    const selectedFileName = document.getElementById('connection-selected-file');
    const uploadForm = document.getElementById('connection-csv-upload-form');
    
    // Database connection functionality
    const dbUriInput = document.getElementById('db-uri-input');
    const dbUriHidden = document.getElementById('db-uri-hidden');
    const dbForm = document.getElementById('connection-db-connect-form');

    // File Upload Handlers
    if (dropArea && fileInput && hiddenFileInput) {
        // Click to select file
        dropArea.addEventListener('click', () => {
            fileInput.click();
        });

        // File selection handler
        fileInput.addEventListener('change', function(e) {
            handleFileSelect(e.target.files[0]);
        });

        // Drag and drop handlers
        dropArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropArea.classList.add('highlight');
        });

        dropArea.addEventListener('dragleave', () => {
            dropArea.classList.remove('highlight');
        });

        dropArea.addEventListener('drop', (e) => {
            e.preventDefault();
            dropArea.classList.remove('highlight');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });

        function handleFileSelect(file) {
            if (file) {
                // Validate file type
                const allowedTypes = ['.csv', '.xls', '.xlsx'];
                const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
                
                if (!allowedTypes.includes(fileExtension)) {
                    alert('Please select a CSV or Excel file (.csv, .xls, .xlsx)');
                    return;
                }

                // Display selected file name
                selectedFileName.textContent = `Selected: ${file.name}`;
                selectedFileName.style.display = 'block';

                // Transfer file to hidden input in the form
                const dt = new DataTransfer();
                dt.items.add(file);
                hiddenFileInput.files = dt.files;
            }
        }

        // Form submission handler
        uploadForm.addEventListener('submit', function(e) {
            if (!hiddenFileInput.files || hiddenFileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select a file to upload');
                return false;
            }
        });
    }

    // Database Connection Handlers
    if (dbUriInput && dbUriHidden && dbForm) {
        // Transfer input value to hidden field on form submission
        dbForm.addEventListener('submit', function(e) {
            const uriValue = dbUriInput.value.trim();
            
            if (!uriValue) {
                e.preventDefault();
                alert('Please enter a database URI');
                return false;
            }
            
            // Transfer value to hidden input
            dbUriHidden.value = uriValue;
        });

        // Update hidden field when input changes
        dbUriInput.addEventListener('input', function() {
            dbUriHidden.value = this.value;
        });
    }

    // Add loading states to buttons
    const uploadButton = document.querySelector('.btn-upload');
    const connectButton = document.querySelector('.btn-connect');

    if (uploadButton) {
        uploadForm?.addEventListener('submit', function() {
            uploadButton.textContent = 'Uploading...';
            uploadButton.disabled = true;
        });
    }

    if (connectButton) {
        dbForm?.addEventListener('submit', function() {
            connectButton.textContent = 'Connecting...';
            connectButton.disabled = true;
        });
    }

    console.log('Upload Connect Macro JavaScript loaded successfully');
});