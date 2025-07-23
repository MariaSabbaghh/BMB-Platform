document.addEventListener("DOMContentLoaded", function() {
    console.log("Page loaded, initializing upload functionality");
    
    var form = document.getElementById("connection-csv-upload-form");
    var uploadSection = document.getElementById("upload-section");
    var uploadedFileInfo = document.getElementById("uploaded-file-info");
    var continueOptions = document.getElementById("continue-options");
    var successBox = document.getElementById("success-box");

    console.log("Elements found:", {
        form: !!form,
        uploadSection: !!uploadSection,
        uploadedFileInfo: !!uploadedFileInfo,
        continueOptions: !!continueOptions,
        successBox: !!successBox
    });

    if (form && uploadSection && uploadedFileInfo && continueOptions && successBox) {
        form.addEventListener("submit", function(e) {
            e.preventDefault();
            console.log("Form submitted, starting upload...");
            
            // Show loading state
            var submitBtn = form.querySelector('button[type="submit"]');
            var originalText = submitBtn.textContent;
            submitBtn.textContent = "Uploading...";
            submitBtn.disabled = true;
            
            var formData = new FormData(form);
            console.log("FormData created, making fetch request to:", form.action);
            
            fetch(form.action, {
                method: "POST",
                body: formData
            })
            .then(response => {
                console.log("Response status:", response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                console.log("Upload response:", data);
                
                if (data.success) {
                    console.log("Upload successful, showing success message");
                    
                    // Store the uploaded file info globally for other scripts to use
                    window.uploadedFileData = {
                        filename: data.filename,
                        columns: data.columns || []
                    };
                    
                    // Fill the success box inside the centered unit
                    successBox.innerHTML = `
                        <strong>✅ File uploaded successfully!</strong><br>
                        <span>File: <a href="/uploads/${encodeURIComponent(data.filename)}" target="_blank" style="color: #1890ff;">${data.filename}</a></span><br>
                        ${data.columns ? `<small>Columns detected: ${data.columns.length} columns</small>` : ''}
                    `;
                    
                    // Hide the upload section and show continue options
                    uploadSection.style.display = "none";
                    continueOptions.style.display = "block";
                    
                    console.log("Success message displayed, navigation buttons shown");
                } else {
                    console.error("Upload failed:", data.message);
                    alert(data.message || "Upload failed. Please try again.");
                    submitBtn.textContent = originalText;
                    submitBtn.disabled = false;
                }
            })
            .catch(error => {
                console.error("Upload error:", error);
                alert("Upload failed: " + error.message);
                submitBtn.textContent = originalText;
                submitBtn.disabled = false;
            });
        });

        // Handle outlier analysis button click
        var outlierBtn = document.getElementById("outlier-analysis-btn");
        if (outlierBtn) {
            outlierBtn.addEventListener("click", function() {
                console.log("Navigating to outlier analysis");
                window.location.href = "/connection/outlier_analysis";
            });
        }

        // Handle correlation analysis button click
        var correlationBtn = document.getElementById("correlation-analysis-btn");
        if (correlationBtn) {
            correlationBtn.addEventListener("click", function() {
                console.log("Navigating to correlation analysis");
                window.location.href = "/connection/correlation_analysis";
            });
        }

        // Handle data operations button click  
        var dataOpsBtn = document.getElementById("data-ops-btn");
        if (dataOpsBtn) {
            dataOpsBtn.addEventListener("click", function() {
                console.log("Navigating to data operations");
                window.location.href = "/connection/data_ops";
            });
        }
    } else {
        console.error("Missing required elements for upload functionality");
    }
});