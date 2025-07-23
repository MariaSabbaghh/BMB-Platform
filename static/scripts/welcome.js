function selectPredictiveAI() {
    document.body.classList.add('fade-out');
    setTimeout(() => {
        window.open('http://localhost:5001', '_blank');
    }, 300);
}

function selectGenerativeAI() {
    alert('Generative AI module coming soon!');
}

function selectComputerVision() {
    document.body.classList.add('fade-out');
    setTimeout(() => {
        window.open('http://localhost:5002', '_blank');
    }, 300);
}
// Logo refresh functionality
function refreshWebsite() {
    // Add a subtle loading effect
    const logo = document.querySelector('.header-logo');
    logo.style.transform = 'scale(0.95)';
    
    // Refresh the page after a brief animation
    setTimeout(() => {
        window.location.reload();
    }, 150);
}

// Welcome page functionality
function selectGenerativeAI() {
    // Add loading animation
    document.body.classList.add('fade-out');
    
    // Add a slight delay for smooth transition
    setTimeout(() => {
        // Redirect to the main website (home page)
        window.location.href = '/generative';
    }, 300);
}

function selectPredictiveAI() {
    // Add loading animation
    document.body.classList.add('fade-out');
    
    // Add a slight delay for smooth transition
    setTimeout(() => {
        // Redirect to predictive AI section (you can customize this URL)
        window.location.href = '/home';
    }, 300);
}

// Add some interactive effects
document.addEventListener('DOMContentLoaded', function() {
    const aiOptions = document.querySelectorAll('.ai-option');
    
    aiOptions.forEach(option => {
        // Add click animation
        option.addEventListener('click', function() {
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
        
        // Add keyboard support
        option.setAttribute('tabindex', '0');
        option.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    });
    
    // Add professional interaction effects
    addProfessionalEffects();
});

// Professional interaction effects
function addProfessionalEffects() {
    // Add smooth scroll behavior
    document.documentElement.style.scrollBehavior = 'smooth';
    
    // Add loading states for better UX
    const options = document.querySelectorAll('.ai-option');
    options.forEach(option => {
        option.addEventListener('click', function() {
            // Add loading state
            const button = this.querySelector('.cta-button span');
            const originalText = button.textContent;
            button.textContent = 'Loading...';
            
            // Restore after navigation starts
            setTimeout(() => {
                button.textContent = originalText;
            }, 500);
        });
    });
}