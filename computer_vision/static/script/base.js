// function toggleSidebar() {
//     const sidebar = document.getElementById("sidebar");
//     const button = document.querySelector(".toggle-button");

//     sidebar.classList.toggle("collapsed");
//     button.classList.toggle("collapsed");

//     button.innerHTML = sidebar.classList.contains("collapsed") ? "❯" : "❮";
// }

// function toggleTheme() {
//     const body = document.body;
//     const themeIcon = document.getElementById('theme-icon');
//     const themeText = document.getElementById('theme-text');

//     body.classList.toggle('dark-theme');

//     if (body.classList.contains('dark-theme')) {
//         themeIcon.textContent = '☀️';
//         themeText.textContent = 'Light Mode';
//         localStorage.setItem('theme', 'dark');
//     } else {
//         themeIcon.textContent = '🌙';
//         themeText.textContent = 'Dark Mode';
//         localStorage.setItem('theme', 'light');
//     }
// }

// document.addEventListener('DOMContentLoaded', function () {
//     const savedTheme = localStorage.getItem('theme');
//     const themeIcon = document.getElementById('theme-icon');
//     const themeText = document.getElementById('theme-text');

//     if (savedTheme === 'dark') {
//         document.body.classList.add('dark-theme');
//         themeIcon.textContent = '☀️';
//         themeText.textContent = 'Light Mode';
//     }
// });

// ====== INITIALIZE THEME ON PAGE LOAD ======
function initializeTheme() {
    const savedTheme = localStorage.getItem('theme');
    const body = document.body;
    const themeIcon = document.getElementById('theme-icon');
    const themeText = document.getElementById('theme-text');
    
    if (savedTheme === 'dark') {
        body.classList.add('dark-theme');
        if (themeIcon) themeIcon.textContent = '☀️';
        if (themeText) themeText.textContent = 'Light Mode';
    } else {
        body.classList.remove('dark-theme');
        if (themeIcon) themeIcon.textContent = '🌙';
        if (themeText) themeText.textContent = 'Dark Mode';
    }
}

// ====== SMOOTH SCROLLING FOR ANCHOR LINKS ======
function initializeSmoothScrolling() {
    const anchors = document.querySelectorAll('a[href^="#"]');
    
    anchors.forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ====== NAVBAR ACTIVE LINK HIGHLIGHTING ======
function updateActiveNavLink() {
    const sections = document.querySelectorAll('section[id]');
    const navLinks = document.querySelectorAll('nav a[href^="#"]');
    
    function highlightActiveSection() {
        let currentSection = '';
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            
            if (pageYOffset >= sectionTop - 200) {
                currentSection = section.getAttribute('id');
            }
        });
        
        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === `#${currentSection}`) {
                link.classList.add('active');
            }
        });
    }
    
    window.addEventListener('scroll', highlightActiveSection);
}

// ====== RESPONSIVE SIDEBAR BEHAVIOR ======
function initializeResponsiveSidebar() {
    const sidebar = document.getElementById('sidebar');
    const toggleButton = document.querySelector('.toggle-button');
    
    function handleResize() {
        if (window.innerWidth <= 768) {
            // Mobile: sidebar should be collapsed by default
            if (sidebar && !sidebar.classList.contains('collapsed')) {
                sidebar.classList.add('collapsed');
                if (toggleButton) {
                    toggleButton.classList.add('collapsed');
                    toggleButton.innerHTML = '❯';
                }
            }
        } else {
            // Desktop: sidebar should be expanded by default
            if (sidebar && sidebar.classList.contains('collapsed')) {
                sidebar.classList.remove('collapsed');
                if (toggleButton) {
                    toggleButton.classList.remove('collapsed');
                    toggleButton.innerHTML = '❮';
                }
            }
        }
    }
    
    window.addEventListener('resize', handleResize);
    handleResize(); // Call once on load
}

// ====== INITIALIZE ALL FUNCTIONALITY ======
document.addEventListener('DOMContentLoaded', function() {
    initializeTheme();
    initializeSmoothScrolling();
    updateActiveNavLink();
    initializeResponsiveSidebar();
    
    // Add some visual feedback for card interactions
    const cards = document.querySelectorAll('.cv-type-card, .application-card');
    cards.forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
});

// ====== KEYBOARD SHORTCUTS ======
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + D to toggle dark mode
    if ((e.ctrlKey || e.metaKey) && e.key === 'd') {
        e.preventDefault();
        toggleTheme();
    }
    
    // Ctrl/Cmd + B to toggle sidebar
    if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault();
        toggleSidebar();
    }
});