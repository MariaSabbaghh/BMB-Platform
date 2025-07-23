       function toggleSidebar() {
            const sidebar = document.getElementById("sidebar");
            const button = document.querySelector(".toggle-button");

            sidebar.classList.toggle("collapsed");
            button.classList.toggle("collapsed");

            button.innerHTML = sidebar.classList.contains("collapsed") ? "❯" : "❮";
        }

        function toggleTheme() {
            const body = document.body;
            const themeIcon = document.getElementById('theme-icon');
            const themeText = document.getElementById('theme-text');

            body.classList.toggle('dark-theme');

            if (body.classList.contains('dark-theme')) {
                themeIcon.textContent = '☀️';
                themeText.textContent = 'Light Mode';
                localStorage.setItem('theme', 'dark');
            } else {
                themeIcon.textContent = '🌙';
                themeText.textContent = 'Dark Mode';
                localStorage.setItem('theme', 'light');
            }
        }

        document.addEventListener('DOMContentLoaded', function () {
            const savedTheme = localStorage.getItem('theme');
            const themeIcon = document.getElementById('theme-icon');
            const themeText = document.getElementById('theme-text');

            if (savedTheme === 'dark') {
                document.body.classList.add('dark-theme');
                themeIcon.textContent = '☀️';
                themeText.textContent = 'Light Mode';
            }
        });