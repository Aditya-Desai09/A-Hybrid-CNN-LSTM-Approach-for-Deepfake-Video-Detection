// Login Page Specific JavaScript
document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const togglePassword = document.getElementById('togglePassword');
    const passwordInput = document.getElementById('password');
    const loginForm = document.getElementById('loginForm');
    const usernameInput = document.getElementById('username');
    const usernameError = document.getElementById('username-error');
    const passwordError = document.getElementById('password-error');
    const loginBtn = document.getElementById('loginBtn');
    const loginText = document.getElementById('loginText');
    const spinner = document.getElementById('spinner');

    // Password visibility toggle
    if (togglePassword && passwordInput) {
        togglePassword.addEventListener('click', function () {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            this.classList.toggle('fa-eye-slash');
        });
    }

    // Username validation
    function validateUsername() {
        const username = usernameInput.value.trim();
        if (username.length === 0) {
            usernameError.textContent = 'Username is required';
            usernameInput.style.borderColor = '#e63946';
            return false;
        } else if (username.length < 5) {
            usernameError.textContent = 'Username must be at least 5 characters';
            usernameInput.style.borderColor = '#e63946';
            return false;
        } else {
            usernameError.textContent = '';
            usernameInput.style.borderColor = '#2a9d8f';
            return true;
        }
    }

    // Password validation
    function validatePassword() {
        const password = passwordInput.value;
        if (password.length === 0) {
            passwordError.textContent = 'Password is required';
            passwordInput.style.borderColor = '#e63946';
            return false;
        }
        passwordError.textContent = '';
        passwordInput.style.borderColor = '#2a9d8f';
        return true;
    }

    // Enable/disable login button
    function checkFormValidity() {
        const isUsernameValid = validateUsername();
        const isPasswordValid = validatePassword();
        if (loginBtn) {
            loginBtn.disabled = !(isUsernameValid && isPasswordValid);
        }
    }

    if (usernameInput && passwordInput) {
        usernameInput.addEventListener('input', checkFormValidity);
        passwordInput.addEventListener('input', checkFormValidity);
    }

    // ✅ Fixed: This block now has proper closing and no e.preventDefault() if valid
    if (loginForm) {
        loginForm.addEventListener('submit', function (e) {
            const validUser = validateUsername();
            const validPass = validatePassword();

            if (!validUser || !validPass) {
                e.preventDefault(); // Stop form submission on invalid input
                return;
            }

            // Allow default form submission, but show spinner
            loginBtn.disabled = true;
            loginText.classList.add('hidden');
            spinner.classList.remove('hidden');
        });
    }

    // Social login placeholders
    const googleBtn = document.querySelector('.google-btn');
    const githubBtn = document.querySelector('.github-btn');

    if (googleBtn) {
        googleBtn.addEventListener('click', function () {
            alert('Redirecting to Google authentication...');
        });
    }

    if (githubBtn) {
        githubBtn.addEventListener('click', function () {
            alert('Redirecting to GitHub authentication...');
        });
    }
});
