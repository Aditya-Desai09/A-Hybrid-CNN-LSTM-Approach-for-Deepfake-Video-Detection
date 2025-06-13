document.addEventListener('DOMContentLoaded', function () {
    // DOM Elements
    const registerForm = document.getElementById('registerForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const passwordInput = document.getElementById('password');
    const confirmPasswordInput = document.getElementById('confirmPassword');
    const phoneInput = document.getElementById('phone');
    const termsCheckbox = document.getElementById('terms');
    const registerBtn = document.getElementById('registerBtn');
    const togglePassword = document.getElementById('togglePassword');
    const confirmIcon = document.getElementById('confirmIcon');
    const strengthBar = document.querySelectorAll('.strength-bar .bar-segment');
    const strengthText = document.querySelector('.strength-text');

    const strengthLevels = [
        { color: '#e63946', text: 'Weak' },
        { color: '#f4a261', text: 'Medium' },
        { color: '#2a9d8f', text: 'Strong' }
    ];

    if (togglePassword) {
        togglePassword.addEventListener('click', function () {
            const type = passwordInput.getAttribute('type') === 'password' ? 'text' : 'password';
            passwordInput.setAttribute('type', type);
            this.classList.toggle('fa-eye-slash');
        });
    }

    const toggleConfirmPassword = document.createElement('i');
    toggleConfirmPassword.className = 'fas fa-eye toggle-password';
    toggleConfirmPassword.style.position = 'absolute';
    toggleConfirmPassword.style.right = '15px';
    toggleConfirmPassword.style.cursor = 'pointer';
    toggleConfirmPassword.addEventListener('click', function () {
        const type = confirmPasswordInput.getAttribute('type') === 'password' ? 'text' : 'password';
        confirmPasswordInput.setAttribute('type', type);
        this.classList.toggle('fa-eye-slash');
    });
    confirmPasswordInput.parentNode.insertBefore(toggleConfirmPassword, confirmIcon);

    function showError(id, message) {
        const el = document.getElementById(id);
        if (el) el.textContent = message;
    }

    function clearError(id) {
        const el = document.getElementById(id);
        if (el) el.textContent = '';
    }

    function updatePasswordStrength(strength) {
        strength = Math.max(0, Math.min(strength, 2));
        strengthBar.forEach((bar, idx) => {
            bar.style.backgroundColor = idx <= strength ? strengthLevels[strength].color : '#e0e0e0';
        });
        strengthText.textContent = strengthLevels[strength].text;
        strengthText.style.color = strengthLevels[strength].color;
    }

    function validateUsername() {
        const val = usernameInput.value.trim();
        const valid = /^[A-Za-z0-9]+$/.test(val);
        if (!val) return showError('username-error', 'Username is required'), false;
        if (val.length < 5) return showError('username-error', 'At least 5 characters'), false;
        if (!valid) return showError('username-error', 'Only letters and numbers allowed'), false;
        clearError('username-error'); return true;
    }

    function validateEmail() {
        const val = emailInput.value.trim();
        const valid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(val);
        if (!val) return showError('email-error', 'Email is required'), false;
        if (!valid) return showError('email-error', 'Invalid email'), false;
        clearError('email-error'); return true;
    }

    function validatePassword() {
        const pwd = passwordInput.value;
        const hasUpper = /[A-Z]/.test(pwd);
        const hasNum = /\d/.test(pwd);
        const hasSpecial = /[!@#$%^&*(),.?":{}|<>]/.test(pwd);
        let strength = 0;
        if (!pwd) return updatePasswordStrength(0), showError('password-error', 'Password is required'), false;
        if (pwd.length < 8) return updatePasswordStrength(0), showError('password-error', 'Minimum 8 characters'), false;

        if ((hasUpper && hasNum) || (hasUpper && hasSpecial) || (hasNum && hasSpecial)) strength++;
        if (hasUpper && hasNum && hasSpecial) strength++;
        updatePasswordStrength(strength);

        if (!(hasUpper && hasNum && hasSpecial)) {
            return showError('password-error', 'Include uppercase, number & special char'), false;
        }

        clearError('password-error');
        return true;
    }

    function validateConfirmPassword() {
        const pwd = passwordInput.value;
        const conf = confirmPasswordInput.value;
        if (!conf) return showError('confirm-error', 'Confirm your password'), confirmIcon.classList.remove('valid'), false;
        if (pwd !== conf) return showError('confirm-error', 'Passwords do not match'), confirmIcon.classList.remove('valid'), false;
        clearError('confirm-error'); confirmIcon.classList.add('valid'); return true;
    }

    function validatePhone() {
        const val = phoneInput.value;
        const valid = /^\d{10}$/.test(val);
        if (!val) return showError('phone-error', 'Phone required'), false;
        if (!valid) return showError('phone-error', 'Invalid 10-digit number'), false;
        clearError('phone-error'); return true;
    }

    function validateTerms() {
        if (!termsCheckbox.checked) return showError('terms-error', 'Accept terms'), false;
        clearError('terms-error'); return true;
    }

    function checkFormValidity() {
        const valid =
            validateUsername() && validateEmail() &&
            validatePassword() && validateConfirmPassword() &&
            validatePhone() && validateTerms();
        registerBtn.disabled = !valid;
    }

    usernameInput.addEventListener('input', () => { validateUsername(); checkFormValidity(); });
    emailInput.addEventListener('input', () => { validateEmail(); checkFormValidity(); });
    passwordInput.addEventListener('input', () => {
        validatePassword();
        validateConfirmPassword();
        checkFormValidity();
    });
    confirmPasswordInput.addEventListener('input', () => {
        validateConfirmPassword();
        checkFormValidity();
    });
    phoneInput.addEventListener('input', () => { validatePhone(); checkFormValidity(); });
    termsCheckbox.addEventListener('change', () => { validateTerms(); checkFormValidity(); });

    if (registerForm) {
        registerForm.addEventListener('submit', function (e) {
            // ⛔ Block only if invalid — otherwise let Django handle POST and redirect
            if (
                !validateUsername() || !validateEmail() || !validatePassword() ||
                !validateConfirmPassword() || !validatePhone() || !validateTerms()
            ) {
                e.preventDefault();
            }
        });
    }

    checkFormValidity();  // Run once on load
});
