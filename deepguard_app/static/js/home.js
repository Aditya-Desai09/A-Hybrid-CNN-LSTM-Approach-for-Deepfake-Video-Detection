// Homepage specific JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // CTA button functionality
    const ctaButtons = document.querySelectorAll('.cta-button');
    ctaButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Replace with actual functionality
            window.location.href = 'login.html';
        });
    });
    
    // Any other homepage specific JS can go here
});