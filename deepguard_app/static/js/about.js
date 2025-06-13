// About Page Specific JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Sticky navigation
    const aboutNav = document.getElementById('aboutNav');
    if (aboutNav) {
        const navOffset = aboutNav.offsetTop;
        
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > navOffset) {
                aboutNav.classList.add('sticky');
            } else {
                aboutNav.classList.remove('sticky');
            }
        });
    }

    // Smooth scrolling for navigation links
    document.querySelectorAll('.about-nav a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            const targetElement = document.querySelector(targetId);
            
            if (targetElement) {
                window.scrollTo({
                    top: targetElement.offsetTop - 100,
                    behavior: 'smooth'
                });
            }
        });
    });

    // Back to top button
    const backToTopBtn = document.getElementById('backToTop');
    if (backToTopBtn) {
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                backToTopBtn.classList.add('visible');
            } else {
                backToTopBtn.classList.remove('visible');
            }
        });
        
        backToTopBtn.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // Contact form submission
    const contactForm = document.querySelector('.contact-form form');
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            alert('Thank you for your message! We will get back to you soon.');
            this.reset();
        });
    }
});