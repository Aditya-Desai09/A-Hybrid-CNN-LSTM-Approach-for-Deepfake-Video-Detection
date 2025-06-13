// Result Page JS for interactivity and enhancements

document.addEventListener('DOMContentLoaded', () => {
    // Optional: Animate sections on scroll
    const sections = document.querySelectorAll('.frames-section, .faces-section, .result-summary');

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.2 });

    sections.forEach(section => {
        section.classList.add('hidden');
        observer.observe(section);
    });

    // Action Buttons
    const backBtn = document.getElementById('goBack');
    const reuploadBtn = document.getElementById('reUpload');

    if (backBtn) {
        backBtn.addEventListener('click', () => {
            window.history.back();
        });
    }

    if (reuploadBtn) {
        reuploadBtn.addEventListener('click', () => {
            window.location.href = '/upload/';
        });
    }
});
