document.addEventListener('DOMContentLoaded', function () {
    const uploadBox = document.getElementById('uploadBox');
    const browseBtn = document.getElementById('browseBtn');
    const fileInput = document.getElementById('fileInput');
    const uploadBtn = document.getElementById('uploadBtn');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const errorMessage = document.getElementById('errorMessage');
    const fileName = document.getElementById('fileName');
    const fileSize = document.getElementById('fileSize');
    const fileDuration = document.getElementById('fileDuration');
    const filePreview = document.getElementById('filePreview');
    const removeBtn = document.getElementById('removeBtn');
    const continueBtn = document.getElementById('continueBtn');
    const uploadContent = document.getElementById('uploadContent');
    const uploadProgress = document.getElementById('uploadProgress');
    const uploadSuccess = document.getElementById('uploadSuccess');

    let selectedFile = null;
    let uploadInProgress = false;

    browseBtn.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);
    removeBtn.addEventListener('click', resetForm);
    continueBtn.addEventListener('click', () => window.location.href = "/result/");

    uploadBox.addEventListener('dragover', function (e) {
        e.preventDefault();
        uploadBox.style.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--primary-color');
        uploadBox.style.backgroundColor = 'rgba(67, 97, 238, 0.05)';
    });

    uploadBox.addEventListener('dragleave', function () {
        uploadBox.style.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--divider-color');
        uploadBox.style.backgroundColor = 'transparent';
    });

    uploadBox.addEventListener('drop', function (e) {
        e.preventDefault();
        uploadBox.style.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--divider-color');
        uploadBox.style.backgroundColor = 'transparent';

        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            handleFileSelect({ target: fileInput });
        }
    });

    uploadBtn.addEventListener('click', handleUpload);

    function handleFileSelect(e) {
    const file = e.target.files[0];
    if (!file) return;
    resetForm();
    hideError();

    const extension = file.name.split('.').pop().toLowerCase();
    const allowedExtensions = ['mp4', 'mov', 'avi'];

    // Log actual file type (for debug)
    console.log("Selected File:", file);
    console.log("Detected MIME type:", file.type);
    console.log("Extension:", extension);

    if (!allowedExtensions.includes(extension)) {
        showError('Unsupported file extension. Only MP4, MOV, and AVI are allowed.');
        return;
    }

    if (file.size > 300 * 1024 * 1024) {
        showError('File size exceeds the 300 MB limit');
        return;
    }

    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatFileSize(file.size);

    const video = document.createElement('video');
    video.preload = 'metadata';

    video.onloadedmetadata = function () {
        window.URL.revokeObjectURL(video.src);

        if (video.duration > 120) {
            showError('Video duration exceeds the 2 minute limit');
            resetForm();
            return;
        }

        fileDuration.textContent = formatDuration(video.duration);
        createThumbnail(video);

        uploadContent.classList.add('hidden');
        uploadProgress.classList.remove('hidden');
    };

    video.onerror = function () {
        showError('Error reading video file. It may be corrupted or unsupported.');
        resetForm();
    };

    video.src = URL.createObjectURL(file);
}


    function createThumbnail(video) {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = 160;
        canvas.height = 90;

        video.currentTime = 1;
        video.onseeked = function () {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            filePreview.innerHTML = '';
            filePreview.appendChild(canvas);
        };
    }

    function handleUpload() {
        if (!selectedFile || uploadInProgress) return;

        const formData = new FormData();
        formData.append('file', selectedFile);
        const scrollValue = document.getElementById('scrollValue')?.value || '';
        formData.append('scrollValue', scrollValue);

        const xhr = new XMLHttpRequest();
        xhr.open('POST', '/upload/', true);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
        xhr.setRequestHeader('X-CSRFToken', getCookie('csrftoken'));

        xhr.upload.onprogress = function (e) {
            if (e.lengthComputable) {
                const percent = (e.loaded / e.total) * 100;
                updateProgress(percent);
            }
        };

        xhr.onload = function () {
            if (xhr.status === 200) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.redirect) {
                        window.location.href = response.redirect;
                    } else {
                        uploadComplete();
                    }
                } catch (e) {
                    showError('Error processing server response');
                    resetForm();
                }
            } else {
                showError('Upload failed: ' + xhr.responseText);
                resetForm();
            }
        };

        xhr.onerror = function () {
            showError('Network error during upload');
            resetForm();
        };

        xhr.send(formData);
        uploadInProgress = true;
        uploadBtn.disabled = true;
    }

    function updateProgress(progress) {
        progressBar.style.width = progress + '%';
        progressText.textContent = Math.round(progress) + '% uploaded';
    }

    function uploadComplete() {
        uploadInProgress = false;
        uploadProgress.classList.add('hidden');
        uploadSuccess.classList.remove('hidden');
    }

    function resetForm() {
        if (uploadInProgress) return;

        fileInput.value = '';
        selectedFile = null;
        uploadBtn.disabled = false;
        uploadInProgress = false;

        uploadProgress.classList.add('hidden');
        uploadSuccess.classList.add('hidden');
        uploadContent.classList.remove('hidden');

        hideError();
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
    }

    function hideError() {
        errorMessage.classList.add('hidden');
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    function formatDuration(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return mins + ':' + (secs < 10 ? '0' : '') + secs;
    }

    function getCookie(name) {
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let cookie of cookies) {
                cookie = cookie.trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
});
