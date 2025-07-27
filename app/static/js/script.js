document.addEventListener('DOMContentLoaded', function() {
    // Image preview for disease detection
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    const preview = document.getElementById('imagePreview');
                    if (!preview) {
                        const previewDiv = document.createElement('div');
                        previewDiv.id = 'imagePreview';
                        previewDiv.className = 'text-center my-3';
                        const img = document.createElement('img');
                        img.src = event.target.result;
                        img.className = 'img-fluid upload-preview rounded';
                        previewDiv.appendChild(img);
                        fileInput.parentNode.insertBefore(previewDiv, fileInput.nextSibling);
                    } else {
                        preview.querySelector('img').src = event.target.result;
                    }
                };
                reader.readAsDataURL(file);
            }
        });
    }

    // Form validation
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            const numberInputs = form.querySelectorAll('input[type="number"]');
            numberInputs.forEach(input => {
                if (input.hasAttribute('min') && input.hasAttribute('max')) {
                    const value = parseFloat(input.value);
                    if (value < parseFloat(input.min) || value > parseFloat(input.max)) {
                        alert(`${input.name} must be between ${input.min} and ${input.max}`);
                        e.preventDefault();
                    }
                }
            });
        });
    });
});
