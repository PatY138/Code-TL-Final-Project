const form = document.getElementById('chat-form');
const uploadInput = document.getElementById('image-upload');
const uploadStatus = document.getElementById('upload-status');
const preview = document.getElementById('image-preview');
const resultBox = document.getElementById('prediction-result');
const submitButton = form.querySelector('button[type="submit"]');
const API_URL = '/predict';

function resetUI() {
  preview.innerHTML = '<span>No image selected</span>';
  uploadStatus.textContent = 'No image selected';
  resultBox.textContent = 'Waiting for an image...';
  resultBox.className = 'prediction-result';
  uploadInput.value = '';
}

window.addEventListener('load', resetUI);

uploadInput.addEventListener('change', () => {
  const file = uploadInput.files[0];
  uploadStatus.textContent = file ? `Selected: ${file.name}` : 'No image selected';
  resultBox.textContent = file ? 'Ready to upload' : 'Waiting for an image...';
  resultBox.className = file ? 'prediction-result' : 'prediction-result';

  if (!file) {
    preview.innerHTML = '<span>No image selected</span>';
    return;
  }

  const reader = new FileReader();
  reader.onload = () => {
    preview.innerHTML = '';
    const image = document.createElement('img');
    image.src = reader.result;
    image.alt = file.name;
    preview.appendChild(image);
  };
  reader.readAsDataURL(file);
});

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  const file = uploadInput.files[0];
  if (!file) {
    uploadStatus.textContent = 'Please choose an image first.';
    resultBox.textContent = 'Please choose an image first.';
    resultBox.className = 'prediction-result error';
    return;
  }

  const formData = new FormData();
  formData.append('image', file, file.name);

  resultBox.textContent = 'Uploading image...';
  resultBox.className = 'prediction-result loading';
  submitButton.disabled = true;
  submitButton.textContent = '…';

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed');
    }

    resultBox.textContent = data.label ? `Label: ${data.label}` : 'No label returned';
    resultBox.className = 'prediction-result success';
    uploadStatus.textContent = `Uploaded: ${file.name}`;
  } catch (error) {
    resultBox.textContent = `Error: ${error.message}`;
    resultBox.className = 'prediction-result error';
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = '↑';
  }
});
