document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnLoader = document.getElementById('btn-loader');
    const resultsSection = document.getElementById('results-section');
    const resultCard = document.getElementById('result-card');
    const predictionText = document.getElementById('prediction-text');
    const probabilityText = document.getElementById('probability-text');
    const probabilityFill = document.getElementById('probability-fill');
    const shapImage = document.getElementById('shap-image');
    const shapSection = document.getElementById('shap-section');
    const historyBody = document.getElementById('history-body');

    let historyData = [];

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // UI Loading State
        submitBtn.querySelector('span').style.display = 'none';
        btnLoader.style.display = 'block';
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.error) {
                alert(result.error);
                return;
            }

            // Show results
            resultsSection.classList.remove('hidden');

            // Set Prediction text and styling
            if (result.prediction === 1) {
                predictionText.textContent = "The model predicts the positive class.";
                resultCard.className = "result-card positive";
            } else {
                predictionText.textContent = "The model predicts the negative class.";
                resultCard.className = "result-card negative";
            }

            // Set Probability
            const probPercent = (result.probability * 100).toFixed(2);
            probabilityText.textContent = `${probPercent}%`;
            
            // Small delay for CSS animation
            setTimeout(() => {
                probabilityFill.style.width = `${probPercent}%`;
            }, 100);

            // Set SHAP Image
            if (result.shap_image) {
                shapImage.src = `data:image/png;base64,${result.shap_image}`;
                shapSection.classList.remove('hidden');
            } else {
                shapSection.classList.add('hidden');
            }

            // Update History
            historyData.unshift({
                glucose: data.Glucose,
                bmi: data.BMI,
                age: data.Age,
                prediction: result.prediction,
                probability: probPercent
            });

            updateHistoryTable();

        } catch (error) {
            console.error('Error:', error);
            alert("Failed to get prediction. Ensure the backend is running and model is trained.");
        } finally {
            // Restore UI state
            submitBtn.querySelector('span').style.display = 'block';
            btnLoader.style.display = 'none';
            submitBtn.disabled = false;
            
            // Scroll to results smoothly
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });

    function updateHistoryTable() {
        historyBody.innerHTML = '';
        historyData.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${row.glucose}</td>
                <td>${row.bmi}</td>
                <td>${row.age}</td>
                <td>${row.prediction}</td>
                <td>${row.probability}%</td>
            `;
            historyBody.appendChild(tr);
        });
    }
});
