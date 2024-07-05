from flask import Flask, request, render_template
import os
from RiceImgClassification.pipeline import PredictionPipeline

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('templates/index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        
        # Create a PredictionPipeline object and predict the class
        prediction_pipeline = PredictionPipeline(file_path)
        predicted_class = prediction_pipeline.predict()
        
        return f'The predicted class is: {predicted_class}'

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True)
