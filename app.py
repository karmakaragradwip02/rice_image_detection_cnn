from flask import Flask, request, render_template, send_from_directory
import os
from src.RiceImgClassification.pipeline.prediction import PredictionPipeline

app = Flask(__name__)

# Route for the homepage
@app.route('/')
def home():
    return render_template('index.html')

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
        
        return render_template('result.html', image_path=file.filename, predicted_class=predicted_class)

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

if __name__ == '__main__':
    # Ensure the uploads directory exists
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True)
