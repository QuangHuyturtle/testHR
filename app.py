from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import json
import os
from datetime import datetime
from werkzeug.utils import secure_filename

# Import HR DSS system
from hr_dss_main import HRDecisionSupportSystem

app = Flask(__name__)
app.secret_key = 'hr_dss_demo_key'
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize HR system
hr_system = HRDecisionSupportSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'POST':
        try:
            accuracy = hr_system.train_model()
            flash(f'Model trained successfully! Accuracy: {accuracy:.3f}', 'success')
            return jsonify({'success': True, 'accuracy': accuracy})
        except Exception as e:
            flash(f'Training failed: {str(e)}', 'error')
            return jsonify({'success': False, 'error': str(e)})
    return render_template('train.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_single():
    if request.method == 'POST':
        try:
            candidate_data = {
                'candidate_id': request.form.get('candidate_id', ''),
                'years_experience': int(request.form.get('years_experience', 0)),
                'education_level': request.form.get('education_level', ''),
                'skills': request.form.get('skills', ''),
                'experience_description': request.form.get('experience_description', ''),
                'position_applied': request.form.get('position_applied', '')
            }
            
            result = hr_system.predict_candidate(candidate_data)
            return render_template('predict.html', result=result, candidate=candidate_data)
            
        except Exception as e:
            flash(f'Prediction failed: {str(e)}', 'error')
            return render_template('predict.html', error=str(e))
    
    return render_template('predict.html')

@app.route('/api/status')
def api_status():
    status = {
        'model_loaded': hr_system.model is not None,
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    return jsonify(status)

if __name__ == '__main__':
    # Train model if not exists
    if hr_system.model is None:
        print("Training initial model...")
        hr_system.train_model()
    
    app.run(debug=True, host='127.0.0.1', port=5000)