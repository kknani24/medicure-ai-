from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import tempfile
import shutil

# Initialize Flask app
app = Flask(__name__)

# Configuration for Render deployment
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Use temporary directory for file uploads (Render-compatible)
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model configurations with relative paths
MODELS = {
    'liver': {
        'path': './models/liver_tumour.pt',
        'classes': ["liver", "tumor"]
    },
    'brain_mri': {
        'path': './models/brain_mri.pt',
        'classes': ["tumour", "eye"]
    },
    'eye': {
        'path': './models/eye.pt',
        'classes': ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
    },
    'kidney': {
        'path': './models/kidney.pt',
        'classes': ["stone"]
    },
    'lung': {
        'path': './models/lung_cancer.pt',
        'classes': ["Tumour"]
    }
}

# Initialize models dictionary
models = {}

def load_models():
    """Load YOLO models with error handling"""
    global models
    for name, config in MODELS.items():
        try:
            if os.path.exists(config['path']):
                models[name] = YOLO(config['path'])
                print(f"Successfully loaded {name} model")
            else:
                print(f"Warning: Model file {config['path']} not found")
        except Exception as e:
            print(f"Error loading {name} model: {e}")

# Load models at startup
load_models()

# Configure Google Gemini API
API_KEY = os.environ.get('GEMINI_API_KEY')
if API_KEY:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
else:
    print("Warning: GEMINI_API_KEY not found in environment variables")
    gemini_model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, model_name=None):
    """Process image with YOLO models"""
    if not models:
        return None, [{'error': 'No models available'}]
    
    # If no specific model is chosen, try all available models
    if model_name is None:
        results_all = {}
        for name, model in models.items():
            try:
                results = model.predict(source=image_path, conf=0.25)
                results_all[name] = {
                    'results': results[0],
                    'classes': MODELS[name]['classes']
                }
            except Exception as e:
                print(f"Error processing with {name} model: {e}")
        
        if results_all:
            return process_multiple_model_results(image_path, results_all)
        else:
            return None, [{'error': 'No models could process the image'}]
    
    # If a specific model is chosen
    if model_name not in models:
        return None, [{'error': f'Model {model_name} not available'}]
    
    model = models[model_name]
    class_names = MODELS[model_name]['classes']
    
    try:
        # Make prediction
        results = model.predict(source=image_path, conf=0.25)
        result = results[0]
        
        # Load and process image
        img = cv2.imread(image_path)
        if img is None:
            return None, [{'error': 'Could not load image'}]
        
        # Store detections info
        detections = []
        
        # Draw boxes and apply pseudo-coloring
        for detection in result.boxes:
            x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
            confidence = float(detection.conf[0])
            class_id = int(detection.cls[0])
            
            # Get class name
            class_label = class_names[class_id] if class_id < len(class_names) else "Unknown"

            # Store detection info
            detections.append({
                'class': class_label,
                'confidence': confidence,
                'bbox': [x_min, y_min, x_max, y_max]
            })
            
            # Define color
            color = (0, 255, 0)  # Default green
            if model_name == 'liver':
                color = (0, 255, 0) if class_label == "liver" else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            # Add text label
            text = f"{class_label.capitalize()} ({confidence:.2f})"
            cv2.putText(img, text, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Pseudo-coloring for specific cases
            if class_label.lower() in ["tumor", "tumour", "cancer", "nodule", "stone"]:
                try:
                    detected_area = img[y_min:y_max, x_min:x_max]
                    if detected_area.size > 0:
                        gray_area = cv2.cvtColor(detected_area, cv2.COLOR_BGR2GRAY)
                        pseudo_colored_area = cv2.applyColorMap(gray_area, cv2.COLORMAP_JET)
                        img[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                            detected_area, 0.5, pseudo_colored_area, 0.5, 0
                        )
                except Exception as e:
                    print(f"Error applying pseudo-coloring: {e}")

        # Save processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                  'processed_' + os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        
        return output_path, detections
        
    except Exception as e:
        print(f"Error processing image with {model_name}: {e}")
        return None, [{'error': f'Processing failed: {str(e)}'}]

def process_multiple_model_results(image_path, results_all):
    """Process results from multiple models"""
    try:
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            return None, {'error': 'Could not load image'}
        
        # Collect all detections
        all_detections = {}
        
        # Process results from each model
        for model_name, model_results in results_all.items():
            result = model_results['results']
            class_names = model_results['classes']
            
            model_detections = []
            
            for detection in result.boxes:
                x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
                confidence = float(detection.conf[0])
                class_id = int(detection.cls[0])
                
                # Get class name
                class_label = class_names[class_id] if class_id < len(class_names) else "Unknown"

                # Store detection info
                model_detections.append({
                    'class': class_label,
                    'confidence': confidence,
                    'bbox': [x_min, y_min, x_max, y_max]
                })
                
                # Define color
                color = (255, 0, 0)  # Default blue
                
                # Draw bounding box
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

                # Add text label
                text = f"{model_name.capitalize()}: {class_label.capitalize()} ({confidence:.2f})"
                cv2.putText(img, text, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Store detections for this model
            all_detections[model_name] = model_detections

        # Save processed image
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                  'processed_multi_' + os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        
        return output_path, all_detections
        
    except Exception as e:
        print(f"Error processing multiple model results: {e}")
        return None, {'error': f'Multi-model processing failed: {str(e)}'}

def generate_response_with_image(image_path, prompt):
    """Generate AI response using Gemini"""
    if not gemini_model:
        return "AI analysis not available - API key not configured"
    
    if not os.path.exists(image_path):
        return "Error: Image file not found"
    
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
    except Exception as e:
        return f"Error reading the image: {e}"

    contents = [
        {"mime_type": "image/jpeg", "data": image_data},
        prompt
    ]

    try:
        response = gemini_model.generate_content(contents=contents)
        response.resolve()
        
        # Clean the response by removing unwanted characters
        cleaned_response = response.text.replace('*', '').strip()
        return cleaned_response
        
    except Exception as e:
        print(f"Error generating AI response: {e}")
        return f"AI analysis unavailable: {str(e)}"

def create_pdf_report(pdf_filepath, image_filename, processed_image_path, detections, gemini_response):
    """Create PDF report"""
    try:
        c = canvas.Canvas(pdf_filepath, pagesize=letter)
        width, height = letter
        
        # Title of the report
        c.setFont("Helvetica-Bold", 16)
        c.drawString(100, height - 40, "Medical Image Detection Report")
        
        # Original image and processed image info
        c.setFont("Helvetica", 12)
        c.drawString(100, height - 80, f"Original Image: {image_filename}")
        c.drawString(100, height - 100, f"Processed Image: {os.path.basename(processed_image_path) if processed_image_path else 'N/A'}")
        
        # Detections info
        c.drawString(100, height - 140, "Detected Objects:")
        y_position = height - 160
        
        # Handle both single model and multi-model detection results
        if isinstance(detections, dict):
            for model_name, model_detections in detections.items():
                if model_name == 'error':
                    c.drawString(100, y_position, f"Error: {model_detections}")
                    y_position -= 20
                    continue
                    
                c.drawString(100, y_position, f"Model: {model_name.capitalize()}")
                y_position -= 20
                
                if isinstance(model_detections, list):
                    for detection in model_detections:
                        if 'error' in detection:
                            detection_text = f"Error: {detection['error']}"
                        else:
                            detection_text = f"{detection['class'].capitalize()} (Confidence: {detection['confidence']*100:.2f}%)"
                        c.drawString(120, y_position, detection_text)
                        y_position -= 20
                y_position -= 10
        else:
            if isinstance(detections, list):
                for detection in detections:
                    if 'error' in detection:
                        detection_text = f"Error: {detection['error']}"
                    else:
                        detection_text = f"{detection['class'].capitalize()} (Confidence: {detection['confidence']*100:.2f}%)"
                    c.drawString(100, y_position, detection_text)
                    y_position -= 20
        
        # Gemini response
        c.drawString(100, y_position - 40, "AI Medical Insight:")
        c.setFont("Helvetica", 10)
        text_object = c.beginText(100, y_position - 60)
        text_object.setFont("Helvetica", 10)
        text_object.setTextOrigin(100, y_position - 60)
        
        # Handle long text by wrapping it
        max_width = 400
        words = gemini_response.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            line_text = ' '.join(current_line)
            if len(line_text) > 60:  # Approximate character limit per line
                if len(current_line) > 1:
                    current_line.pop()
                    lines.append(' '.join(current_line))
                    current_line = [word]
                else:
                    lines.append(line_text)
                    current_line = []
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            text_object.textLine(line)
        
        c.drawText(text_object)
        c.save()
        
    except Exception as e:
        print(f"Error creating PDF report: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Get optional model selection from form
    selected_model = request.form.get('model', None)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process the image
            processed_path, detections = process_image(filepath, selected_model)
            
            # Generate description using Gemini model
            prompt_text = "Describe the medical findings in the image with a professional perspective, including possible next steps in diagnosis and treatment, such as examinations, tests, and medications. If no significant findings are present, please confirm that. Provide insights as a doctor with 50 years of experience, considering both traditional and modern approaches, in no more than 10 lines."
            
            if processed_path:
                response_text = generate_response_with_image(processed_path, prompt_text)
            else:
                response_text = "Unable to generate AI analysis due to processing errors."
            
            # Generate PDF report
            pdf_filename = f"report_{filename.rsplit('.', 1)[0]}.pdf"
            pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            create_pdf_report(pdf_filepath, filename, processed_path, detections, response_text)
            
            # Prepare response
            response_data = {
                'message': 'Detection completed',
                'detections': detections,
                'gemini_response': response_text,
            }
            
            # Only include image paths if processing was successful
            if processed_path and os.path.exists(processed_path):
                # For Render, we need to serve files differently
                # Copy files to static folder if it exists, otherwise provide download endpoints
                try:
                    static_folder = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
                    if os.path.exists(static_folder):
                        shutil.copy2(filepath, static_folder)
                        shutil.copy2(processed_path, static_folder)
                        if os.path.exists(pdf_filepath):
                            shutil.copy2(pdf_filepath, static_folder)
                        
                        response_data.update({
                            'original_image': f'/static/uploads/{filename}',
                            'processed_image': f'/static/uploads/{os.path.basename(processed_path)}',
                            'pdf_report': f'/static/uploads/{pdf_filename}'
                        })
                    else:
                        # Serve from temporary directory
                        response_data.update({
                            'original_image': f'/temp/{filename}',
                            'processed_image': f'/temp/{os.path.basename(processed_path)}',
                            'pdf_report': f'/temp/{pdf_filename}'
                        })
                except Exception as e:
                    print(f"Error copying files: {e}")
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"Detection error: {e}")
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/temp/<filename>')
def serve_temp_file(filename):
    """Serve files from temporary directory"""
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({'error': 'File not found'}), 404

# Health check endpoint for Render
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})

if __name__ == '__main__':
    # Use environment variables for port (Render requirement)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
