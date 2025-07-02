from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load models
MODELS = {
    'liver': {
        'path': 'liver_tumour.pt',
        'classes': ["liver", "tumor"]
    },
    'brain_mri': {
        'path': 'brain_mri.pt',
        'classes': ["tumour", "eye"]
    },
    'eye': {
        'path': 'eye.pt',
        'classes': ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
    },
    'kidney': {
        'path': 'kidney.pt',
        'classes': ["stone"]
    },
    'lung': {
        'path': 'lung cancer.pt',  # Consider making this relative too
        'classes': ["Tumour"]
    }
}

# Load all models
models = {name: YOLO(config['path']) for name, config in MODELS.items()}

# Configure Google Gemini API with key from environment
API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please check your .env file.")

genai.configure(api_key=API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, model_name=None):
    # If no specific model is chosen, try all models
    if model_name is None:
        results_all = {}
        for name, model in models.items():
            results = model.predict(source=image_path, conf=0.25)
            results_all[name] = {
                'results': results[0],
                'classes': MODELS[name]['classes']
            }
        
        # Combine results
        return process_multiple_model_results(image_path, results_all)
    
    # If a specific model is chosen
    model = models[model_name]
    class_names = MODELS[model_name]['classes']
    
    # Make prediction
    results = model.predict(source=image_path, conf=0.25)
    result = results[0]
    
    # Load and process image
    img = cv2.imread(image_path)
    
    # Store detections info
    detections = []
    
    # Draw boxes and apply pseudo-coloring
    for detection in result.boxes:
        x_min, y_min, x_max, y_max = map(int, detection.xyxy[0])
        confidence = float(detection.conf[0])
        class_id = int(detection.cls[0])  # Extract class index
        
        # Get class name
        class_label = class_names[class_id] if class_id < len(class_names) else "Unknown"

        # Store detection info
        detections.append({
            'class': class_label,
            'confidence': confidence,
            'bbox': [x_min, y_min, x_max, y_max]
        })
        
        # Define color (you might want to customize this per model/class)
        color = (0, 255, 0)  # Default green
        if model_name == 'liver':
            color = (0, 255, 0) if class_label == "liver" else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

        # Add text label
        text = f"{class_label.capitalize()} ({confidence:.2f})"
        cv2.putText(img, text, (x_min, y_min - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Pseudo-coloring for specific cases (customize as needed)
        if class_label in ["tumor", "tumour", "cancer", "nodule", "stone"]:
            detected_area = img[y_min:y_max, x_min:x_max]
            gray_area = cv2.cvtColor(detected_area, cv2.COLOR_BGR2GRAY)
            pseudo_colored_area = cv2.applyColorMap(gray_area, cv2.COLORMAP_JET)
            img[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                detected_area, 0.5, pseudo_colored_area, 0.5, 0
            )

    # Save processed image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                              'processed_' + os.path.basename(image_path))
    cv2.imwrite(output_path, img)
    
    return output_path, detections

def process_multiple_model_results(image_path, results_all):
    # Load original image
    img = cv2.imread(image_path)
    
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
            
            # Define color (you might want to customize this)
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

def generate_response_with_image(image_path, prompt):
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

        # Clean the response by removing unwanted characters or asterisks
        cleaned_response = response.text.replace('*', '').strip()

        return cleaned_response  # Return cleaned response directly
    except Exception as e:
        return f"Error Generating Response: {e}"

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
            response_text = generate_response_with_image(processed_path, prompt_text)
            
            # Generate PDF report
            pdf_filename = f"report_{filename.rsplit('.', 1)[0]}.pdf"
            pdf_filepath = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
            create_pdf_report(pdf_filepath, filename, processed_path, detections, response_text)
            
            # Return the response with a link to the PDF report
            return jsonify({
                'message': 'Detection completed',
                'original_image': f'/static/uploads/{filename}',
                'processed_image': f'/static/uploads/{os.path.basename(processed_path)}',
                'detections': detections,
                'gemini_response': response_text,
                'pdf_report': f'/static/uploads/{pdf_filename}'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

# Function to create a PDF report
def create_pdf_report(pdf_filepath, image_filename, processed_image_path, detections, gemini_response):
    c = canvas.Canvas(pdf_filepath, pagesize=letter)
    width, height = letter
    
    # Title of the report
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 40, "Medical Image Detection Report")
    
    # Original image and processed image info
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Original Image: {image_filename}")
    c.drawString(100, height - 100, f"Processed Image: {os.path.basename(processed_image_path)}")
    
    # Detections info
    c.drawString(100, height - 140, "Detected Objects:")
    y_position = height - 160
    
    # Handle both single model and multi-model detection results
    if isinstance(detections, dict):
        for model_name, model_detections in detections.items():
            c.drawString(100, y_position, f"Model: {model_name.capitalize()}")
            y_position -= 20
            for detection in model_detections:
                detection_text = f"{detection['class'].capitalize()} (Confidence: {detection['confidence']*100:.2f}%)"
                c.drawString(120, y_position, detection_text)
                y_position -= 20
            y_position -= 10
    else:
        for detection in detections:
            detection_text = f"{detection['class'].capitalize()} (Confidence: {detection['confidence']*100:.2f}%)"
            c.drawString(100, y_position, detection_text)
            y_position -= 20
    
    # Gemini response
    c.drawString(100, y_position - 40, "AI Medical Insight:")
    c.setFont("Helvetica", 10)
    text_object = c.beginText(100, y_position - 60)
    text_object.setFont("Helvetica", 10)
    text_object.setTextOrigin(100, y_position - 60)
    text_object.textLines(gemini_response)
    c.drawText(text_object)
    
    # Save the PDF
    c.save()

if __name__ == '__main__':
    app.run(debug=True)
