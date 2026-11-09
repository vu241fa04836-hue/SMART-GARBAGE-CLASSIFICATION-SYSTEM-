"""
Smart Garbage Classification System - Flask Web Application
Main controller for web routes, API endpoints, file uploads, and analytics.
"""

import os
import csv
import io
import base64
import uuid
from datetime import datetime
from flask import (
    Flask, render_template, request, jsonify, 
    send_from_directory, redirect, url_for, flash, Response
)
from werkzeug.utils import secure_filename

import database
from model import (
    classify_garbage_image, 
    WASTE_METADATA, 
    ORDERED_CLASSES,
    load_ml_model
)

# Initialize Flask application
app = Flask(__name__)
app.secret_key = "smart-garbage-classifier-secret-key-2026"

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SAMPLE_FOLDER = os.path.join(BASE_DIR, 'sample_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'jfif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16 MB max upload

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAMPLE_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

# Initialize Database
database.init_db()

def allowed_file(filename):
    """Check if uploaded file has an allowed image extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.context_processor
def inject_global_data():
    """Make categories and current year available to all templates."""
    _, active_engine, _ = load_ml_model()
    return {
        'waste_categories': WASTE_METADATA,
        'ordered_classes': ORDERED_CLASSES,
        'active_engine': active_engine,
        'current_year': datetime.now().year
    }

# ==================== MAIN PAGES ====================

@app.route('/')
def home():
    """Home / Landing page."""
    stats = database.get_dashboard_stats()
    return render_template('index.html', stats=stats)

@app.route('/classify')
def classify_page():
    """Garbage Classification & Live Camera Page."""
    # List available sample images for one-click testing
    sample_files = []
    if os.path.exists(SAMPLE_FOLDER):
        for f in os.listdir(SAMPLE_FOLDER):
            if allowed_file(f):
                sample_files.append(f)
    return render_template('classify.html', sample_files=sample_files)

@app.route('/dashboard')
def dashboard():
    """Analytics and Statistics Dashboard."""
    stats = database.get_dashboard_stats()
    return render_template('dashboard.html', stats=stats)

@app.route('/history')
def history():
    """Classification History Table with Search and Filtering."""
    category_filter = request.args.get('category', 'All')
    search_query = request.args.get('q', '').strip()
    records = database.get_all_classifications(category_filter=category_filter, search_term=search_query)
    return render_template(
        'history.html', 
        records=records, 
        current_filter=category_filter, 
        search_query=search_query
    )

@app.route('/about')
def about():
    """About the Project, Objectives, Technology Stack, and Impact."""
    return render_template('about.html')

# ==================== STATIC FILE SERVING ====================

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded user images."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/sample_images/<filename>')
def sample_file(filename):
    """Serve sample demo images."""
    return send_from_directory(SAMPLE_FOLDER, filename)

# ==================== API / AJAX ENDPOINTS ====================

@app.route('/api/classify', methods=['POST'])
def api_classify():
    """
    Classify an uploaded image or camera snapshot.
    Accepts:
      - Multipart file upload ('image')
      - OR JSON payload with base64 data URL ('camera_image')
      - OR sample image filename ('sample_filename')
    """
    try:
        saved_filename = None
        file_path = None
        
        # 1. Check for standard multipart file upload
        if 'image' in request.files and request.files['image'].filename != '':
            file = request.files['image']
            if not allowed_file(file.filename):
                return jsonify({
                    'success': False,
                    'error': 'Unsupported file format. Please upload JPG, PNG, WEBP, or BMP.'
                }), 400
                
            original_ext = file.filename.rsplit('.', 1)[1].lower()
            unique_name = f"waste_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.{original_ext}"
            saved_filename = secure_filename(unique_name)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(file_path)

        # 2. Check for base64 camera snapshot
        elif request.is_json and 'camera_image' in request.json:
            data_url = request.json['camera_image']
            if ',' in data_url:
                header, base64_str = data_url.split(',', 1)
            else:
                base64_str = data_url
                
            image_data = base64.b64decode(base64_str)
            unique_name = f"cam_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
            saved_filename = secure_filename(unique_name)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            with open(file_path, 'wb') as f:
                f.write(image_data)

        # 3. Check for sample image quick test
        elif request.is_json and 'sample_filename' in request.json:
            sample_name = request.json['sample_filename']
            sample_source = os.path.join(SAMPLE_FOLDER, sample_name)
            if not os.path.exists(sample_source):
                return jsonify({'success': False, 'error': 'Sample file not found.'}), 404
                
            unique_name = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}_{sample_name}"
            saved_filename = secure_filename(unique_name)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            with open(sample_source, 'rb') as src, open(file_path, 'wb') as dst:
                dst.write(src.read())

        else:
            return jsonify({
                'success': False,
                'error': 'No image provided. Please select a file or take a camera snapshot.'
            }), 400

        # Run AI/ML Classification
        result = classify_garbage_image(file_path)
        
        # Save record into SQLite database
        record_id = database.add_classification(
            filename=saved_filename,
            category=result['category'],
            confidence=result['confidence'],
            disposal_recommendation=result['recommendation'],
            is_recyclable=result['is_recyclable'],
            is_biodegradable=result['is_biodegradable'],
            all_probabilities=result['probabilities']
        )
        
        result['id'] = record_id
        result['filename'] = saved_filename
        result['image_url'] = url_for('uploaded_file', filename=saved_filename)
        result['success'] = True
        
        return jsonify(result)
        
    except Exception as e:
        app.logger.error(f"Classification error: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f"Failed to process image: {str(e)}"
        }), 500

@app.route('/api/dashboard-data')
def api_dashboard_data():
    """Return JSON statistics for asynchronous Chart.js rendering."""
    stats = database.get_dashboard_stats()
    return jsonify(stats)

@app.route('/api/history/delete/<int:record_id>', methods=['POST', 'DELETE'])
def api_delete_history(record_id):
    """Delete a single classification record and its corresponding uploaded image."""
    try:
        filename = database.delete_classification(record_id)
        if filename:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        return jsonify({'success': True, 'message': 'Record deleted successfully.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    """Clear all classification history and associated uploaded image files."""
    try:
        filenames = database.clear_all_history()
        for fn in filenames:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        return jsonify({'success': True, 'message': 'All classification history cleared.'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/export-csv')
def export_csv():
    """Download classification history as a CSV spreadsheet report."""
    records = database.get_all_classifications(limit=1000)
    
    si = io.StringIO()
    writer = csv.writer(si)
    
    # CSV Header
    writer.writerow([
        'Classification ID', 
        'Timestamp', 
        'Image Filename', 
        'Predicted Category', 
        'Confidence (%)', 
        'Is Recyclable', 
        'Is Biodegradable', 
        'Disposal Recommendation'
    ])
    
    for r in records:
        writer.writerow([
            r['id'],
            r['created_at'],
            r['filename'],
            r['category'],
            f"{r['confidence']:.1f}%",
            'Yes' if r['is_recyclable'] else 'No',
            'Yes' if r['is_biodegradable'] else 'No',
            r['disposal_recommendation']
        ])
        
    output = si.getvalue()
    filename = f"garbage_classification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={filename}"}
    )

# ==================== ERROR HANDLERS ====================

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'success': False,
        'error': 'Image file size is too large. Maximum allowed size is 16MB.'
    }), 413

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found. Redirected to home."), 404

@app.errorhandler(500)
def server_error(error):
    return render_template('index.html', error="Internal server error. Please try again."), 500

if __name__ == '__main__':
    print("\n" + "=" * 65)
    print(" SMART GARBAGE CLASSIFICATION SYSTEM - SERVER READY")
    print("=" * 65)
    print(" Accessible at: http://127.0.0.1:5000")
    print("=" * 65 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
