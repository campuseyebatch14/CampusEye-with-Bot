from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
import cv2
import numpy as np
from dotenv import load_dotenv
import cloudinary
from cloudinary.uploader import upload
from cloudinary.exceptions import Error as CloudinaryError
import os
import requests
from io import StringIO, BytesIO
import csv
import pandas as pd  # Required for Chatbot

import mongo_utils
import model_utils

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
env_path = os.path.join(os.path.dirname(__file__), '.env')
if not os.path.exists(env_path):
    raise RuntimeError(".env file not found")

load_dotenv(env_path)
print(".env loaded successfully")

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret")

# Cloudinary Config
cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("API_KEY"),
    api_secret=os.getenv("API_SECRET")
)

# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route('/')
def index():
    students_list = list(
        mongo_utils.students_collection.find({}, {'_id': 0, 'embedding': 0})
    )
    return render_template('index.html', students_list=students_list)


@app.route('/add-student', methods=['GET', 'POST'])
def add_student():
    if request.method == 'GET':
        return render_template('student_form.html', student=None)

    name = request.form['name']
    student_id = request.form['student_id']
    branch = request.form['branch']
    photo = request.files['photo']

    if photo.filename == '':
        flash('Empty photo file', 'error')
        return redirect(url_for('add_student'))

    try:
        upload_result = upload(photo)
        photo_url = upload_result['secure_url']

        photo.seek(0)
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        embedding = model_utils.getEmbedding(img)

        if embedding is None:
            flash('Clear face not detected. Upload a better photo.', 'error')
            return redirect(url_for('add_student'))

        if mongo_utils.students_collection.find_one({'studentId': student_id}):
            flash('Student ID already exists.', 'error')
            return redirect(url_for('add_student'))

        mongo_utils.students_collection.insert_one({
            'name': name,
            'studentId': student_id,
            'branch': branch,
            'embedding': embedding,
            'photoUrl': photo_url
        })

        flash('Student added successfully', 'success')
        return redirect(url_for('index'))

    except CloudinaryError as e:
        flash(f'Cloudinary error: {e}', 'error')
        return redirect(url_for('add_student'))


@app.route('/edit-student/<student_id>', methods=['GET', 'POST'])
def edit_student(student_id):
    student = mongo_utils.getStudentDetails(student_id)

    if request.method == 'GET':
        return render_template('student_form.html', student=student)

    name = request.form['name']
    branch = request.form['branch']
    photo = request.files['photo']

    try:
        upload_result = upload(photo)
        photo_url = upload_result['secure_url']

        photo.seek(0)
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        embedding = model_utils.getEmbedding(img)

        if embedding is None:
            flash('Clear face not detected.', 'error')
            return redirect(url_for('edit_student', student_id=student_id))

        mongo_utils.students_collection.update_one(
            {'studentId': student_id},
            {'$set': {
                'name': name,
                'branch': branch,
                'embedding': embedding,
                'photoUrl': photo_url
            }}
        )

        flash('Student updated successfully', 'success')
        return redirect(url_for('index'))

    except CloudinaryError as e:
        flash(f'Cloudinary error: {e}', 'error')
        return redirect(url_for('edit_student', student_id=student_id))


@app.route('/delete-student/<student_id>')
def delete_student(student_id):
    mongo_utils.deleteStudent(student_id)
    flash('Student removed successfully', 'success')
    return redirect(url_for('index'))


# --------------------------------------------------
# ✅ IMPROVED AI CHATBOT ROUTE
# --------------------------------------------------
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_query = data.get('query', '').lower()
    
    try:
        # Fetch Data from MongoDB
        detections = list(mongo_utils.detections_collection.find({}, {'_id': 0}))
        
        if not detections:
            return jsonify({'success': True, 'answer': "I don't see any detection records in the system yet."})
            
        df = pd.DataFrame(detections)
        df['timestamp'] = df['timestamp'].astype(str)
        
        # --- SMART LOGIC: CHECK FOR NAMES FIRST ---
        
        # 1. Check if any student name is in the query
        # This handles: "Govind detected timings", "Logs for Jitho", "Show me Govind"
        all_names = df['name'].unique()
        found_name = None
        
        for name in all_names:
            # Check if the name appears in the user's question
            if name.lower() in user_query:
                found_name = name
                break
        
        if found_name:
            # If name found, get their specific logs
            student_logs = df[df['name'] == found_name]
            times = student_logs['timestamp'].tolist()
            count = len(times)
            
            # Formatting the response nicely
            if count > 0:
                # Show only last 5 if there are too many
                if count > 5:
                    recent_times = times[-5:]
                    time_list_html = "<br>• " + "<br>• ".join(recent_times)
                    response_text = f"Found {count} records for <b>{found_name}</b>.<br>Most recent detections:{time_list_html}"
                else:
                    time_list_html = "<br>• " + "<br>• ".join(times)
                    response_text = f"<b>{found_name}</b> was detected at:{time_list_html}"
            else:
                response_text = f"I found <b>{found_name}</b> in the database, but there are no detection logs yet."
                
            return jsonify({'success': True, 'answer': response_text})

        # 2. General Questions ("Who", "List", "Show")
        elif any(k in user_query for k in ['who', 'list', 'show', 'all']):
            if 'cse' in user_query:
                result = df[df['branch'].str.upper() == 'CSE']
                branch_name = "CSE"
            elif 'cs' in user_query:
                 result = df[df['branch'].str.upper() == 'CS']
                 branch_name = "CS"
            else:
                result = df
                branch_name = "all"
                
            names = result['name'].unique()
            names_str = ", ".join(names)
            return jsonify({'success': True, 'answer': f"Students detected ({branch_name}): {names_str}."})

        # 3. Count Questions
        elif any(k in user_query for k in ['count', 'how many', 'total']):
            count = df['studentId'].nunique()
            return jsonify({'success': True, 'answer': f"There are {count} unique students detected in the logs."})

        # Fallback
        else:
            return jsonify({'success': True, 'answer': "I didn't understand that. <br>Try typing just a name like <b>'Govind'</b>, or ask <b>'Who was seen?'</b>."})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/send-email', methods=['POST'])
def send_email():
    live_image = request.files.get('live_image')
    
    if live_image:
        data = request.form
    else:
        data = request.get_json() or {}

    if not data:
        return jsonify({'success': False, 'error': 'No data provided'})

    final_photo_url = data.get("photoUrl")

    if live_image:
        try:
            print("Uploading live capture to Cloudinary...")
            upload_result = upload(live_image)
            final_photo_url = upload_result['secure_url']
            print(f"Live image uploaded: {final_photo_url}")
        except Exception as e:
            print(f"Cloudinary upload failed: {e}")

    payload = {
        "service_id": os.getenv("EMAILJS_SERVICE_ID"),
        "template_id": os.getenv("EMAILJS_TEMPLATE_ID"),
        "user_id": os.getenv("EMAILJS_USER_ID"),         
        "accessToken": os.getenv("EMAILJS_PRIVATE_KEY"),  
        "template_params": {
            "to_name": data.get("name"),
            "student_id": data.get("studentId"),
            "branch": data.get("branch"),
            "timestamp": data.get("timestamp"),
            "photo_url": final_photo_url,  
            "to_email": os.getenv("RECIPIENT_EMAIL")
        }
    }

    try:
        response = requests.post(
            "https://api.emailjs.com/api/v1.0/email/send",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=10 
        )
        return jsonify({"success": response.status_code == 200})
    except requests.RequestException as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/download-report')
def download_report():
    detections = list(mongo_utils.detections_collection.find({}, {'_id': 0}))
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Name', 'Student ID', 'Branch', 'Timestamp', 'Photo URL'])

    for d in detections:
        writer.writerow([
            d['name'], d['studentId'], d['branch'], d['timestamp'], d['photoUrl']
        ])

    output.seek(0)
    return send_file(
        BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='detection_report.csv'
    )


if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(debug=True)