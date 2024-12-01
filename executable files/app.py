from flask import Flask, render_template, request, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model('ECG_Model2.h5')
labels = ['Normal', 'Left bundle branch block ','Premature atrial contraction', 'Premature ventricular contraction', 'Right bundle branch block', 'Ventricular fibrillation']  # Example classes

def predict_ecg(file_path):
    img = load_img(file_path, target_size=(224, 224))  # Adjust to your model's input size
    img_array = img_to_array(img) / 255.0
    img_array = img_array.reshape((1, *img_array.shape))
    prediction = model.predict(img_array)
    return labels[prediction.argmax()]

@app.route('/')
def index():
    return render_template('index6.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        prediction = predict_ecg(file_path)
        return render_template('index6.html', uploaded_file=file.filename, prediction=prediction)
    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return url_for('static', filename=os.path.join('uploads', filename))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/info')
def info():
    return render_template('info.html')

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
