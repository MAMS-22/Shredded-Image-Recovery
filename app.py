from environment_and_utilities import *
from flask import Flask, request, send_file, render_template, redirect, url_for
import subprocess
from PIL import Image
from evaluate import evaluate

app = Flask(__name__)

# Define paths for input and output images
DEMO_PATH = './demo/'
INPUT_IMAGE_PATH = DEMO_PATH + 'input_image.jpg'         # Fixed input filename
OUTPUT_IMAGE_PATH = './static/'       # Fixed output filename
grid_size = 4

def shred():
    im = cv2.imread(INPUT_IMAGE_PATH)
    if im is None:
        raise ValueError("Failed to decode uploaded image. Please ensure you upload a valid image file.")
    shreds_path = os.path.join(DEMO_PATH, 'shreds') + '/'
    os.makedirs(shreds_path, exist_ok=True)
    height, width, _ = im.shape
    frac_h = height // grid_size
    frac_w = width // grid_size
    i = 0
    for h in range(grid_size):
        for w in range(grid_size):
            crop = im[h * frac_h:(h + 1) * frac_h, w * frac_w:(w + 1) * frac_w]
            if i < 10:
                cv2.imwrite(shreds_path + "img_0{}.jpg".format(i), crop)
            else:
                cv2.imwrite(shreds_path + "img_{}.jpg".format(i), crop)
            i = i + 1
    return shreds_path


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file', 400
    
    if file:
        if os.path.exists(DEMO_PATH):
            shutil.rmtree(DEMO_PATH)
        os.makedirs(DEMO_PATH, exist_ok=True)
        os.makedirs(OUTPUT_IMAGE_PATH, exist_ok=True)
        file.save(INPUT_IMAGE_PATH)
        try:
            path = shred()
            evaluate(path, OUTPUT_IMAGE_PATH, display=False)
        except Exception as e:
            return f"Error during processing: {e}", 500
        
        return redirect(url_for('display_images'))

@app.route('/display_images')
def display_images():
    original_image = url_for('static', filename='before.jpg')
    processed_image = url_for('static', filename='after.jpg')
    return render_template('index.html', original_image=original_image, processed_image=processed_image)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 15500))
    app.run(host='0.0.0.0', port=port, debug=True)