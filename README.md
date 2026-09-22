# Shredded Image Recovery (Color)

An automated computer vision and deep learning project that reconstructs shredded/jumbled color images into their original form. It splits images into grid-based shreds (e.g., $4 \times 4$), scores the likelihood of adjacency between every pair of shreds using a trained Deep Convolutional Neural Network (CNN), and reassembles the full image using a greedy optimization algorithm.

---

## Features

- **Deep CNN Pairwise Classifier**: Evaluates whether two image tiles are legitimately adjacent horizontally or vertically.
- **Greedy Reassembly Solver**: Assembles rows from left-to-right using horizontal adjacency probabilities, then stacks rows top-to-bottom using vertical edge probabilities.
- **Flask Web Interface**: Interactive web dashboard allowing users to upload an image, shred it into tiles, and view the reassembled result side-by-side.
- **Pretrained Weights Included**: Comes with pretrained model weights (`2024102912.weights.h5`) trained on COCO dataset crops.
- **Benchmark Test Suite**: Includes automated testing on 2×2, 3×3, 4×4, and 5×5 shred test sets.

---

## Project Structure

```text
├── app.py                        # Flask web application (upload -> shred -> reconstruct -> view)
├── evaluate.py                   # High-level evaluation functions
├── evaluate_aux.py               # Core puzzle-solving and tile-assembly logic
├── legit_img_model.py            # CNN architecture and weights loader
├── environment_and_utilities.py  # Image transformations, paths, and helpers
├── prepare_data.py               # Training data generator (creates positive & negative tile pairs)
├── test_images.py                # Benchmark script measuring reassembly accuracy
├── requirements.txt              # Python package dependencies
├── 2024102912.weights.h5         # Pretrained Keras 3 model weights
├── templates/
│   └── index.html                # Web app frontend template
├── static/
│   ├── before.jpg                # Last shredded/jumbled image preview
│   └── after.jpg                 # Last reassembled output image
├── project/
│   └── test_images/              # Test datasets for 2x2, 3x3, 4x4, and 5x5 grids
├── COCO_dataset/                 # Optional training dataset (COCO val2017)
└── demo001/, demo002/, demo003/  # Sample demo folders
```

---

## Prerequisites & Installation

### 1. System Requirements
- Python 3.10 to 3.12
- Linux / macOS / Windows

On Debian/Ubuntu systems, install the Python venv package if not already present:
```bash
sudo apt update
sudo apt install python3-venv python3-full
```

### 2. Set Up a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # On Linux/macOS
# or: venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on GPU Acceleration**: For NVIDIA GPU support with CUDA, ensure compatible CUDA drivers are installed, or install `tensorflow[and-cuda]`.

---

## How to Run

### 1. Web Application
Launch the Flask web server:
```bash
python3 app.py
```
By default, the server runs on `http://localhost:15500`.

- Open your browser to `http://localhost:15500`.
- Upload any image (JPEG or PNG).
- The server will shred the image into a $4 \times 4$ grid, run the neural network solver, and display the before and after images.

You can customize the port via the `PORT` environment variable:
```bash
PORT=8080 python3 app.py
```

### 2. Run Benchmark Accuracy Tests
To test the model's accuracy on the provided test dataset (2×2, 3×3, and 4×4 folders):
```bash
python3 test_images.py
```

### 3. Data Preparation & Retraining (Optional)
If you wish to train your own model using the COCO dataset:
1. Ensure the COCO dataset images are placed in `COCO_dataset/val2017/`.
2. Generate training pairs:
   ```bash
   python3 prepare_data.py
   ```
3. Run training via `legit_img_model.py`.
