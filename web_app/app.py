# app.py (PyTorch + Grad-CAM)
import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from matplotlib import pyplot as plt

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
# MODEL_PATH can be set via environment variable or use default
MODEL_PATH = os.getenv("MODEL_PATH", "best_brain_tumor_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
CLASS_NAMES = ["meningioma", "glioma", "pituitary"]

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load model
def load_model(path, num_classes=len(CLASS_NAMES)):
    ck = torch.load(path, map_location=DEVICE)
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if isinstance(ck, dict) and 'model_state_dict' in ck:
        model.load_state_dict(ck['model_state_dict'])
    elif isinstance(ck, dict) and 'state_dict' in ck:
        model.load_state_dict(ck['state_dict'])
    else:
        model.load_state_dict(ck)
    model.to(DEVICE).eval()
    return model

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        outputs = self.model(input_tensor)
        if class_idx is None:
            class_idx = outputs.argmax(dim=1).item()
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx

# Global variables for model
MODEL = None
TARGET_LAYER = None
GRADCAM_OBJ = None
MODEL_ERROR = None

if os.path.exists(MODEL_PATH):
    try:
        MODEL = load_model(MODEL_PATH)
        TARGET_LAYER = MODEL.layer4[-1].conv2
        GRADCAM_OBJ = GradCAM(MODEL, TARGET_LAYER)
        print(f"✓ Model loaded successfully from: {MODEL_PATH}")
    except Exception as e:
        MODEL_ERROR = f"Error loading model: {str(e)}"
        print(f"✗ {MODEL_ERROR}")
else:
    MODEL_ERROR = f"Model file not found at: {MODEL_PATH}"
    print(f"✗ {MODEL_ERROR}")

def make_gradcam_overlay(img_path, cam, out_path, alpha=0.4):
    img = Image.open(img_path).convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, resample=Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.imshow(cam_arr, cmap='jet', alpha=alpha)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

@app.route("/", methods=["GET", "POST"])
def index():
    if MODEL is None:
        error_msg = MODEL_ERROR if MODEL_ERROR else "Model not loaded."
        return render_template("index.html", uploaded_img=None, gradcam_img=None, 
                               prediction=None, error=error_msg)
    
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == "":
            return redirect(request.url)
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)

        # preprocess and classify
        img = Image.open(save_path).convert("RGB")
        inp = preprocess(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = MODEL(inp)
            pred = out.argmax(dim=1).item()

        # Generate Grad-CAM (requires gradients)
        cam, idx = GRADCAM_OBJ(inp, None)
        result_fname = f"gradcam_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_fname)
        make_gradcam_overlay(save_path, cam, result_path)
        
        return render_template("index.html", 
                               uploaded_img=url_for('static', filename=f"uploads/{filename}"),
                               gradcam_img=url_for('static', filename=f"results/{result_fname}"),
                               prediction=CLASS_NAMES[pred])
    
    return render_template("index.html", uploaded_img=None, gradcam_img=None, prediction=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
