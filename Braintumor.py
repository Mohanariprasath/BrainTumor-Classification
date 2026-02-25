import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

CONFIG = {
    "UPLOAD_FOLDER": "static/uploads",
    "RESULT_FOLDER": "static/results",
    "MODEL_PATH": "best_brain_tumor_model.pth",
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "IMG_SIZE": 224,
    "CLASS_NAMES": ["meningioma", "glioma", "pituitary", "no_tumor"]
}

os.makedirs(CONFIG["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(CONFIG["RESULT_FOLDER"], exist_ok=True)

preprocess = transforms.Compose([
    transforms.Resize((CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CONFIG["CLASS_NAMES"]))
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=CONFIG["DEVICE"])
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.to(CONFIG["DEVICE"]).eval()
    return model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()
        grads = self.gradients[0]
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        if cam.max() != 0:
            cam = cam / cam.max()
        return cam.cpu().numpy(), class_idx

def generate_heatmap(img_path, cam, out_path):
    img = Image.open(img_path).convert("RGB")
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize(img.size, resample=Image.BILINEAR)
    cam_arr = np.array(cam_img) / 255.0
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.imshow(cam_arr, cmap='jet', alpha=0.4)
    plt.axis('off')
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    global model, gradcam
    if 'model' not in globals():
        model = load_model(CONFIG["MODEL_PATH"])
        gradcam = GradCAM(model, model.layer4[-1].conv1)

    if request.method == "POST":
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == "": return redirect(request.url)
        filename = secure_filename(file.filename)
        save_path = os.path.join(CONFIG["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        img = Image.open(save_path).convert("RGB")
        img_tensor = preprocess(img).unsqueeze(0).to(CONFIG["DEVICE"])
        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = output.argmax(dim=1).item()
            confidence = torch.nn.functional.softmax(output, dim=1)[0, pred_idx].item()
        heatmap, _ = gradcam(img_tensor, pred_idx)
        result_name = f"result_{filename}"
        result_path = os.path.join(CONFIG["RESULT_FOLDER"], result_name)
        generate_heatmap(save_path, heatmap, result_path)
        return f"{CONFIG['CLASS_NAMES'][pred_idx]} ({confidence*100:.2f}%)"
    return "OK"

if __name__ == "__main__":
    app.run(debug=True)
