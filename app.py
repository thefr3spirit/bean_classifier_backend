from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
from functools import lru_cache
from PIL import Image
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from transformers import ViTForImageClassification, ViTConfig

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
MODEL_DIR   = Path("models")
CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

app = Flask(__name__)
CORS(app)  # Enable CORS for API access from frontend
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit
app.config["UPLOAD_FOLDER"] = Path("tmp_uploads")
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Lazy‑loaded model helpers
# ----------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_vit():
    config = ViTConfig(num_labels=3)
    model = ViTForImageClassification(config)
    sd = torch.load(MODEL_DIR / "vit_model.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()

@lru_cache(maxsize=1)
def load_resnet():
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 3)
    sd = torch.load(MODEL_DIR / "resnet_model.pth", map_location=DEVICE)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()

@lru_cache(maxsize=1)
def load_mobilenet():
    model = models.mobilenet_v2()
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    sd = torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE)
    model.load_state_dict(sd)
    return model.to(DEVICE).eval()

MODEL_REGISTRY = {
    "vit": load_vit,
    "resnet18": load_resnet,
    "mobilenetv2": load_mobilenet
}

# ----------------------------------------------------------------------
# HTML UI Route (For local/manual testing)
# ----------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model_key = request.form.get("model")
        files = request.files.getlist("image")
        if not files or model_key not in MODEL_REGISTRY:
            return redirect(url_for("index"))

        results = []
        upload_folder = Path("static/uploads")
        upload_folder.mkdir(exist_ok=True)

        for file in files:
            filename = secure_filename(file.filename)
            save_path = upload_folder / filename
            file.save(save_path)

            img = Image.open(save_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                model = MODEL_REGISTRY[model_key]()
                output = model(tensor)
                logits = getattr(output, "logits", output)
                probs = torch.softmax(logits, dim=1)[0]

            pred_idx = probs.argmax().item()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = probs[pred_idx].item() * 100

            results.append({
                "filename": filename,
                "pred_class": pred_class,
                "confidence": f"{confidence:.2f}",
                "image_url": url_for('static', filename=f"uploads/{filename}")
            })

        return render_template(
            "index.html",
            results=results,
            choice=model_key
        )

    return render_template("index.html", results=None, choice=None)


# ----------------------------------------------------------------------
# JSON API Route for React frontend
# ----------------------------------------------------------------------
@app.route("/api/classify", methods=["POST"])
def api_classify():
    model_key = request.form.get("model")
    files = request.files.getlist("image")

    if not files or model_key not in MODEL_REGISTRY:
        return jsonify({"error": "Missing image(s) or invalid model"}), 400

    results = []
    upload_folder = Path("static/uploads")
    upload_folder.mkdir(exist_ok=True)

    for file in files:
        if file.filename == "":
            continue

        filename = secure_filename(file.filename)
        save_path = upload_folder / filename
        file.save(save_path)

        try:
            img = Image.open(save_path).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Could not open image {filename}: {str(e)}"}), 500

        tensor = transform(img).unsqueeze(0).to(DEVICE)

        try:
            with torch.no_grad():
                model = MODEL_REGISTRY[model_key]()
                output = model(tensor)
                logits = getattr(output, "logits", output)
                probs = torch.softmax(logits, dim=1)[0]

            pred_idx = probs.argmax().item()
            pred_class = CLASS_NAMES[pred_idx]
            confidence = round(probs[pred_idx].item() * 100, 2)

            results.append({
                "filename": filename,
                "pred_class": pred_class,
                "confidence": confidence,
                "image_url": url_for('static', filename=f"uploads/{filename}", _external=True)
            })
        except Exception as e:
            return jsonify({"error": f"Prediction failed on {filename}: {str(e)}"}), 500

    return jsonify({"results": results})


# ----------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
