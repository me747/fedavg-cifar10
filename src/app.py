from flask import Flask, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
from model import ResNet18_Fed

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load global model
model = ResNet18_Fed()
model.load_state_dict(torch.load("../results/global_model.pt", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor()
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(file).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = torch.argmax(logits, dim=1).item()

    return jsonify({"prediction": pred})

if __name__ == "__main__":
    # Run on 0.0.0.0 to allow external access
    app.run(host="0.0.0.0", port=8000)
