from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import uuid
import os
import json

app = Flask(__name__)

 
 
 
# Frontend (5500) ile Backend (5000)   
CORS(app)

 
# Hata almamak için dosyanın varlığını kontrol et
try:
    with open("BirdInfo.json", "r", encoding="utf-8") as f:
        BIRD_INFO = json.load(f)
except FileNotFoundError:
    print("UYARI: BirdInfo.json bulunamadı! Boş sözlük kullanılıyor.")
    BIRD_INFO = {}

# Normalize JSON keys (lowercase & strip)
BIRD_INFO_NORMALIZED = {
    key.strip().lower().replace(" ", "_"): value
    for key, value in BIRD_INFO.items()
}
#yolo modeli
MODEL_PATH = "best.pt"
# Modeli global yükle
model = YOLO(MODEL_PATH)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
RESULT_DIR = "outputs/result"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
 
# FastAPI'deki 'app.mount' yerine Flask'ta bu şekilde statik dosya sunulur

# Frontend şu adrese istek atacak: http://localhost:5000/outputs/result/resim.jpg

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory(OUTPUT_DIR, filename)


#  HOME ENDPOINT
 
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "BirdBase Flask API is running"})


#  PREDICT ENDPOINT
 
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image part"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 1) Random file name
    random_name = f"{uuid.uuid4()}.jpg"
    upload_path = os.path.join(UPLOAD_DIR, random_name)

    # 2) Save image as JPG
    pil_image = Image.open(file).convert("RGB")
    pil_image.save(upload_path, "JPEG")

    # 3) YOLO prediction
    # save=True dediğimizde ultralytics otomatik olarak 'outputs/result' altına kaydeder.
    results = model.predict(
        source=upload_path,
        conf=0.25,
        save=True,
        project="outputs",
        name="result",
        exist_ok=True,
        verbose=False
    )

    # 4) Parse detections
    detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            confidence = float(box.conf[0]) * 100

            detections.append({
                "class": cls_name,
                "confidence": round(confidence, 2)
            })

    # 5) Select best detection & match bird info
    bird_info = None
    if detections:
        best_detection = max(detections, key=lambda x: x["confidence"])

        detected_class = (
            best_detection["class"]
            .strip()
            .lower()
            .replace(" ", "_")
        )

        if detected_class in BIRD_INFO_NORMALIZED:
            bird_info = BIRD_INFO_NORMALIZED[detected_class]

    # 6) Output image path (Frontend için URL)
    
    output_image = f"/outputs/result/{random_name}"

    return jsonify({
        "output_image": output_image,
        "detections": detections,
        "bird_info": bird_info
    })

if __name__ == '__main__':
    # Flask varsayılan olarak 5000 portunda çalışıyo
    app.run(debug=True, port=5000)


    # source venv/bin/activate