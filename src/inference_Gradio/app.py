import gradio as gr
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO
from PIL import Image

# Load models once
model_yolo = YOLO("model_yolov8.pt")

model_resnet = models.resnet18()
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, 2)
model_resnet.load_state_dict(torch.load("model_resnet18.pth", map_location="cpu"))
model_resnet.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_health(image):
    img = np.array(image)
    if img.shape[2] == 4:
        B, G, R, NIR = cv2.split(img)
        R = R.astype(np.float32)
        NIR = NIR.astype(np.float32)
        bottom = NIR + R
        bottom[bottom == 0] = 0.01
        ndvi = (NIR - R) / bottom

        NDVI_LOW, NDVI_MED, NDVI_HIGH = 0.11, 0.22, 0.42
        mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        mask[ndvi < NDVI_LOW] = [0, 0, 255]
        mask[(ndvi >= NDVI_LOW) & (ndvi < NDVI_MED)] = [0, 255, 255]
        mask[(ndvi >= NDVI_MED) & (ndvi < NDVI_HIGH)] = [0, 165, 255]
        mask[ndvi >= NDVI_HIGH] = [0, 255, 0]

        B = B.astype(np.uint8)
        G = G.astype(np.uint8)
        R = R.astype(np.uint8)
        rgb = cv2.merge([B, G, R])
        ndvi_mask = cv2.addWeighted(rgb, 0.4, mask, 0.6, 0)
    else:
        rgb = img
        ndvi_mask = img.copy()

    results = model_yolo(rgb)[0]
    boxes = results.boxes.xyxy.cpu().numpy()

    crops, centers = [], []
    for (x1, y1, x2, y2) in boxes:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        crops.append(ndvi_mask[y1:y2, x1:x2])
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))

    labels = []
    for crop in crops:
        tensor = transform(crop).unsqueeze(0)
        with torch.no_grad():
            output = model_resnet(tensor)
            labels.append(torch.argmax(output, dim=1).item())

    total = len(labels)
    sehat = sum(1 for l in labels if l == 1)
    kurang_sehat = total - sehat

    colors = {0: (255, 0, 0), 1: (0, 255, 0)} 
    labels_text = {0: "Kurang Sehat", 1: "Sehat"}
    for (cx, cy), label in zip(centers, labels):
        cv2.circle(rgb, (cx, cy), 12, colors[label], -1)
        cv2.putText(rgb, labels_text[label], (cx - 30, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)

    summary = f"""
**🌴 Total Pohon Sawit:** {total} pohon  
🟢 **Sehat:** {sehat}  
🔴 **Kurang Sehat:** {kurang_sehat}
"""

    return Image.fromarray(rgb), summary

demo = gr.Interface(
    fn=predict_health,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil", label="Output Image"), gr.Markdown()],
    title="🌴 Palm Tree Health Detector",
    description="Upload a 4-channel image (RGB + NIR) or a normal image for palm tree health detection."
)

if __name__ == "__main__":
    demo.launch()

