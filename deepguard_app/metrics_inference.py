import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
import face_recognition
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# MODEL CLASS (same as before)
class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DeepFakeDetector, self).__init__()
        model = models.resnext50_32x4d(weights='IMAGENET1K_V1')
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        out = self.dropout(self.linear1(x_lstm[:, -1, :]))
        return fmap, out

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = r"E:\deepguard_django\models\model_90_acc_60_frames_final_data.pt"
model = DeepFakeDetector().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Preprocessing
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Predict function
def predict(model, frames_tensor):
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        fmap, logits = model(frames_tensor.to(device))
        probs = softmax(logits)
        _, pred_class = torch.max(probs, 1)
        confidence = probs[0][pred_class.item()].item() * 100
        return pred_class.item(), confidence, probs.cpu().numpy()

# Inference function (return also probabilities for ROC)
def run_inference(video_path, frame_count=20):
    video_tensor = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // frame_count, 1)
    selected = 0

    for i in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        face_locations = face_recognition.face_locations(frame)
        if face_locations:
            top, right, bottom, left = face_locations[0]
            face = frame[top:bottom, left:right]
        else:
            face = frame

        try:
            face_tensor = transform(face)
            video_tensor.append(face_tensor)
            selected += 1
        except:
            continue

        if selected >= frame_count:
            break

    cap.release()

    if len(video_tensor) < 5:
        return 'UNCERTAIN', 0.0, None

    video_tensor = torch.stack(video_tensor).unsqueeze(0)
    label_idx, confidence, prob = predict(model, video_tensor)
    label = 'REAL' if label_idx == 1 else 'FAKE'
    return label, confidence, prob[0][1]  # Return REAL class probability

# ----------------- FULL INFERENCE ---------------------
real_folder = r"E:\back\1.6\real_val"
fake_folder = r"E:\back\1.6\fake_val"

y_true, y_pred, y_prob = [], [], []

# Real videos
for filename in os.listdir(real_folder):
    if filename.endswith('.mp4'):
        path = os.path.join(real_folder, filename)
        label, conf, prob = run_inference(path)
        y_true.append(1)
        y_pred.append(1 if label == 'REAL' else 0)
        y_prob.append(prob)
        print(f"Real: {filename} => {label} ({conf:.2f}%)")

# Fake videos
for filename in os.listdir(fake_folder):
    if filename.endswith('.mp4'):
        path = os.path.join(fake_folder, filename)
        label, conf, prob = run_inference(path)
        y_true.append(0)
        y_pred.append(1 if label == 'REAL' else 0)
        y_prob.append(prob)
        print(f"Fake: {filename} => {label} ({conf:.2f}%)")

# -------------------- METRICS --------------------
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=["FAKE", "REAL"])

print("\n---- METRICS ----")
print(f"Accuracy: {acc*100:.2f}%")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:\n", report)

# -------------------- PLOTTING --------------------

# ROC Curve
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("AUC-ROC Curve")
plt.legend(loc="lower right")
plt.savefig("auc_roc.png")
plt.close()

print(" ROC Curve Saved!")
