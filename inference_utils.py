import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from torch.utils.data import Dataset
import face_recognition

# ========= Device Setup =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========= Model Architecture =========
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        resnext = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(resnext.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()  
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# ========= Load Model =========
MODEL_PATH = r"E:\deepguard_django\models\model_90_acc_60_frames_final_data.pt"
model = Model().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ========= Transform =========
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# ========= Dataset Class =========
class validation_dataset(Dataset):
    def __init__(self, video_paths, sequence_length, transform=None):
        self.video_paths = video_paths
        self.transform = transform
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        frames = []
        a = max(1, int(100 / self.sequence_length))
        first_frame = np.random.randint(0, a)

        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.sequence_length:
                break

        frames = torch.stack(frames)
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vid = cv2.VideoCapture(path)
        while True:
            success, frame = vid.read()
            if not success:
                break
            yield frame

# ========= Prediction Function =========
sm = nn.Softmax(dim=1)

def predict(model, img):
    with torch.no_grad():
        fmap, logits = model(img.to(device))
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, prediction.item()].item() * 100
        return prediction.item(), round(confidence, 2)

# ========= Main Inference Pipeline =========
def run_inference(video_path, frame_count=10):
    video_dataset = validation_dataset([video_path], sequence_length=frame_count, transform=transform)
    input_tensor = video_dataset[0]  # shape: (1, T, C, H, W)
    label_idx, confidence = predict(model, input_tensor)
    label = "REAL" if label_idx == 1 else "FAKE"
    return label, confidence

# ========= Extract Suspicious Frames =========
def extract_suspicious_frames(video_path, output_dir='media/frames/'):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // 5, 1)
    paths = []

    for i in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        path = os.path.join(output_dir, f'frame_{i}.jpg')
        cv2.imwrite(path, frame)
        paths.append(path)

    cap.release()
    return paths

# ========= Extract Faces =========
def extract_faces(video_path, output_dir='media/faces/'):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // 5, 1)
    paths = []
    idx = 0

    for i in range(0, total, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        faces = face_recognition.face_locations(frame)
        for (top, right, bottom, left) in faces:
            face_img = frame[top:bottom, left:right]
            path = os.path.join(output_dir, f'face_{idx}.jpg')
            cv2.imwrite(path, face_img)
            paths.append(path)
            idx += 1

    cap.release()
    return paths


# Helper used for metrics page
def extract_frames_tensor(video_path, transform, frame_count=20):
    video_tensor = []
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // frame_count, 1)
    selected = 0

    for i in range(0, total, interval):
        if selected >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue
        faces = face_recognition.face_locations(frame)
        if faces:
            top, right, bottom, left = faces[0]
            face = frame[top:bottom, left:right]
        else:
            face = frame
        try:
            face_tensor = transform(face)
            video_tensor.append(face_tensor)
            selected += 1
        except:
            continue
    cap.release()
    return torch.stack(video_tensor).unsqueeze(0)
