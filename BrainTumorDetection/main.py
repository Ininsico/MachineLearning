import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import os
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'C:\\Users\\arsla\\.cache\\kagglehub\\models\\eyppler\\vbai-2-5\\pyTorch\\2.5f-fast\\1\\Vbai-2.5f.pt'
MODEL_TYPE = 'f'

DEMENTIA_DIR = 'C:\\Users\\arsla\\Desktop\\MachineLearning\\BrainTumorDetection\\data\\dementia'
TUMOR_DIR = 'C:\\Users\\arsla\\Desktop\\MachineLearning\\BrainTumorDetection\\data\\tumor'

TEST_IMAGE_PATH = 'download.jpg'

DEM_CLASSES = ['AD Alzheimer Diseases', 'AD Mild Demented', 'AD Moderate Demented',
               'AD Very Mild Demented', 'CN Non Demented', 'PD Parkinson Diseases']
TUM_CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']


# --- MODEL ARCHITECTURE ---
class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, max(1, in_channels // 8), 1)
        self.conv2 = nn.Conv2d(max(1, in_channels // 8), 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.conv2(F.relu(self.conv1(x))))
        return x * att, att


class MultiTaskBrainModel(nn.Module):
    def __init__(self, num_dem_classes, num_tum_classes, model_type='f'):
        super(MultiTaskBrainModel, self).__init__()
        self.model_type = model_type

        if model_type == 'f':
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            final_ch = 128
        elif model_type == 'q':
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
            self.bn4 = nn.BatchNorm2d(512)
            final_ch = 512

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5 if model_type == 'q' else 0.3)

        self.edge_conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.edge_conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.edge_pool = nn.AdaptiveAvgPool2d(28)

        self.dem_att = AttentionModule(final_ch)
        self.tum_att = AttentionModule(final_ch)

        if model_type == 'q':
            self.feat_dim = final_ch * 14 * 14
            self.edge_dim = 32 * 14 * 14
            self.edge_pool = nn.AdaptiveAvgPool2d(14)
        else:
            self.feat_dim = final_ch * 28 * 28
            self.edge_dim = 32 * 28 * 28

        self.dem_fc = nn.Sequential(
            nn.Linear(self.feat_dim + self.edge_dim, 512 if model_type == 'f' else 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512 if model_type == 'f' else 1024, num_dem_classes)
        )

        self.tum_fc = nn.Sequential(
            nn.Linear(self.feat_dim + self.edge_dim, 512 if model_type == 'f' else 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512 if model_type == 'f' else 1024, num_tum_classes)
        )

    def forward(self, x, edge_x):
        if self.model_type == 'f':
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.relu(self.conv3(x))
            x = self.pool(x)
        else:
            x = self.pool(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(self.relu(self.bn3(self.conv3(x))))
            x = self.relu(self.bn4(self.conv4(x)))
            x = self.pool(x)

        e = self.pool(self.relu(self.edge_conv1(edge_x)))
        e = self.relu(self.edge_conv2(e))
        e = self.edge_pool(e)
        e_flat = e.view(e.size(0), -1)

        d_x, d_att = self.dem_att(x)
        d_flat = d_x.view(d_x.size(0), -1)
        dem_out = self.dem_fc(torch.cat([d_flat, e_flat], dim=1))

        t_x, t_att = self.tum_att(x)
        t_flat = t_x.view(t_x.size(0), -1)
        tum_out = self.tum_fc(torch.cat([t_flat, e_flat], dim=1))

        return dem_out, tum_out, d_att, t_att


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

edge_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])


def analyze_single_image(model, image_path):
    print(f"\nIs being analyzed: {image_path}")
    if not os.path.exists(image_path):
        print("File not found.")
        return

    try:
        raw_image = Image.open(image_path).convert('RGB')
        img_tensor = transform(raw_image).unsqueeze(0).to(DEVICE)
        edge_tensor = edge_transform(raw_image).unsqueeze(0).to(DEVICE)

        model.eval()
        with torch.no_grad():
            d_out, t_out, d_att, t_att = model(img_tensor, edge_tensor)

            d_prob = F.softmax(d_out, dim=1)
            t_prob = F.softmax(t_out, dim=1)

            d_pred = torch.argmax(d_prob).item()
            t_pred = torch.argmax(t_prob).item()

        print("-" * 50)
        print(f"DEMENTIA: {DEM_CLASSES[d_pred]} (%{d_prob[0][d_pred] * 100:.1f})")
        print(f"TUMOR : {TUM_CLASSES[t_pred]} (%{t_prob[0][t_pred] * 100:.1f})")
        print("-" * 50)

        d_map = d_att.detach().cpu().numpy()[0, 0]
        t_map = t_att.detach().cpu().numpy()[0, 0]

        d_map = cv2.resize(d_map, (224, 224))
        t_map = cv2.resize(t_map, (224, 224))

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.imshow(raw_image)
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(raw_image)
        plt.imshow(d_map, cmap='Blues', alpha=0.5)
        plt.title(f"Dementia Attention\n{DEM_CLASSES[d_pred]}")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(raw_image)
        plt.imshow(t_map, cmap='Reds', alpha=0.5)
        plt.title(f"Tumor Attention\n{TUM_CLASSES[t_pred]}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def evaluate_dataset_performance(model, dementia_dir, tumor_dir):
    print("\n" + "=" * 60)
    print("Starting the performance test")
    print("=" * 60)

    model.eval()

    print(f"\nDementia data set: {dementia_dir}")
    d_true, d_pred = [], []

    if os.path.exists(dementia_dir):
        for idx, cls_name in enumerate(DEM_CLASSES):
            cls_path = os.path.join(dementia_dir, cls_name)
            if not os.path.exists(cls_path): continue

            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png'))][:50]

            for fname in files:
                try:
                    img = Image.open(os.path.join(cls_path, fname)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    edge_t = edge_transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        out, _, _, _ = model(img_t, edge_t)
                        d_true.append(idx)
                        d_pred.append(torch.argmax(out, 1).item())
                except:
                    pass

        if d_true:
            print(classification_report(d_true, d_pred, target_names=DEM_CLASSES, digits=3))

    print(f"\nTumor data set: {tumor_dir}")
    t_true, t_pred = [], []

    if os.path.exists(tumor_dir):
        for idx, cls_name in enumerate(TUM_CLASSES):
            cls_path = os.path.join(tumor_dir, cls_name)
            if not os.path.exists(cls_path): continue

            files = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png'))][:50]

            for fname in files:
                try:
                    img = Image.open(os.path.join(cls_path, fname)).convert('RGB')
                    img_t = transform(img).unsqueeze(0).to(DEVICE)
                    edge_t = edge_transform(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        _, out, _, _ = model(img_t, edge_t)
                        t_true.append(idx)
                        t_pred.append(torch.argmax(out, 1).item())
                except:
                    pass

        if t_true:
            print(classification_report(t_true, t_pred, target_names=TUM_CLASSES, digits=3))


def main():
    print("System loading...")
    model = MultiTaskBrainModel(len(DEM_CLASSES), len(TUM_CLASSES), model_type=MODEL_TYPE).to(DEVICE)

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Model is ready: {MODEL_PATH}")
    except Exception as e:
        print(f"Model loading error: {e}")
        return

    print("\nOPTIONS:")
    print("1. Analyze the single image")
    print("2. Analyze the all data set")

    secim = input("Your choice (1 or 2): ")

    if secim == '1':
        analyze_single_image(model, TEST_IMAGE_PATH)
    elif secim == '2':
        evaluate_dataset_performance(model, DEMENTIA_DIR, TUMOR_DIR)
    else:
        print("Invalid choose.")


if __name__ == '__main__':
    main()
