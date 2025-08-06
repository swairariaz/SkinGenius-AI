"""
SKIN GENIUS - CLASSIFIER (MobileNetV3) v2.1
===========================================
Purpose:
- Analyzes skin images for medical conditions.
- Maps results to cosmetic concerns.

Key Technologies:
- PyTorch (MobileNetV3 architecture)
- HAM10000 dataset (training)
- Pure-PIL augmentations (no OpenCV dependency)

Workflow:
1. Preprocesses image (resize/normalize).
2. Predicts condition (7 medical labels).
3. Converts to skincare terms (e.g., "akiec" → "redness/dryness").
4. Recommends ingredients (e.g., aloe for redness).

Output:
{
  "condition": "mel",
  "confidence": 0.92,
  "cosmetic_concerns": ["acne", "blackheads"],
  "ingredients": ["niacinamide", "green tea"]
}
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageEnhance
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Dict, Optional, List

# ===== CONFIG =====
class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data", "skin_images")
    METADATA_PATH = os.path.join(DATA_DIR, "HAM10000_metadata.csv")
    MODEL_PATH = os.path.join(BASE_DIR, "models", "skin_model.pt")

    MEDICAL_LABELS = {
        'akiec': 0, 'bcc': 1, 'bkl': 2,
        'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6
    }

    CONDITION_MAPPING = {
        'akiec': ['redness', 'dryness'],
        'bcc': ['pores', 'dullness'],
        'bkl': ['dullness'],
        'df': ['dryness'],
        'mel': ['acne', 'blackheads'],
        'nv': ['oiliness'],
        'vasc': ['redness']
    }

    BATCH_SIZE = 32
    EPOCHS = 25
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DATASET WITH PURE-PIL AUGMENTATIONS ====
class SkinDataset(Dataset):
    def __init__(self, df, transform=None, augment: bool = True):
        self.df = df
        self.augment = augment
        self.transform = transform or self._get_transforms()
        self._load_image_paths()

    def _load_image_paths(self):
        """Load all valid image paths"""
        self.image_paths = []
        self.labels = []

        for _, row in self.df.iterrows():
            img_name = row['image_id'] + '.jpg'
            for part in ['part_1', 'part_2']:
                img_path = os.path.join(Config.DATA_DIR, f'HAM10000_images_{part}', img_name)
                if os.path.exists(img_path):
                    self.image_paths.append(img_path)
                    self.labels.append(Config.MEDICAL_LABELS[row['dx']])
                    break

    def _get_transforms(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')

        # PIL-based augmentations (no cv2 needed)
        if self.augment and np.random.rand() > 0.5:
            # Color jitter
            img = transforms.functional.adjust_brightness(img, brightness_factor=np.random.uniform(0.7, 1.3))
            img = transforms.functional.adjust_contrast(img, contrast_factor=np.random.uniform(0.8, 1.2))

            # Rotation
            if np.random.rand() > 0.5:
                img = transforms.functional.rotate(img, angle=np.random.randint(-15, 15))

            # Flip
            if np.random.rand() > 0.5:
                img = transforms.functional.hflip(img)

        return self.transform(img), self.labels[idx]


# ===== MODEL =====
class SkinModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.base = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        in_features = self.base.classifier[0].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)


# ===== CLASSIFIER WITH CONFIDENCE =====
class SkinClassifier:
    def __init__(self):
        self.device = Config.DEVICE
        self.model = SkinModel().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        # Load existing model if available
        if os.path.exists(Config.MODEL_PATH):
            self.model.load_state_dict(
                torch.load(Config.MODEL_PATH, map_location=self.device))
            print("Loaded pre-trained model")
        else:
            print("No model found - training required")

    def train(self):
        """Trains with aggressive augmentations"""
        df = pd.read_csv(Config.METADATA_PATH)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        train_loader = DataLoader(
            SkinDataset(train_df, augment=True),
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            SkinDataset(val_df, augment=False),
            batch_size=Config.BATCH_SIZE
        )

        best_acc = 0
        for epoch in range(Config.EPOCHS):
            self.model.train()
            running_loss = 0.0

            for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_acc = self._validate(val_loader)
            print(f'Epoch {epoch + 1} - Loss: {running_loss / len(train_loader):.4f} - Val Acc: {val_acc:.2f}%')

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), Config.MODEL_PATH)
                print('💾 Saved new best model!')

    def _validate(self, loader):
        """Validation with clean images"""
        self.model.eval()
        correct = 0
        total = 0

        with (((torch.no_grad()))):
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return 100 * correct / total

    def predict(self, image_path: str) -> Dict:
        """Predict with confidence checks and fallbacks"""
        try:
            # Validate image
            if not os.path.exists(image_path):
                return {'error': 'image_not_found', 'fallback': 'Please check the file path'}

            img = Image.open(image_path).convert('RGB')

            # Transform without augmentations
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                output = self.model(img_tensor)
                prob = torch.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(prob).item()
                confidence = prob[pred_idx].item()

            # Confidence-based response
            if confidence < 0.7:
                return {
                    'condition': 'uncertain',
                    'confidence': confidence,
                    'message': 'Please describe your concern in text for better analysis'
                }

            # Get condition details
            rev_map = {v: k for k, v in Config.MEDICAL_LABELS.items()}
            condition = rev_map[pred_idx]
            concerns = Config.CONDITION_MAPPING.get(condition, ['dullness'])

            # Safe ingredient mapping
            SAFE_INGREDIENTS = {
                'acne': ['niacinamide', 'green tea'],
                'oiliness': ['witch hazel', 'clay'],
                'redness': ['centella asiatica', 'aloe vera'],
                'dryness': ['hyaluronic acid', 'honey'],
                'dullness': ['vitamin C', 'rosehip oil']
            }

            ingredients = []
            for concern in concerns:
                ingredients.extend(SAFE_INGREDIENTS.get(concern, ['aloe vera']))

            return {
                'condition': condition,
                'confidence': float(confidence),
                'concerns': concerns,
                'ingredients': list(set(ingredients)),  # Remove duplicates
                'warning': 'Consult a dermatologist for serious conditions' if confidence < 0.8 else None
            }

        except Exception as e:
            return {
                'error': 'prediction_failed',
                'technical': str(e),
                'fallback': 'Please describe your skin concern in text'
            }


# ===== TEST =====
if __name__ == "__main__":
    classifier = SkinClassifier()

    # Train if no model
    if not os.path.exists(Config.MODEL_PATH):
        print("Training...")
        classifier.train()

    # Test prediction
    test_img = os.path.join(Config.DATA_DIR, "HAM10000_images_part_1", "ISIC_0024307.jpg")
    result = classifier.predict(test_img if os.path.exists(test_img) else "dummy.jpg")

    print("\n=== TEST RESULTS ===")
    print(f"Condition: {result.get('condition', 'error')}")
    print(f"Confidence: {result.get('confidence', 0):.1%}")
    if 'ingredients' in result:
        print("Recommended Ingredients:")
        for i, ing in enumerate(result['ingredients'], 1):
            print(f"{i}. {ing.title()}")
    if 'error' in result:
        print(f"Error: {result['error']} - {result.get('fallback', '')}")