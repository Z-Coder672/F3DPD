"""
Test best_print_classifier_t4.pth on all images in the dataset.
Optimized for Apple Silicon (MPS backend).
"""

import torch
import torch.nn as nn
from PIL import Image
import clip
from pathlib import Path
from tqdm import tqdm
import csv
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

# Apple Silicon device selection
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Using device: {device}")

# Load CLIP model
print("Loading CLIP ViT-B/32 model...")
clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
clip_model.eval()

# ============================================================================
# Image Preprocessing (16:9 warp + cover crop to 224x224)
# ============================================================================

class WarpToSquareCover:
    def __init__(self, size=224, base_aspect=16 / 9, warp_axis="width"):
        if warp_axis not in ("width", "height"):
            raise ValueError("warp_axis must be 'width' or 'height'")
        self.size = size
        self.base_aspect = base_aspect
        self.warp_axis = warp_axis

    def __call__(self, img):
        w, h = img.size

        if self.warp_axis == "width":
            new_w = max(1, int(round(w / self.base_aspect)))
            new_h = h
        else:
            new_w = w
            new_h = max(1, int(round(h * self.base_aspect)))

        if (new_w, new_h) != (w, h):
            img = F.resize(img, (new_h, new_w), interpolation=InterpolationMode.BICUBIC)

        scale = max(self.size / new_w, self.size / new_h)
        scaled_w = max(1, int(round(new_w * scale)))
        scaled_h = max(1, int(round(new_h * scale)))
        img = F.resize(img, (scaled_h, scaled_w), interpolation=InterpolationMode.BICUBIC)
        img = F.center_crop(img, (self.size, self.size))
        return img


preprocess = transforms.Compose([
    WarpToSquareCover(size=224, base_aspect=16 / 9, warp_axis="width"),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711),
    ),
])

class PrintClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = features.float()
        return self.classifier(features)

# Load model
model = PrintClassifier(clip_model).to(device)
checkpoint = torch.load('best_print_classifier_t4_v2.pth', map_location=device, weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"Loaded model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.2f}%")

# Collect all images
def collect_images(root_dir):
    """Collect all images with their ground truth labels."""
    images = []
    root = Path(root_dir)
    
    # Success images (label 0)
    for folder in ['success', 'successful']:
        folder_path = root / folder
        if folder_path.exists():
            for img_path in folder_path.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    images.append((str(img_path), 0, 'success'))
    
    # Failed images (label 1)
    failed_path = root / 'failed'
    if failed_path.exists():
        for img_path in failed_path.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                images.append((str(img_path), 1, 'failed'))
    
    return images

# Collect from all dataset folders
all_images = []
for split in ['train', 'validation', 'test']:
    split_path = Path('dataset') / split
    if split_path.exists():
        imgs = collect_images(split_path)
        for img_path, label, label_name in imgs:
            all_images.append((img_path, label, label_name, split))
        print(f"Found {len(imgs)} images in {split}")

print(f"\nTotal images to test: {len(all_images)}")

# Run inference
results = []
correct = 0
total = 0
class_correct = [0, 0]
class_total = [0, 0]
misclassified = []

print("\nRunning inference...")
with torch.no_grad():
    for img_path, true_label, true_name, split in tqdm(all_images):
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_label = outputs.argmax(1).item()
            confidence = probs[0][pred_label].item() * 100
            
            pred_name = 'success' if pred_label == 0 else 'failed'
            is_correct = pred_label == true_label
            
            results.append({
                'path': img_path,
                'split': split,
                'true_label': true_name,
                'predicted': pred_name,
                'confidence': f"{confidence:.1f}%",
                'correct': is_correct
            })
            
            total += 1
            class_total[true_label] += 1
            if is_correct:
                correct += 1
                class_correct[true_label] += 1
            else:
                misclassified.append({
                    'path': img_path,
                    'true': true_name,
                    'predicted': pred_name,
                    'confidence': confidence
                })
                
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

# Synchronize MPS operations before reporting
if device == "mps":
    torch.mps.synchronize()

# Print results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Overall Accuracy: {100*correct/total:.2f}% ({correct}/{total})")
print(f"Success Accuracy: {100*class_correct[0]/class_total[0]:.2f}% ({class_correct[0]}/{class_total[0]})")
print(f"Failed Accuracy:  {100*class_correct[1]/class_total[1]:.2f}% ({class_correct[1]}/{class_total[1]})")

if misclassified:
    print(f"\n{'='*60}")
    print(f"MISCLASSIFIED ({len(misclassified)} images)")
    print("="*60)
    for m in misclassified:
        print(f"  {Path(m['path']).name}")
        print(f"    True: {m['true']}, Predicted: {m['predicted']} ({m['confidence']:.1f}%)")

# Save results to CSV
csv_path = 'test_results_t4_mps.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['path', 'split', 'true_label', 'predicted', 'confidence', 'correct'])
    writer.writeheader()
    writer.writerows(results)
print(f"\nResults saved to {csv_path}")
