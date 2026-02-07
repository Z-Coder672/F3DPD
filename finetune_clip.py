import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode
from PIL import Image
import clip
from pathlib import Path
from tqdm import tqdm
import os

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load CLIP model
print("Loading CLIP ViT-B/32 model...")
clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)

# Freeze CLIP model - we only train the classifier head
for param in clip_model.parameters():
    param.requires_grad = False

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

# Create classifier head
class PrintClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        # CLIP ViT-B/32 outputs 512-dim features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)  # 2 classes: success, failed
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.clip_model.encode_image(x)
            features = features.float()
        return self.classifier(features)

model = PrintClassifier(clip_model).to(device)

# Dataset class
class PrintDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load successful prints (label 0)
        for folder_name in ['success', 'successful']:
            success_dir = Path(root_dir) / folder_name
            if success_dir.exists():
                for img_path in success_dir.glob('*'):
                    if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        self.samples.append((str(img_path), 0))
        
        # Load failed prints (label 1)
        failed_dir = Path(root_dir) / 'failed'
        if failed_dir.exists():
            for img_path in failed_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} images from {root_dir}")
        success_count = sum(1 for _, label in self.samples if label == 0)
        failed_count = sum(1 for _, label in self.samples if label == 1)
        print(f"  Success: {success_count}, Failed: {failed_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Training function
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total

# Validation function
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Track per-class accuracy
    class_correct = [0, 0]
    class_total = [0, 0]
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == labels[i]).item()
                class_total[label] += 1
    
    overall_acc = 100. * correct / total
    success_acc = 100. * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
    failed_acc = 100. * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
    
    return total_loss / len(loader), overall_acc, success_acc, failed_acc


if __name__ == '__main__':
    # Create datasets
    train_dataset = PrintDataset('dataset/train', transform=preprocess)
    val_dataset = PrintDataset('dataset/validation', transform=preprocess)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Training loop
    num_epochs = 20
    best_val_acc = 0

    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 60)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, success_acc, failed_acc = validate(model, val_loader, criterion)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  Success Acc: {success_acc:.2f}%, Failed Acc: {failed_acc:.2f}%")
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_print_classifier.pth')
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*60)

    # Save final model
    torch.save(model.state_dict(), 'final_print_classifier.pth')
    print("\nModels saved:")
    print("  - best_print_classifier.pth (best validation accuracy)")
    print("  - final_print_classifier.pth (final epoch)")
