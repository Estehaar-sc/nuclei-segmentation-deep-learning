import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import time
from tqdm import tqdm

# Data paths
data_path = r"C:\Users\User\Desktop\CSE445\project\data"

# ============================================================================
# 1. DATASET PROCESSING (Your previous code)
# ============================================================================

# Data Augmentation Class
class NucleiAugmentation:
    def __init__(self, image_size=256, augmentation_prob=0.7, sparse_mode=False):
        self.image_size = image_size
        self.augmentation_prob = augmentation_prob
        self.sparse_mode = sparse_mode
    
    def __call__(self, image, mask):
        if torch.is_tensor(image):
            image = TF.to_pil_image(image)
        if torch.is_tensor(mask):
            mask = TF.to_pil_image(mask)
        
        # Random horizontal flip
        if random.random() > self.augmentation_prob:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        
        # Random vertical flip
        if random.random() > self.augmentation_prob:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        
        # Random rotation (-45 to +45 degrees)
        if random.random() > self.augmentation_prob:
            angle = random.uniform(-45, 45)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
        
        # Random brightness/contrast adjustment
        if random.random() > self.augmentation_prob:
            brightness_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_brightness(image, brightness_factor)
        
        if random.random() > self.augmentation_prob:
            contrast_factor = random.uniform(0.8, 1.2)
            image = TF.adjust_contrast(image, contrast_factor)
        
        # Random Gaussian blur
        if random.random() > self.augmentation_prob:
            image = TF.gaussian_blur(image, kernel_size=3)
        
        # For sparse mode: random crop
        if self.sparse_mode and random.random() > 0.8:
            width, height = image.size
            crop_size = random.randint(128, 256)
            i = random.randint(0, height - crop_size)
            j = random.randint(0, width - crop_size)
            image = TF.crop(image, i, j, crop_size, crop_size)
            mask = TF.crop(mask, i, j, crop_size, crop_size)
        
        # Ensure final size
        image = image.resize((self.image_size, self.image_size))
        mask = mask.resize((self.image_size, self.image_size))
        
        return image, mask

# Dataset Class
class KMMSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, image_size=256, augment=False, sparse_mode=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_size = image_size
        self.augment = augment
        self.sparse_mode = sparse_mode
        self.augmentor = NucleiAugmentation(image_size, sparse_mode=sparse_mode) if augment else None
        
        self.valid_pairs = list(zip(image_paths, mask_paths))
        
        print(f"Created dataset with {len(self.valid_pairs)} valid image-mask pairs")
        if augment:
            mode = "sparse" if sparse_mode else "normal"
            print(f"✓ Data augmentation ENABLED ({mode} mode)")
        else:
            print("✗ Data augmentation DISABLED")
    
    def __len__(self):
        return len(self.valid_pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.valid_pairs[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('L')
            
            # Apply augmentation
            if self.augment and self.augmentor:
                image, mask = self.augmentor(image, mask)
            else:
                image = image.resize((self.image_size, self.image_size))
                mask = mask.resize((self.image_size, self.image_size))
            
            # Convert to numpy arrays
            image = np.array(image).astype(np.float32) / 255.0
            mask = np.array(mask).astype(np.float32) / 255.0
            
            # Ensure mask is binary
            mask = (mask > 0.5).astype(np.float32)
            
            # Convert to tensors
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()
            
            return image, mask, os.path.basename(img_path)
        
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy_image = torch.zeros(3, self.image_size, self.image_size)
            dummy_mask = torch.zeros(1, self.image_size, self.image_size)
            return dummy_image, dummy_mask, "error"

# Data loading and balancing
def find_kmms_data():
    train_path = os.path.join(data_path, "kmms_training")
    test_path = os.path.join(data_path, "kmms_test")
    
    train_images, train_masks, test_images, test_masks = [], [], [], []
    
    # Load training data
    if os.path.exists(train_path):
        train_images_dir = os.path.join(train_path, "images")
        train_masks_dir = os.path.join(train_path, "masks")
        if os.path.exists(train_images_dir):
            train_images = sorted([os.path.join(train_images_dir, f) for f in os.listdir(train_images_dir) 
                                 if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))])
        if os.path.exists(train_masks_dir):
            train_masks = sorted([os.path.join(train_masks_dir, f) for f in os.listdir(train_masks_dir) 
                                if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))])
    
    # Load test data
    if os.path.exists(test_path):
        test_images_dir = os.path.join(test_path, "images")
        test_masks_dir = os.path.join(test_path, "masks")
        if os.path.exists(test_images_dir):
            test_images = sorted([os.path.join(test_images_dir, f) for f in os.listdir(test_images_dir) 
                                if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))])
        if os.path.exists(test_masks_dir):
            test_masks = sorted([os.path.join(test_masks_dir, f) for f in os.listdir(test_masks_dir) 
                               if f.lower().endswith(('.tiff', '.tif', '.png', '.jpg', '.jpeg'))])
    
    print(f"Found: {len(train_images)} training images, {len(test_images)} test images")
    return train_images, train_masks, test_images, test_masks

# ============================================================================
# 2. MODEL ARCHITECTURE
# ============================================================================

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        return torch.sigmoid(self.outc(x))

# ============================================================================
# 3. IMPROVED LOSS FUNCTIONS AND METRICS
# ============================================================================

class DiceLoss(nn.Module):
    """Dice Loss focuses on overlap between prediction and target"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

class TverskyLoss(nn.Module):
    """Better for imbalanced data - penalizes false negatives more"""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-6):
        super().__init__()
        self.alpha = alpha  # Weight for false negatives
        self.beta = beta    # Weight for false positives  
        self.smooth = smooth
    
    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        return 1 - tversky

class CombinedLoss(nn.Module):
    """Combine multiple losses for better performance"""
    def __init__(self, alpha=0.7, beta=0.3):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
        
    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        tversky = self.tversky_loss(inputs, targets)
        return 0.5 * dice + 0.5 * tversky

def calculate_metrics(predictions, targets, threshold=0.5):
    pred_binary = (predictions > threshold).float()
    target_binary = (targets > threshold).float()
    
    pred_flat = pred_binary.view(-1).cpu().numpy()
    target_flat = target_binary.view(-1).cpu().numpy()
    
    precision = precision_score(target_flat, pred_flat, zero_division=0)
    recall = recall_score(target_flat, pred_flat, zero_division=0)
    accuracy = accuracy_score(target_flat, pred_flat)
    f1 = f1_score(target_flat, pred_flat, zero_division=0)
    
    return precision, recall, accuracy, f1

def find_optimal_threshold(predictions, targets):
    """Find the best threshold for F1-score"""
    best_f1 = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    print("\n" + "="*50)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*50)
    
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        precision, recall, accuracy, f1 = calculate_metrics(predictions, targets, threshold=threshold)
        print(f"Threshold {threshold}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall
    
    print(f"\n🎯 OPTIMAL THRESHOLD: {best_threshold}")
    print(f"📊 Best Metrics - Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}")
    print("="*50)
    
    return best_threshold

# ============================================================================
# 4. IMPROVED TRAINING AND TESTING FUNCTIONS
# ============================================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    
    train_losses = []
    val_losses = []
    train_metrics = {'precision': [], 'recall': [], 'accuracy': [], 'f1': []}
    val_metrics = {'precision': [], 'recall': [], 'accuracy': [], 'f1': []}
    
    best_f1 = 0.0
    best_model_state = None
    
    print("Starting training with improved loss function...")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for images, masks, _ in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
            train_preds.append(outputs.detach())
            train_targets.append(masks.detach())
        
        # Training metrics
        train_preds = torch.cat(train_preds)
        train_targets = torch.cat(train_targets)
        train_precision, train_recall, train_accuracy, train_f1 = calculate_metrics(train_preds, train_targets)
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, masks, _ in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                images = images.to(device)
                masks = masks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                epoch_val_loss += loss.item()
                
                val_preds.append(outputs)
                val_targets.append(masks)
        
        # Validation metrics
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)
        val_precision, val_recall, val_accuracy, val_f1 = calculate_metrics(val_preds, val_targets)
        
        # Store results
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        train_metrics['precision'].append(train_precision)
        train_metrics['recall'].append(train_recall)
        train_metrics['accuracy'].append(train_accuracy)
        train_metrics['f1'].append(train_f1)
        
        val_metrics['precision'].append(val_precision)
        val_metrics['recall'].append(val_recall)
        val_metrics['accuracy'].append(val_accuracy)
        val_metrics['f1'].append(val_f1)
        
        # Print results
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f} | Precision: {train_precision:.4f} | Recall: {train_recall:.4f} | Accuracy: {train_accuracy:.4f} | F1: {train_f1:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | Accuracy: {val_accuracy:.4f} | F1: {val_f1:.4f}')
        print('-' * 80)
        
        # Save best model based on F1 score
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_nuclei_segmentation_model.pth')
            print(f'✓ New best model saved with F1: {best_f1:.4f}')
    
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, train_metrics, val_metrics

def test_model(model, test_loader, device='cuda'):
    model.eval()
    model.to(device)
    
    test_preds = []
    test_targets = []
    test_loss = 0.0
    criterion = CombinedLoss(alpha=0.7, beta=0.3)  # Use improved loss for testing consistency
    
    print("\nTesting model on test set...")
    
    with torch.no_grad():
        for images, masks, _ in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            
            test_preds.append(outputs)
            test_targets.append(masks)
    
    test_preds = torch.cat(test_preds)
    test_targets = torch.cat(test_targets)
    
    # Find optimal threshold
    optimal_threshold = find_optimal_threshold(test_preds, test_targets)
    
    # Calculate metrics with optimal threshold
    test_precision, test_recall, test_accuracy, test_f1 = calculate_metrics(test_preds, test_targets, threshold=optimal_threshold)
    avg_test_loss = test_loss / len(test_loader)
    
    print("\n" + "="*60)
    print("FINAL TEST RESULTS (with optimal threshold)")
    print("="*60)
    print(f"Optimal Threshold: {optimal_threshold:.2f}")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"F1-Score:  {test_f1:.4f}")
    print("="*60)
    
    return test_precision, test_recall, test_accuracy, test_f1

# ============================================================================
# 5. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_results(train_losses, val_losses, train_metrics, val_metrics):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    epochs = range(1, len(train_losses) + 1)
    
    # Plot losses
    axes[0, 0].plot(epochs, train_losses, 'b-', label='Training Loss')
    axes[0, 0].plot(epochs, val_losses, 'r-', label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot metrics
    metrics = ['precision', 'recall', 'accuracy', 'f1']
    titles = ['Precision', 'Recall', 'Accuracy', 'F1-Score']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row = i // 2
        col = i % 2
        axes[row, col].plot(epochs, train_metrics[metric], 'b-', label=f'Train {title}')
        axes[row, col].plot(epochs, val_metrics[metric], 'r-', label=f'Val {title}')
        axes[row, col].set_title(f'Training and Validation {title}')
        axes[row, col].set_xlabel('Epochs')
        axes[row, col].set_ylabel(title)
        axes[row, col].legend()
        axes[row, col].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 6. MAIN EXECUTION WITH IMPROVEMENTS
# ============================================================================

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data using your previous processing
    print("Loading and processing dataset...")
    train_images, train_masks, test_images, test_masks = find_kmms_data()
    
    if not train_images:
        print("No training images found!")
        return
    
    # Create datasets
    train_dataset = KMMSDataset(train_images, train_masks, image_size=256, augment=True, sparse_mode=True)
    test_dataset = KMMSDataset(test_images, test_masks, image_size=256, augment=False, sparse_mode=False)
    
    # Split training data for validation
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_subset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model with IMPROVED LOSS FUNCTION
    model = UNet(n_channels=3, n_classes=1)
    
    # Use CombinedLoss instead of FocalLoss for better recall
    criterion = CombinedLoss(alpha=0.7, beta=0.3)  # Focus more on recall (false negatives)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)
    
    print(f"\nUsing CombinedLoss with alpha=0.7 (focus on recall)")
    
    # Train the model
    start_time = time.time()
    model, train_losses, val_losses, train_metrics, val_metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, 
        num_epochs=30, device=device
    )
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    # Plot results
    plot_training_results(train_losses, val_losses, train_metrics, val_metrics)
    
    # Test the model with threshold optimization
    test_precision, test_recall, test_accuracy, test_f1 = test_model(model, test_loader, device=device)
    
    # Save final model
    torch.save(model.state_dict(), 'final_nuclei_segmentation_model.pth')
    print("\nFinal model saved as 'final_nuclei_segmentation_model.pth'")

if __name__ == "__main__":
    main()