# ml/training/train_distillation.py
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.models.base_models import LightMultiDetect, TeacherModel, distillation_loss
from ml.data.datasets import DeepfakeDataset
from ml.training.utils import AverageMeter, ProgressMeter, save_checkpoint, accuracy


def train_epoch(train_loader, student_model, teacher_model, criterion, optimizer, epoch, args):
    """Train for one epoch"""
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    student_model.train()
    if teacher_model:
        teacher_model.eval()

    end = time.time()
    for i, (images, targets) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        # Move data to the same device as model
        images = images.to(args.device)
        targets = targets.to(args.device)

        # Compute output
        student_logits, _ = student_model(images)

        # Knowledge distillation
        if teacher_model:
            with torch.no_grad():
                teacher_logits = teacher_model(images)
            loss = distillation_loss(
                student_logits, 
                teacher_logits, 
                targets, 
                alpha=args.alpha, 
                temperature=args.temperature
            )
        else:
            loss = criterion(student_logits, targets)

        # Measure accuracy and record loss
        acc1 = accuracy(student_logits, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, args):
    """Validate the model"""
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1],
        prefix='Test: ')

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, targets) in enumerate(val_loader):
            # Move data to the same device as model
            images = images.to(args.device)
            targets = targets.to(args.device)

            # Compute output
            output, _ = model(images)
            loss = criterion(output, targets)

            # Measure accuracy and record loss
            acc1 = accuracy(output, targets)
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return losses.avg, top1.avg


def main(args):
    # Create saving directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up TensorBoard
    writer = SummaryWriter(os.path.join(args.output_dir, 'tensorboard'))
    
    # Set up device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = DeepfakeDataset(
        args.data_dir,
        split='train',
        transform=True,
        image_size=args.image_size
    )
    
    val_dataset = DeepfakeDataset(
        args.data_dir,
        split='val',
        transform=True,
        image_size=args.image_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Create student model
    student_model = LightMultiDetect(num_classes=args.num_classes)
    student_model = student_model.to(args.device)
    
    # Create teacher model if using distillation
    if args.distillation:
        teacher_model = TeacherModel(num_classes=args.num_classes)
        
        # Load pretrained teacher model
        if args.teacher_checkpoint:
            print(f"Loading teacher checkpoint from {args.teacher_checkpoint}")
            checkpoint = torch.load(args.teacher_checkpoint, map_location=args.device)
            teacher_model.load_state_dict(checkpoint['state_dict'])
            
        teacher_model = teacher_model.to(args.device)
        teacher_model.eval()  # Teacher model is always in eval mode
    else:
        teacher_model = None
    
    # Define loss function
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    # Define optimizer
    optimizer = optim.Adam(
        student_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            student_model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"No checkpoint found at {args.resume}")
    else:
        args.start_epoch = 0
        best_acc1 = 0
    
    # Train the model
    for epoch in range(args.start_epoch, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            train_loader,
            student_model,
            teacher_model,
            criterion,
            optimizer,
            epoch,
            args
        )
        
        # Evaluate on validation set
        val_loss, val_acc = validate(
            val_loader,
            student_model,
            criterion,
            args
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Track progress with TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)
        
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': student_model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.output_dir)
        
    # Close TensorBoard writer
    writer.close()
    
    print(f"Training completed. Best validation accuracy: {best_acc1:.2f}%")


# ml/training/utils.py
import os
import torch
import shutil


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """Display training progress"""
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target):
    """Compute accuracy given model output and targets"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.max(1)
        correct = pred.eq(target).sum().item()
        return 100.0 * correct / batch_size


def save_checkpoint(state, is_best, output_dir):
    """Save training checkpoint"""
    filename = os.path.join(output_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(output_dir, 'model_best.pth.tar'))


# ml/training/export.py
import torch
import onnx
import os
import argparse
from ml.models.base_models import LightMultiDetect


def export_onnx(model, output_path, input_shape=(1, 3, 224, 224), dynamic_axes=True):
    """
    Export PyTorch model to ONNX format
    
    Args:
        model: PyTorch model to export
        output_path: Path to save ONNX model
        input_shape: Input tensor shape
        dynamic_axes: Whether to use dynamic axes for variable batch size
    
    Returns:
        Path to exported model
    """
    # Create dummy input
    dummy_input = torch.randn(input_shape)
    
    # Set up dynamic axes if needed
    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    
    # Export model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        verbose=False,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes_dict,
        opset_version=13
    )
    
    # Verify exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"Model exported to {output_path}")
    return output_path


def export_torchscript(model, output_path):
    """
    Export PyTorch model to TorchScript format
    
    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
    
    Returns:
        Path to exported model
    """
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save model
    torch.jit.save(traced_model, output_path)
    
    print(f"Model exported to {output_path}")
    return output_path


def optimize_for_mobile(model, output_path):
    """
    Optimize model for mobile deployment
    
    Args:
        model: PyTorch model to optimize
        output_path: Path to save optimized model
    
    Returns:
        Path to exported model
    """
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Trace model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save model
    optimized_model.save(output_path)
    
    print(f"Model optimized for mobile and exported to {output_path}")
    return output_path


def main(args):
    # Load model
    model = LightMultiDetect(num_classes=args.num_classes)
    
    # Load weights
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    
    # Set model to evaluation mode
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export to ONNX
    if 'onnx' in args.formats:
        onnx_path = os.path.join(args.output_dir, 'model.onnx')
        export_onnx(model, onnx_path)
    
    # Export to TorchScript
    if 'torchscript' in args.formats:
        script_path = os.path.join(args.output_dir, 'model.pt')
        export_torchscript(model, script_path)
    
    # Optimize for mobile
    if 'mobile' in args.formats:
        mobile_path = os.path.join(args.output_dir, 'model_mobile.pt')
        optimize_for_mobile(model, mobile_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export model to various formats')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', default='exported_models', help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['onnx', 'torchscript', 'mobile'],
                        help='Export formats (onnx, torchscript, mobile)')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of output classes')
    
    args = parser.parse_args()
    main(args)


# ml/evaluation/evaluate.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
import seaborn as sns
import pandas as pd
import time
from tqdm import tqdm

from ml.models.base_models import LightMultiDetect
from ml.data.datasets import DeepfakeDataset


def evaluate_model(model, data_loader, device):
    """
    Evaluate model performance
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader with test data
        device: Device to run evaluation on
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    # Tracking variables
    y_true = []
    y_scores = []
    inference_times = []
    
    # Evaluate with no grad
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            images = images.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs, _ = model(images)
            end_time = time.time()
            
            # Get predictions and scores
            probabilities = torch.softmax(outputs, dim=1)
            scores = probabilities[:, 1].cpu().numpy()  # Score for positive class
            
            # Add to tracking
            y_true.extend(targets.cpu().numpy())
            y_scores.extend(scores)
            inference_times.append((end_time - start_time) * 1000)  # ms
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # Calculate metrics
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auroc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    
    # Find best threshold based on F1 score
    f1_scores = []
    thresholds = np.arange(0, 1, 0.01)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Compute final predictions with best threshold
    y_pred = (y_scores >= best_threshold).astype(int)
    
    # Compute accuracy
    accuracy = np.mean(y_pred == y_true)
    
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Average inference time
    avg_inference_time = np.mean(inference_times[1:])  # Skip first as it includes warm-up
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'auprc': auprc,
        'best_threshold': best_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'precision_curve': precision,
        'recall_curve': recall,
        'inference_time': avg_inference_time,
        'confusion_matrix': np.array([[tn, fp], [fn, tp]])
    }
    
    return metrics


def plot_metrics(metrics, output_dir):
    """
    Plot evaluation metrics
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot ROC curve
    plt.figure(figsize=(10, 8))
    plt.plot(metrics['fpr'], metrics['tpr'], label=f'AUROC = {metrics["auroc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 8))
    plt.plot(metrics['recall_curve'], metrics['precision_curve'], label=f'AUPRC = {metrics["auprc"]:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics summary
    metrics_summary = {
        'Accuracy': f"{metrics['accuracy']:.4f}",
        'Precision': f"{metrics['precision']:.4f}",
        'Recall': f"{metrics['recall']:.4f}",
        'F1 Score': f"{metrics['f1']:.4f}",
        'AUROC': f"{metrics['auroc']:.4f}",
        'AUPRC': f"{metrics['auprc']:.4f}",
        'Best Threshold': f"{metrics['best_threshold']:.4f}",
        'Avg. Inference Time (ms)': f"{metrics['inference_time']:.2f}"
    }
    
    # Convert to DataFrame and save as CSV
    pd.DataFrame(metrics_summary.items(), columns=['Metric', 'Value']).to_csv(
        os.path.join(output_dir, 'metrics_summary.csv'), index=False
    )
    
    # Also save as text file
    with open(os.path.join(output_dir, 'metrics_summary.txt'), 'w') as f:
        for metric, value in metrics_summary.items():
            f.write(f"{metric}: {value}\n")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = LightMultiDetect(num_classes=args.num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = DeepfakeDataset(
        args.data_dir,
        split='test',
        transform=True,
        image_size=args.image_size
    )
    
    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, device)
    
    # Print summary
    print("\nEvaluation Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    print(f"Best Threshold: {metrics['best_threshold']:.4f}")
    print(f"Average Inference Time: {metrics['inference_time']:.2f} ms")
    
    # Plot and save metrics
    plot_metrics(metrics, args.output_dir)
    print(f"Plots and metrics saved to {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate deepfake detection model')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-dir', required=True, help='Path to dataset')
    parser.add_argument('--output-dir', default='evaluation_results', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes')
    
    args = parser.parse_args()
    main(args)