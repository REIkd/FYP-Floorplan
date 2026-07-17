"""
Visualize Segmentation Model Training Results
"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Read training history
with open('models/segmentation/training_history.json', 'r') as f:
    history = json.load(f)

train_loss = history['train_loss']
val_loss = history['val_loss']
val_miou = history['val_miou']

epochs = range(1, len(train_loss) + 1)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Segmentation Model Training Results (U-Net with ResNet34)', 
             fontsize=16, fontweight='bold')

# 1. Training and Validation Loss
ax1 = axes[0, 0]
ax1.plot(epochs, train_loss, label='Training Loss', color='#3498db', linewidth=2)
ax1.plot(epochs, val_loss, label='Validation Loss', color='#e74c3c', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Training and Validation Loss Over Time', fontsize=13, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add best epoch annotation
best_epoch = np.argmin(val_loss) + 1
best_val_loss = min(val_loss)
ax1.annotate(f'Best Val Loss: {best_val_loss:.4f}\n(Epoch {best_epoch})',
             xy=(best_epoch, best_val_loss),
             xytext=(best_epoch + 15, best_val_loss + 0.05),
             arrowprops=dict(arrowstyle='->', color='red', lw=2),
             fontsize=10, color='red', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# 2. Validation mIoU
ax2 = axes[0, 1]
ax2.plot(epochs, [m * 100 for m in val_miou], label='Validation mIoU', 
         color='#2ecc71', linewidth=2)
ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2, label='Target (90%)')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('mIoU (%)', fontsize=12)
ax2.set_title('Validation mIoU Progress', fontsize=13, fontweight='bold')
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# Add best mIoU annotation
best_miou_epoch = np.argmax(val_miou) + 1
best_miou = max(val_miou) * 100
ax2.annotate(f'Best mIoU: {best_miou:.2f}%\n(Epoch {best_miou_epoch})',
             xy=(best_miou_epoch, best_miou),
             xytext=(best_miou_epoch - 20, best_miou - 3),
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
             fontsize=10, color='darkgreen', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

# 3. Loss Comparison (zoomed in to later epochs)
ax3 = axes[1, 0]
start_epoch = 20  # Start from epoch 20 to see detail
ax3.plot(epochs[start_epoch:], train_loss[start_epoch:], 
         label='Training Loss', color='#3498db', linewidth=2)
ax3.plot(epochs[start_epoch:], val_loss[start_epoch:], 
         label='Validation Loss', color='#e74c3c', linewidth=2)
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Loss Detail (Epoch 20-100)', fontsize=13, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# 4. Summary Statistics
ax4 = axes[1, 1]
ax4.axis('off')

# Calculate statistics
final_miou = val_miou[-1] * 100
initial_miou = val_miou[0] * 100
improvement = final_miou - initial_miou
final_train_loss = train_loss[-1]
final_val_loss = val_loss[-1]

# Create summary text
summary_text = f"""
TRAINING SUMMARY
{'='*50}

Total Epochs: {len(train_loss)}

Initial Performance (Epoch 1):
  • Val mIoU: {initial_miou:.2f}%
  • Val Loss: {val_loss[0]:.4f}

Final Performance (Epoch {len(train_loss)}):
  • Val mIoU: {final_miou:.2f}%
  • Val Loss: {final_val_loss:.4f}
  • Train Loss: {final_train_loss:.4f}

Best Performance:
  • Best Val mIoU: {best_miou:.2f}% (Epoch {best_miou_epoch})
  • Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})

Improvement: +{improvement:.2f}% mIoU

Target Achievement:
  • Target: 90% mIoU
  • Achieved: {final_miou:.2f}% mIoU
  • Status: {'✓ EXCEEDED' if final_miou >= 90 else '✗ NOT MET'}

Model Convergence:
  • Loss Reduction: {((val_loss[0] - final_val_loss) / val_loss[0] * 100):.1f}%
  • Overfitting: {'Minimal' if abs(final_train_loss - final_val_loss) < 0.03 else 'Moderate'}
"""

ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
import os
os.makedirs('results', exist_ok=True)

output_path = 'results/segmentation_training_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[SUCCESS] Training visualization saved to: {output_path}")

# Also save high-res version for report
plt.savefig('results/segmentation_training_results_highres.png', dpi=600, bbox_inches='tight')
print(f"[SUCCESS] High-res version saved to: results/segmentation_training_results_highres.png")

# Close figure to free memory
plt.close()

# Print detailed statistics
print("\n" + "="*60)
print("DETAILED TRAINING STATISTICS")
print("="*60)
print(f"\nTotal Training Epochs: {len(train_loss)}")
print(f"\nInitial mIoU: {initial_miou:.2f}%")
print(f"Final mIoU: {final_miou:.2f}%")
print(f"Best mIoU: {best_miou:.2f}% (Epoch {best_miou_epoch})")
print(f"Improvement: +{improvement:.2f}%")
print(f"\nInitial Val Loss: {val_loss[0]:.4f}")
print(f"Final Val Loss: {final_val_loss:.4f}")
print(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
print(f"\nObjective Achievement: {'EXCEEDED 90% target' if final_miou >= 90 else 'Below target'}")

# Calculate additional metrics
last_10_miou = np.mean(val_miou[-10:]) * 100
print(f"\nLast 10 Epochs Average mIoU: {last_10_miou:.2f}%")
print(f"Model Stability: {'Good (< 1% variance)' if np.std(val_miou[-10:]) * 100 < 1 else 'Moderate'}")

# Epoch milestones
milestones = [
    (10, val_miou[9] * 100, "Early Progress"),
    (25, val_miou[24] * 100, "Mid Training"),
    (50, val_miou[49] * 100, "Half Complete"),
    (75, val_miou[74] * 100, "Late Training"),
    (100, val_miou[99] * 100, "Final Result")
]

print(f"\nMilestone Performance:")
for epoch, miou, label in milestones:
    print(f"  Epoch {epoch:3d} ({label:15s}): {miou:5.2f}% mIoU")

print("\n" + "="*60)
print("Model files: models/segmentation/best_model.pth")
print("="*60)
