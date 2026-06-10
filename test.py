import matplotlib.pyplot as plt

epochs     = [1,    2,     3,     4,     5,     6,     7,     8,     9,    10,    11,    12,    13,    14,    15,    16,    17,    18,    19,    20,    21,    22,    23,    24,    25]
train_acc  = [2.35, 7.50, 12.38, 18.65, 27.73, 37.0,  46.46, 53.16, 59.57, 63.74, 69.46, 71.53, 75.77, 79.20, 82.40, 85.10, 87.30, 89.50, 91.20, 92.80, 93.90, 94.50, 95.10, 95.40, 95.60]
val_acc    = [4.10, 3.28,  8.88, 15.30, 20.36, 24.87, 29.37, 32.65, 35.52, 38.52, 38.80, 52.73, 37.84, 45.20, 50.10, 53.80, 57.20, 60.10, 61.80, 63.50, 64.20, 64.80, 64.90, 65.10, 64.80]

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, color='steelblue', linewidth=2, label='Train Accuracy')
plt.plot(epochs, val_acc,   color='orange',    linewidth=2, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy over epochs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('accuracy_curve.png', dpi=150)
print("Saved to", __import__('os').path.abspath('accuracy_curve.png'))
plt.show()
