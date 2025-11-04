# Data Augmentation - Quick Start

## 🎯 One-Sentence Summary

**Label only 101 original images, automatically generate augmented images and labels to get 303 training data!**

## 📋 Preparation (1 minute only)

```bash
# 1. Install dependencies
pip install opencv-python numpy labelImg

# 2. Confirm files are ready
ls data/labels_detection/predefined_classes.txt
```

## 🚀 Three Steps to Complete (15 hours → Complete all annotations)

### Step 1: Label Original Images (about 13.5 hours)

```bash
# Start LabelImg, label only the 101 original images
labelImg data/images data/labels_detection data/labels_detection/predefined_classes.txt
```

**Labeling tips:**
- Press `W` to create bounding box
- Select class from list (no manual input needed)
- Press `Ctrl+S` to save
- Press `D` for next image
- **Only label the 101 original images!**

### Step 2: Generate Augmented Data (about 2 minutes)

```bash
# Automatically generate mirrored and rotated versions of images + labels
python scripts/generate_augmented_dataset.py
```

**Script will automatically:**
- ✅ Generate 101 mirrored images (horizontal flip)
- ✅ Generate 101 rotated images (counterclockwise 90 degrees)
- ✅ Generate corresponding 202 label files
- ✅ Skip existing files (support incremental update)

### Step 3: Verify Results (about 5 minutes)

```bash
# Check file count
ls data/images/*.jpg | wc -l        # Should be 303
ls data/labels_detection/*.txt | wc -l  # Should be 305 (303 labels + 2 config)

# Visual verification
python scripts/visualize_dataset.py
```

## ✅ Done!

Now you have:
- ✅ 303 images (101 original + 101 mirrored + 101 rotated)
- ✅ 303 label files
- ✅ Ready to train the model!

## 🔄 Incremental Workflow (Recommended)

No need to label all at once, can do in batches:

```bash
# Daily workflow
labelImg data/images data/labels_detection data/labels_detection/predefined_classes.txt  # Label a few
python scripts/generate_augmented_dataset.py                                              # Generate augmented data

# Repeat above steps until all 101 are done
```

## 📊 File Structure

```
data/
├── images/
│   ├── FloorPlan-1-xxx.jpg         # Original (you label)
│   ├── FloorPlan-1-xxx_flip.jpg    # Mirrored (auto-generated)
│   ├── FloorPlan-1-xxx_rot90.jpg   # Rotated (auto-generated)
│   └── ...
│
└── labels_detection/
    ├── predefined_classes.txt       # Predefined classes (18)
    ├── classes.txt                  # YOLO format classes
    ├── FloorPlan-1-xxx.txt         # Original labels
    ├── FloorPlan-1-xxx_flip.txt    # Mirrored labels (auto)
    ├── FloorPlan-1-xxx_rot90.txt   # Rotated labels (auto)
    └── ...
```

## 🎓 Predefined Classes (18)

```
0.  door
1.  window
2.  table
3.  chair
4.  bed
5.  sofa
6.  toilet
7.  sink
8.  bathtub
9.  stove
10. refrigerator
11. wardrobe
12. tv
13. desk
14. washingmachine
15. loadbearing_wall
16. aircondition
17. cupboard
```

## ⚙️ Advanced Options

```bash
# Force regeneration (overwrite existing files)
python scripts/generate_augmented_dataset.py --force

# View detailed options
python scripts/generate_augmented_dataset.py --help
```

## 🐛 Troubleshooting

### Issue 1: LabelImg not showing class list

**Solution:** Confirm `predefined_classes.txt` is in correct location
```bash
cat data/labels_detection/predefined_classes.txt
```

### Issue 2: Augmentation script error

**Solution:** Check if opencv-python is installed
```bash
pip install opencv-python
```

### Issue 3: Wrong number of generated images

**Solution:** View script output, look for warnings
```bash
python scripts/generate_augmented_dataset.py 2>&1 | grep "Warning"
```

## 📚 More Documentation

- **Complete Tutorial**: [docs/数据增强工作流程.md](../docs/数据增强工作流程.md)
- **LabelImg Usage**: [docs/如何使用LabelImg标注.md](../docs/如何使用LabelImg标注.md)
- **Model Training**: [docs/TUTORIAL_02_模型训练.md](../docs/TUTORIAL_02_模型训练.md)

## 🎉 Time Saved

- **Traditional method**: 303 images × 8 min = **40 hours**
- **This method**: 101 images × 8 min + 2 min = **13.5 hours**
- **Saved**: **26.5 hours (66%)** 

---

**Start labeling! Remember: Only label 101 images, let the script handle the rest!** 🚀

