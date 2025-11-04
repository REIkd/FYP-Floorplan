# Auto Label Generation Tool - Quick Reference

## 🎯 Goal
Label only 101 original images, automatically generate labels for the other 202 (mirrored + rotated), **save 65% of annotation time!**

## 📝 Complete Workflow (3 Steps)

### Step 1: Check Image Order ✅

**First verify images are in expected order:**

```bash
# Check image order and augmentation relationship
python scripts/check_image_order.py
```

**Expected result:**
- ✓ Mirror relationship correct
- ✓ Rotation relationship correct

If check fails, images are in wrong order and need reorganization.

---

### Step 2: Label Original Images ✏️

**Label only the first 101 original images:**

```bash
# Start LabelImg
labelImg data/images data/labels_detection data/labels_detection/predefined_classes.txt
```

**Labeling tips:**
- Only label first 101
- Don't worry about the other 202
- Label normally, save

**Label Classes (18):**
```
0.  door                   10. refrigerator
1.  window                 11. wardrobe
2.  table                  12. tv
3.  chair                  13. desk
4.  bed                    14. washingmachine
5.  sofa                   15. loadbearing_wall
6.  toilet                 16. aircondition
7.  sink                   17. cupboard
8.  bathtub
9.  stove
```

---

### Step 3: Auto-Generate Other Labels 🚀

**After labeling 101 images, run generation script:**

```bash
# Automatically generate labels for mirrored and rotated images
python scripts/auto_generate_labels.py
```

Script will automatically:
- Read labels from original images (1-101)
- Generate labels for mirrored images (102-202)
- Generate labels for rotated images (203-303)

**Verify generation results:**

```bash
# Visual check
python scripts/visualize_dataset.py
```

---

## 🔧 Common Commands

### Test coordinate transformation
```bash
python scripts/auto_generate_labels.py --test
```

### List all images
```bash
python scripts/check_image_order.py --list-all
```

### View image grouping
```bash
python scripts/check_image_order.py
```

### Count labels
```bash
# Windows PowerShell
(Get-ChildItem data\labels_detection\*.txt).Count

# Linux/Mac
ls data/labels_detection/*.txt | wc -l
```

---

## 📊 Coordinate Transformation Explanation

### Horizontal Flip (Mirror)
```
Original coords: (x, y, w, h)
Mirrored coords: (1-x, y, w, h)
```

**Example:**
```
Original: 0.3 0.5 0.1 0.2  → Mirror: 0.7 0.5 0.1 0.2
(30% from left)            (becomes 70% from left)
```

### Rotate 90 degrees (Counterclockwise)
```
Original coords: (x, y, w, h)
Rotated coords: (y, 1-x, h, w)
```

**Example:**
```
Original: 0.3 0.7 0.1 0.2  → Rotated: 0.7 0.7 0.2 0.1
(30% horizontal, 70% vertical) (coordinates and w/h both change)
```

---

## ⚠️ Important Notes

### 1. Image order is critical!
Script assumes:
- Images 1-101 = Original
- Images 102-202 = Mirrored
- Images 203-303 = Rotated

**Please run `check_image_order.py` first to verify!**

### 2. Incremental running
- Script won't overwrite existing labels
- Can label part of original images then run
- Continue labeling later, run again

### 3. Regenerate
To regenerate certain labels:
```bash
# Delete labels to regenerate
rm data/labels_detection/FloorPlan-102-*.txt

# Rerun script
python scripts/auto_generate_labels.py
```

---

## 🎓 Workflow Example

### Scenario: Label 10 images per day

**Day 1:**
```bash
# 1. Label first 10 original images
labelImg data/images data/labels_detection

# 2. Generate corresponding 20 augmented labels
python scripts/auto_generate_labels.py
# Result: 30 labels total
```

**Day 2:**
```bash
# 3. Continue labeling images 11-20
labelImg data/images data/labels_detection

# 4. Run again, only generates new 20 labels
python scripts/auto_generate_labels.py
# Result: 60 labels total
```

**Repeat until all 101 are done!**

---

## ✅ Completion Checklist

After labeling, check:

- [ ] Labeled 101 original images
- [ ] Ran auto-generation script
- [ ] Have 303 label files (.txt)
- [ ] Used visualization tool to verify a few
- [ ] Mirrored image labels are correct (left-right symmetric)
- [ ] Rotated image labels are correct (rotated 90 degrees)
- [ ] `classes.txt` file is complete (18 lines)

---

## 🆘 Troubleshooting

### Problem 1: Generated label positions are wrong

**Solution:**
```bash
# 1. Check image order
python scripts/check_image_order.py

# 2. If order is wrong, reorganize images
# 3. Confirm augmentation method (mirror/rotation direction)
```

### Problem 2: Script skips some images

**Reason:** Original images not yet labeled

**Solution:**
```bash
# 1. Check which original images are missing labels
python scripts/check_image_order.py --list-all

# 2. Label missing original images
labelImg data/images data/labels_detection

# 3. Rerun script
python scripts/auto_generate_labels.py
```

### Problem 3: classes.txt lost again

**Solution:**
```bash
# Create backup
copy data\labels_detection\classes.txt data\labels_detection\classes.txt.backup

# Or use Git version control
git add data/labels_detection/classes.txt
git commit -m "Backup classes.txt"
```

---

## 📚 Related Documentation

- **Detailed Tutorial**: [docs/标注工作流程_自动增强.md](../docs/标注工作流程_自动增强.md)
- **Annotation Guide**: [docs/TUTORIAL_01_数据标注指南.md](../docs/TUTORIAL_01_数据标注指南.md)
- **Model Training**: [docs/TUTORIAL_02_模型训练.md](../docs/TUTORIAL_02_模型训练.md)

---

## 🎉 Time Saved

- **Traditional method**: 303 images × 8 min = 40 hours
- **Auto method**: 101 images × 8 min = 13.5 hours
- **Saved**: **26.5 hours (65%)** 🎊

---

**Happy labeling! Remember: Only label 101 images, let the script handle the rest!** 🚀

