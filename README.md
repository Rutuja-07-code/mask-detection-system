# Mask Detection System

This project builds a face-mask detection system from the dataset at:

`/Volumes/HP USB20FD/Smart Workplace Mask Compliance Monitor 2`

The training flow uses the XML bounding boxes in that dataset to crop faces, trains a 3-class PyTorch classifier, and then combines the trained classifier with OpenCV face detection for runtime inference.

The three supported classes are:

- `with_mask`
- `without_mask`
- `mask_weared_incorrect`

## Project Layout

- `prepare_dataset.py`: converts XML annotations into cropped `train/val/test` face images
- `train.py`: trains the face-mask classifier and saves the best checkpoint
- `predict.py`: runs mask detection on an image, folder, or webcam
- `src/mask_detection/`: shared training and inference code

## Requirements

Install the Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## 1. Prepare the Dataset

This reads the XML annotations from your dataset and creates cropped face images under `data/mask_faces`.

```bash
python3 prepare_dataset.py \
  --dataset-root "/Volumes/HP USB20FD/Smart Workplace Mask Compliance Monitor 2" \
  --output-dir data/mask_faces \
  --force
```

## 2. Train the Model

This trains a lightweight CNN classifier and stores the best model in `artifacts/best_model.pt`.

```bash
python3 train.py \
  --dataset-root "/Volumes/HP USB20FD/Smart Workplace Mask Compliance Monitor 2" \
  --data-dir data/mask_faces \
  --epochs 12 \
  --batch-size 32 \
  --image-size 128
```

If you want `train.py` to rebuild the cropped dataset before training, add `--force-prepare`.

## 3. Run Inference

Run on one image:

```bash
python3 predict.py \
  --checkpoint artifacts/best_model.pt \
  --image path/to/test_image.jpg \
  --output-dir predictions
```

Run on a folder of images:

```bash
python3 predict.py \
  --checkpoint artifacts/best_model.pt \
  --input-dir path/to/images \
  --output-dir predictions
```

Run with webcam:

```bash
python3 predict.py --checkpoint artifacts/best_model.pt --webcam
```

Press `q` to close the webcam window.

## Notes

- The dataset contains 853 images and 4,072 labeled faces.
- The training code automatically handles class imbalance by weighting the loss function.
- On CPU, training may take some time. Lower `--epochs` or `--image-size` for faster experiments.
