# Bearing Fault Detection Using Signal Analysis and 1D-CNN

This project implements an end-to-end machine learning pipeline for bearing fault diagnosis using the CWRU vibration dataset.

The pipeline:
- Loads raw accelerometer MAT files from `raw/`.
- Extracts drive-end vibration signals (`*_DE_time`).
- Generates fixed windows of time-series data.
- Applies signal processing features:
  - FFT magnitude profile.
  - STFT spectral profile.
- Builds a 3-channel 1D input:
  - Channel 1: normalized raw signal.
  - Channel 2: normalized FFT feature profile.
  - Channel 3: normalized STFT feature profile.
- Trains a GPU-enabled PyTorch 1D-CNN classifier.
- Classifies into 4 health states:
  - healthy
  - inner_race_fault
  - outer_race_fault
  - ball_fault

## Project Files

- `src/data_utils.py`: data loading, preprocessing, FFT/STFT feature construction, splitting.
- `src/model.py`: 1D-CNN architecture.
- `src/train.py`: training and evaluation entry point.
- `requirements.txt`: dependencies.

## Environment Setup

Create or activate your virtual environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

For NVIDIA GPU training with CUDA wheels (recommended):

```powershell
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU visibility:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

## Train the 1D-CNN

From the workspace root:

```powershell
python src/train.py --device auto --epochs 35 --batch-size 256
```

Useful overrides:

```powershell
python src/train.py --device cuda --epochs 50 --learning-rate 5e-4 --window-size 2048 --step-size 1024
```

If you want to force regeneration of dataset cache:

```powershell
python src/train.py --rebuild-cache
```

## Outputs

Training creates the `artifacts/` folder with:
- `best_model.pt`: best checkpoint by validation accuracy.
- `metrics.json`: device info, dataset metadata, split sizes, metrics.
- `classification_report.txt`: precision/recall/F1 summary.
- `training_curves.png`: loss and accuracy trends.
- `confusion_matrix.png`: confusion matrix visualization.
- `dataset_cache.npz`: cached preprocessed tensors for faster reruns.

## Notes

- The script also reads `feature_time_48k_2048_load_1.csv` to log dataset-level feature summary and label counts for sanity checking.
- The classifier is trained on features generated from raw vibration windows to satisfy signal-analysis-driven diagnosis.
