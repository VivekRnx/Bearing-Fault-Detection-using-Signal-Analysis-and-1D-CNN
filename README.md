# Bearing Fault Detection Using Signal Analysis and 1D-CNN

End-to-end bearing fault diagnosis pipeline built on the CWRU vibration dataset.

This project:
- Loads raw accelerometer signals from MAT files.
- Applies signal analysis (FFT and STFT) on fixed windows.
- Builds 3-channel 1D features for each window.
- Trains a PyTorch 1D-CNN to classify bearing condition.
- Uses NVIDIA GPU automatically when CUDA is available.

## Classes

The model predicts 4 classes:
- healthy
- inner_race_fault
- outer_race_fault
- ball_fault

## Project Structure

```
raw/                        # CWRU MAT files
src/
  data_utils.py             # Data loading + preprocessing + feature extraction
  model.py                  # 1D-CNN architecture
  train.py                  # Training/evaluation entry point
artifacts/                  # Generated model and reports
feature_time_48k_2048_load_1.csv
requirements.txt
README.md
```

## Signal Processing and Features

For each signal window:
- Channel 1: z-score normalized time-domain waveform.
- Channel 2: FFT magnitude profile (log-scaled + resampled + z-score).
- Channel 3: STFT profile (mean spectrum over time, log-scaled + resampled + z-score).

Default preprocessing parameters:
- sampling rate: 48000 Hz
- window size: 2048
- step size: 1024

## Model

`src/model.py` defines a compact 1D-CNN with:
- stacked Conv1d + BatchNorm + ReLU blocks
- MaxPool downsampling
- AdaptiveAvgPool1d head
- dropout-regularized MLP classifier

The model uses a fixed architecture:
- conv channels: 32, 64, 128, 256
- kernel sizes: 7, 5, 5, 3
- FC hidden units: 128
- dropout: 0.30, then 0.20

## Environment Setup

PowerShell (from project root):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Enable NVIDIA GPU (Recommended)

Install CUDA-enabled PyTorch wheels:

```powershell
pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify GPU:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU found')"
```

## Train

Default full training:

```powershell
python src/train.py --device auto --epochs 35 --batch-size 256
```

If you want to rebuild cache from raw MAT files first:

```powershell
python src/train.py --rebuild-cache --device auto --epochs 35 --batch-size 256
```

Quick smoke run (small subset):

```powershell
python src/train.py --rebuild-cache --max-windows-per-file 30 --epochs 1 --num-workers 0 --device auto
```

## Useful CLI Options

`src/train.py` supports:
- `--raw-dir` custom MAT folder
- `--csv-path` feature CSV for summary logging
- `--output-dir` output folder (default: `artifacts`)
- `--cache-path` dataset cache path
- `--rebuild-cache` force rebuilding dataset cache
- `--window-size` and `--step-size` preprocessing controls
- `--sampling-rate` sampling frequency
- `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`
- `--num-workers` DataLoader workers
- `--pin-memory` optional CUDA pinned memory (opt-in)
- `--no-class-weights` disable class-weighted loss
- `--device {auto,cuda,cpu}`

## Outputs

After training, files are saved to `artifacts/`:
- `best_model.pt` best checkpoint by validation accuracy
- `dataset_cache.npz` cached feature tensor dataset
- `metrics.json` run configuration, device info, metrics, confusion matrix
- `classification_report.txt` precision/recall/F1 report
- `training_curves.png` train/val loss + accuracy curves
- `confusion_matrix.png` confusion matrix plot

## Notes

- The feature CSV is used for summary/sanity logging. Training data is built from raw MAT signals.
- Class-weighted cross-entropy is enabled by default to reduce class-imbalance bias.
- On some Windows systems, leaving `--pin-memory` off is more stable.

## Troubleshooting

If CUDA is not detected:
- Ensure NVIDIA driver is installed and up to date.
- Reinstall CUDA wheels using the PyTorch index URL shown above.
- Confirm `torch.cuda.is_available()` returns `True`.

If training is slow or unstable on your laptop:
- Start with `--num-workers 0`.
- Use smaller `--batch-size` (for example `128`).
- Keep `--pin-memory` disabled unless your system handles it well.
