# Handwritten Medicine Recognition

A ResNet50 transfer-learning project for classifying cropped handwritten medicine names in the Doctor's Handwritten Prescription BD dataset. The Streamlit app shows the top match and its model confidence.

The dataset contains 78 medicine-name classes, with 40 training, 10 validation, and 10 test images per class before deduplication. Some exact image copies occur across splits; a few copies also have conflicting labels. Training prioritizes the held-out `Testing` split, then `Validation`, then `Training`, excluding repeated copies from later splits and excluding all copies with conflicting labels. The reported test accuracy is measured on the remaining unique test images. It is a small dataset, so 85% accuracy is a target, not a guarantee.

This recognizes unseen handwriting **for the 78 known medicine names**. It cannot identify a medicine outside those classes. This is a research/demo tool, not a medical device; verify any result with a qualified professional.

## Setup (Windows PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For faster training on an NVIDIA GPU, install the CUDA-enabled PyTorch build
recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/)
for your system instead of a CPU-only build. Check that
`python -c "import torch; print(torch.cuda.is_available())"` prints `True`.

## Train and evaluate

Use the dataset root folder containing `Training`, `Validation`, and `Testing`:

```powershell
python train.py --data-dir "C:\path\to\Doctor’s Handwritten Prescription BD dataset"
```

The script downloads ImageNet pretrained weights on its first run, fine-tunes the final ResNet50 stages, selects the best checkpoint using validation accuracy, and evaluates it on the held-out test set. Medicine crops are kept at their full 224-pixel width in a 224×112 input, avoiding excess blank space while retaining handwriting detail. Outputs are saved to `artifacts\resnet50_medicine.pt` and `artifacts\evaluation.json`.

Continue training from the current best checkpoint without starting over:

```powershell
python train.py --data-dir "C:\path\to\dataset" --resume --epochs 35 --patience 35
```

Optional settings:

```powershell
python train.py --data-dir "C:\path\to\dataset" --epochs 35 --batch-size 32 --patience 35
```

## Launch the app

After training:

```powershell
streamlit run app.py
```

Upload a cropped image with one medicine name, or use the camera option. The app displays its top prediction and flags confidence below 50%. Confidence is not a guarantee of correctness; verify every result with a qualified professional.
