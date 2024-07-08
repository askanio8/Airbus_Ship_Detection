# 🚢 Airbus Ship Detection Test Task

## Project Description

The goal of the project is to build a semantic segmentation model to detect ships in images.

## Model Architecture

The model is based on the U-Net architecture, which is widely used for image segmentation tasks. Binary cross-entropy was used as the loss function, and Dice Score was used as the metric.

## Installation and Usage

1. Clone the repository:

```bash
git clone https://github.com/username/airbus-ship-detection.git
cd airbus-ship-detection
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model:

```bash
python src/train_model.py
```

4. Run inference:

```bash
python src/inference_model.py
```
