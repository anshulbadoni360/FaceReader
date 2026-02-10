# Face Emotion API

Fast facial emotion analysis using Action Units.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Copy your model files
cp /path/to/best_model.pt checkpoints/
cp /path/to/scaler_params.json checkpoints/

# Copy your face pipeline
cp -r /path/to/face ./face/

# Run
python run.py
