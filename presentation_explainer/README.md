# XTTS v2 TTS Project

This is a custom setup for running **XTTS v2** from [Coqui TTS](https://github.com/coqui-ai/TTS ) with support for multilingual speech synthesis, including Arabic.

> ⚠️ This repo does **not include large files** like models (`xtts_v2/`), or virtual environments.

---
## 📦 Manual Model Installation (Important)
The XTTS v2 model is **not included in this repository** due to its large size. You need to download it separately from Hugging Face.

### 1. **Download the Model**

Go to:  
👉 [https://huggingface.co/coqui/XTTS-v2 ](https://huggingface.co/coqui/XTTS-v2 )

Click on the **"Files and versions"** tab, and download these files:

- `model.pth`
- `config.json`
- `vocab.json`
- `speaker_xtts.pth`

### 2. **Create a Folder in Your Project**

Inside your project root folder, create:

```bash
mkdir xtts_v2/
put it all the files related to the model
## 🧾 Requirements

- **Python 3.10** (recommended)

> ⚠️ Newer versions of Python (e.g., 3.11+) may cause issues with unpickling due to PyTorch's `weights_only=True` default behavior.

---

## 🛠️ Setup Instructions

### 1. Clone the repo
```bash
git clone https://github.com/rahafAlhallay/XTTS.git 
cd XTTS

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # On Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run your script
python xtts_clone.py
