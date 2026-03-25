# StyleGAN2 Latent Explorer

Read more in my [blog](https://medium.com/merzazine/stylegan2-locally-part-2-infinite-latent-space-browser-4e53432e28fd?sk=14d557b399c601402345aa5cca73e05e)

**A local toolkit for exploring StyleGAN2 models trained on your own images.**  
Navigate, browse, and export from the latent space of your GAN — running entirely on your own GPU, no cloud required.

> Created by [Claude](https://claude.ai/) & [Merzmensch](https://merzmensch.com/) for the worldwide GenAI creative community.

---

## What is this?

StyleGAN2 Latent Explorer gives you three different ways to explore the latent space of any StyleGAN2 model trained on your own photos or artwork:

| Tool | Description |
|------|-------------|
| **Latent Browser** | 2D PCA projection of your model as an interactive grid. Move your mouse to explore, click to preview at full resolution. |
| **Walk Explorer** | Smooth random walk through latent space using spherical interpolation (slerp). Pin positions, jump to seeds, record MP4 videos. |
| **Infinite Latent Map** | Drag to pan through an infinite 2D slice of latent space. Tiles generate on demand. |

All three tools are connected — you can send a position from the Browser or Map directly into the Walk Explorer and continue from there.

---

## Requirements

- Windows 10/11 (Linux/Mac also works with minor path adjustments)
- NVIDIA GPU (GTX 1080 or better recommended, 8GB+ VRAM)
- CUDA 11.8, 12.1, or 12.4+
- A trained StyleGAN2 `.pkl` model file
- ffmpeg (for MP4 export)

---

## Installation

### Step 1 — Install Miniconda

Download **Miniconda3 Windows 64-bit** from:  
👉 https://docs.conda.io/en/latest/miniconda.html

During installation:
- ✅ Create shortcuts
- ❌ **Do NOT add to PATH** (keeps your system clean, avoids conflicts)
- ❌ Do not register as default Python

After installation, open **Anaconda Prompt** from the Start Menu for all following steps.

---

### Step 2 — Check your CUDA version

```bash
nvidia-smi
```

Look for `CUDA Version: XX.X` in the top right. Note this number.

---

### Step 3 — Create a Conda environment

```bash
conda create -n stylegan python=3.9
conda activate stylegan
```

---

### Step 4 — Install PyTorch with CUDA

Choose the command matching your CUDA version:

**CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CUDA 12.4 or higher (including 12.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verify GPU is detected:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

You should see something like `2.6.0+cu124` and `True`.

---

### Step 5 — Install remaining dependencies

```bash
pip install flask flask-cors ninja Pillow requests
```

---

### Step 6 — Clone StyleGAN2-ADA-PyTorch

Navigate to your project folder first, then clone:

```bash
cd C:\Users\YourName\stylegan2-explorer
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git
```

---

### Step 7 — Install ffmpeg (for MP4 export)

Download **ffmpeg-release-essentials.zip** from:  
👉 https://www.gyan.dev/ffmpeg/builds/

Extract to `C:\ffmpeg\` so the structure looks like:
```
C:\ffmpeg\ffmpeg-X.X-essentials_build\bin\ffmpeg.exe
```

Then update the ffmpeg path in `server.py`. Find this line and adjust to your exact path:
```python
r'C:\ffmpeg\ffmpeg-X.X-essentials_build\bin\ffmpeg.exe', '-y',
```

---

### Step 8 — Set up the project folder

Your final folder structure should look like this:

```
stylegan2-explorer/
├── stylegan2-ada-pytorch/     ← cloned in Step 6
├── models/
│   └── your_model.pkl         ← place your StyleGAN2 .pkl model here
├── server.py
├── index.html
├── latent_explorer.html
├── latent_browser.html
├── latent_infinite.html
└── start.bat
```

---

### Step 9 — Configure start.bat

Edit `start.bat` and update the username and model filename:

```batch
@echo off
title StyleGAN2 Latent Explorer

call C:\Users\YourName\miniconda3\Scripts\activate.bat stylegan

cd /d C:\Users\YourName\stylegan2-explorer

start http://localhost:5000

python server.py --pkl models\your_model.pkl

pause
```

Replace `YourName` with your Windows username and `your_model.pkl` with your actual model filename.

---

## Running

Double-click `start.bat` — a black CMD window opens, the server starts, and your browser opens automatically at `http://localhost:5000`.

You will see the landing page with links to all three tools.

---

## Usage

### Latent Browser (`/browser`)

1. Click **Build Grid** — samples 512 random z-vectors, computes PCA, generates thumbnails (30–120 seconds depending on grid size and GPU)
2. Move your mouse over the grid — a full-resolution preview appears on the right
3. Click any position to pin its z-vector
4. Click **→ Walk Explorer** to open that exact position in the Walk Explorer

### Walk Explorer (`/ui`)

| Key | Action |
|-----|--------|
| Space | Play / pause walk |
| → | Step once |
| R | Random jump |
| P | Pin current position |
| B | Recall pinned position |
| S | Save current frame |

Use **⏺ Start Recording** → walk around → **⏹ Stop & Save MP4** to export a video to the project folder.

### Infinite Latent Map (`/infinite`)

| Action | Effect |
|--------|--------|
| Drag | Pan through latent space |
| Scroll | Zoom in / out |
| Click tile | Preview + pin z-vector |
| R | Random jump to new region |
| 0 | Reset view to origin |
| E | Send pinned position to Walk Explorer |

Adjust **Truncation ψ**: low values (0.3–0.5) = similar, average images. High values (0.8–1.2) = diverse, extreme images.

Adjust **Sampling Distance** to control how different adjacent tiles look.

---

## Compatible model formats

The server uses `legacy.load_network_pkl` from stylegan2-ada-pytorch which handles both:

- **PyTorch** `.pkl` files (stylegan2-ada-pytorch format)
- **TensorFlow** `.pkl` files (original NVIDIA StyleGAN2 format — converted automatically on load)

---

## VRAM notes

Your GPU VRAM is shared between all running applications. If you run ComfyUI, LM Studio, or other GPU applications alongside the explorer, you may run low on VRAM with larger models. Close or idle other GPU applications if you encounter out-of-memory errors.

Tested on RTX 4070 (8GB) with 1024×1024 models.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'legacy'`**  
Make sure `stylegan2-ada-pytorch` is cloned inside your project folder and `server.py` has the correct path at the top.

**`ModuleNotFoundError: No module named 'dnnlib.tflib'`**  
Your `.pkl` is a TensorFlow-format model. The `legacy` loader handles this automatically — make sure you are using the `server.py` from this repo which includes the path fix.

**`conda: command not found` in Git Bash**  
Use **Anaconda Prompt** from the Start Menu instead of Git Bash. Git Bash does not see Conda by default.

**MP4 export fails**  
Check that the ffmpeg path in `server.py` matches your exact installation path including the version number in the folder name.

**Walk repeats the same image after a while**  
Make sure you have the latest `server.py` — earlier versions had a bug in the slerp interpolation where `z_start` was not updated correctly, causing the walk to converge to a fixed point.

**Browser shows connection error**  
Make sure `server.py` is running (check the CMD window), then refresh the browser. The server needs a few seconds to start before the browser can connect.

---

## Server command line options

```bash
python server.py --pkl models/your_model.pkl --port 5000 --grid 6 --res 128
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pkl` | _(required)_ | Path to your `.pkl` model file |
| `--port` | `5000` | Port to run the server on |
| `--grid` | `6` | Default grid size for Latent Browser |
| `--res` | `128` | Thumbnail resolution for grid |

---

## License

MIT — free to use, modify, and share.

---

## Acknowledgements

- [NVIDIA StyleGAN2-ADA PyTorch](https://github.com/NVlabs/stylegan2-ada-pytorch) — the generator backbone
- [RunwayML ML Lab](https://runwayml.com/) — the original inspiration for this project
