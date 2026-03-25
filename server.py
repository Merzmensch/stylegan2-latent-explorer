"""
StyleGAN2 Latent Space Server
==============================
Serves Walk Explorer (/ui), Latent Browser (/browser), Infinite Map (/infinite).

Usage:
    python server.py --pkl path/to/your_model.pkl

Then open:
    http://localhost:5000           ← Landing page
    http://localhost:5000/browser   ← 2D Latent Browser
    http://localhost:5000/ui        ← Walk Explorer
    http://localhost:5000/infinite  ← Infinite Latent Map
"""

import argparse, os, sys, glob, io, base64
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'stylegan2-ada-pytorch'))
import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, request, Response, send_from_directory
from flask_cors import CORS
import threading
import legacy

_recording     = False
_record_frames = []

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--pkl',  type=str, default='', help='Path to .pkl model file')
parser.add_argument('--port', type=int, default=5000)
parser.add_argument('--grid', type=int, default=6,  help='Grid size (grid x grid images)')
parser.add_argument('--res',  type=int, default=128, help='Thumbnail resolution for grid')
args = parser.parse_args()

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'[server] Device: {device}')

# ── Model state ───────────────────────────────────────────────────────────────
G         = None
Z_DIM     = None
RES       = None
PKL_PATH  = None

def load_model(path):
    global G, Z_DIM, RES, PKL_PATH
    print(f'[server] Loading: {path}')
    with open(path, 'rb') as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    G.eval()
    Z_DIM    = G.z_dim
    RES      = G.img_resolution
    PKL_PATH = path
    _reset_walk()
    _grid_cache.clear()
    _tile_cache.clear()
    print(f'[server] ✅ z_dim={Z_DIM}  resolution={RES}')

# ── Image generation ──────────────────────────────────────────────────────────
def z_to_pil(z_vector, truncation_psi=0.7, size=None):
    with torch.no_grad():
        z     = torch.tensor(z_vector, dtype=torch.float32, device=device).unsqueeze(0)
        label = torch.zeros([1, G.c_dim], device=device)
        img   = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
        img   = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil = Image.fromarray(img[0].cpu().numpy(), 'RGB')
    if size:
        pil = pil.resize((size, size), Image.LANCZOS)
    elif pil.width > 512:
        pil = pil.resize((512, 512), Image.LANCZOS)
    return pil

def pil_to_b64(pil, quality=85):
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()

def z_to_b64(z_vector, truncation_psi=0.7, size=None):
    return pil_to_b64(z_to_pil(z_vector, truncation_psi, size))

def z_to_seed(z):
    """Deterministic seed number from a z vector."""
    return int(abs(hash(np.array(z).tobytes())) % 999999)

def slerp(z1, z2, t):
    z1n = z1 / (np.linalg.norm(z1) + 1e-8)
    z2n = z2 / (np.linalg.norm(z2) + 1e-8)
    dot = np.clip(np.dot(z1n, z2n), -1.0, 1.0)
    omega = np.arccos(dot)
    if np.abs(omega) < 1e-6:
        return (1 - t) * z1 + t * z2
    return (np.sin((1-t)*omega)/np.sin(omega))*z1 + (np.sin(t*omega)/np.sin(omega))*z2

# ── Walk state ────────────────────────────────────────────────────────────────
_walk = {}

def _reset_walk():
    global _walk
    if Z_DIM is None: return
    z_start = np.random.randn(Z_DIM)
    _walk = {
        'z':          z_start.copy(),
        'z_start':    z_start.copy(),
        'z_target':   np.random.randn(Z_DIM),
        't':          0.0,
        'truncation': 0.7,
        'step_size':  0.05,
        'pinned_z':   None,
    }

# ── Pending z — one-shot Browser → Explorer handoff ──────────────────────────
_pending_z = None

# ── PCA Grid ──────────────────────────────────────────────────────────────────
_grid_cache = {}

def build_pca_grid(n_samples=512, grid_size=None, thumb_res=None, truncation=0.7):
    """
    Sample n_samples z vectors, compute 2D PCA, build a grid_size x grid_size
    thumbnail grid covering the PCA space. Returns grid metadata + images.
    """
    if grid_size is None: grid_size = args.grid
    if thumb_res is None: thumb_res = args.res

    print(f'[PCA] Sampling {n_samples} vectors...')
    zs = np.random.randn(n_samples, Z_DIM).astype(np.float32)

    # PCA — manual 2-component via SVD (no sklearn needed)
    zs_centered = zs - zs.mean(axis=0)
    _, _, Vt = np.linalg.svd(zs_centered, full_matrices=False)
    pc1 = Vt[0]
    pc2 = Vt[1]

    # Project all samples onto PC1/PC2
    coords = zs_centered @ np.stack([pc1, pc2], axis=1)
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    # Build grid: for each cell, find the z closest to that PCA coordinate
    print(f'[PCA] Building {grid_size}x{grid_size} grid...')
    grid_zs   = []
    grid_imgs = []

    for row in range(grid_size):
        for col in range(grid_size):
            tx = x_min + (col + 0.5) / grid_size * (x_max - x_min)
            ty = y_min + (row + 0.5) / grid_size * (y_max - y_min)
            dists = np.sum((coords - np.array([tx, ty])) ** 2, axis=1)
            idx   = np.argmin(dists)
            z     = zs[idx]
            grid_zs.append(z.tolist())
            img_b64 = z_to_b64(z, truncation, size=thumb_res)
            grid_imgs.append(img_b64)
            print(f'[PCA] Generated {row*grid_size+col+1}/{grid_size*grid_size}', end='\r')

    print(f'\n[PCA] Grid ready.')

    result = {
        'grid_size':  grid_size,
        'thumb_res':  thumb_res,
        'n_samples':  n_samples,
        'x_min': float(x_min), 'x_max': float(x_max),
        'y_min': float(y_min), 'y_max': float(y_max),
        'pc1': pc1.tolist(),
        'pc2': pc2.tolist(),
        'z_mean': zs.mean(axis=0).tolist(),
        'grid_zs':   grid_zs,
        'grid_imgs': grid_imgs,
    }
    _grid_cache['data'] = result
    return result


def z_from_pca_coord(nx, ny, truncation=0.7):
    """
    Given normalized mouse position (nx, ny) in [0,1],
    reconstruct a z vector in PCA space.
    NOTE: This is an approximation — use grid/get_z for exact thumbnail z.
    """
    d = _grid_cache.get('data')
    if d is None:
        return None

    pc1    = np.array(d['pc1'])
    pc2    = np.array(d['pc2'])
    z_mean = np.array(d['z_mean'])
    x_min, x_max = d['x_min'], d['x_max']
    y_min, y_max = d['y_min'], d['y_max']

    px = x_min + nx * (x_max - x_min)
    py = y_min + ny * (y_max - y_min)

    z = z_mean + px * pc1 + py * pc2
    return z


# ── Infinite map tile cache ───────────────────────────────────────────────────
_tile_cache = {}

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')
CORS(app, origins='*')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route('/status')
def status():
    return jsonify({
        'status':     'ok' if G is not None else 'no_model',
        'z_dim':      int(Z_DIM) if Z_DIM else None,
        'resolution': int(RES)   if RES   else None,
        'pkl':        PKL_PATH,
    })


@app.route('/list_models')
def list_models():
    pkls = glob.glob('./**/*.pkl', recursive=True)
    return jsonify({'models': sorted(set(pkls))})


@app.route('/load_model', methods=['POST'])
def load_model_route():
    path = (request.json or {}).get('path', '')
    if not os.path.isfile(path):
        return jsonify({'error': 'not found'}), 400
    try:
        load_model(path)
        return jsonify({'status': 'ok', 'z_dim': int(Z_DIM), 'resolution': int(RES)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ── Walk endpoints ────────────────────────────────────────────────────────────

@app.route('/walk', methods=['POST'])
def walk():
    if G is None: return jsonify({'error': 'no model'}), 400
    step = float((request.json or {}).get('step_size', _walk['step_size']))
    _walk['step_size'] = step
    _walk['t'] += step
    if _walk['t'] >= 1.0:
        _walk['z_start']  = _walk['z_target'].copy()
        _walk['z']        = _walk['z_target'].copy()
        _walk['z_target'] = np.random.randn(Z_DIM)
        _walk['t']        = 0.0
    else:
        _walk['z'] = slerp(_walk['z_start'], _walk['z_target'], _walk['t'])
    img = z_to_b64(_walk['z'], _walk['truncation'])
    if _recording:
        _record_frames.append(img)
    return jsonify({'image': img})


@app.route('/random', methods=['POST'])
def random_jump():
    if G is None: return jsonify({'error': 'no model'}), 400
    z = np.random.randn(Z_DIM)
    _walk['z']        = z.copy()
    _walk['z_start']  = z.copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


@app.route('/pin', methods=['POST'])
def pin():
    _walk['pinned_z'] = _walk['z'].copy()
    return jsonify({'status': 'pinned'})


@app.route('/recall', methods=['POST'])
def recall():
    if _walk.get('pinned_z') is None:
        return jsonify({'error': 'nothing pinned'}), 400
    _walk['z']        = _walk['pinned_z'].copy()
    _walk['z_start']  = _walk['pinned_z'].copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


@app.route('/set_seed', methods=['POST'])
def set_seed():
    if G is None: return jsonify({'error': 'no model'}), 400
    seed = int((request.json or {}).get('seed', 0))
    rng  = np.random.RandomState(seed)
    z    = rng.randn(Z_DIM)
    _walk['z']        = z.copy()
    _walk['z_start']  = z.copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


@app.route('/truncation', methods=['POST'])
def set_truncation():
    if G is None: return jsonify({'error': 'no model'}), 400
    _walk['truncation'] = float((request.json or {}).get('value', 0.7))
    return jsonify({'image': z_to_b64(_walk['z'], _walk['truncation'])})


# ── Pending z — one-shot Browser → Explorer handoff ──────────────────────────

@app.route('/pending_z')
def get_pending_z():
    """
    Returns the z vector sent from Browser to Explorer.
    Clears itself after the first read — one-shot delivery.
    """
    global _pending_z
    if _pending_z is None:
        return jsonify({'image': None})
    z          = _pending_z
    _pending_z = None   # clear after reading
    img  = z_to_b64(z, _walk.get('truncation', 0.7))
    seed = z_to_seed(z)
    return jsonify({'image': img, 'seed': seed})


# ── PCA Grid endpoints ────────────────────────────────────────────────────────

@app.route('/grid/build', methods=['POST'])
def grid_build():
    """Build (or rebuild) the PCA grid."""
    if G is None: return jsonify({'error': 'no model'}), 400
    data       = request.json or {}
    n_samples  = int(data.get('n_samples', 512))
    grid_size  = int(data.get('grid_size', args.grid))
    thumb_res  = int(data.get('thumb_res', args.res))
    truncation = float(data.get('truncation', 0.7))

    result = build_pca_grid(n_samples, grid_size, thumb_res, truncation)

    return jsonify({
        'status':    'ok',
        'grid_size': result['grid_size'],
        'thumb_res': result['thumb_res'],
        'grid_imgs': result['grid_imgs'],
    })


@app.route('/grid/probe', methods=['POST'])
def grid_probe():
    """
    Generate a preview image at a PCA coordinate (mouse position).
    Returns PCA-reconstructed z — for preview only, not for pinning.
    Use /grid/get_z for the exact thumbnail z vector.
    """
    if G is None: return jsonify({'error': 'no model'}), 400
    data       = request.json or {}
    nx         = float(data.get('nx', 0.5))
    ny         = float(data.get('ny', 0.5))
    truncation = float(data.get('truncation', 0.7))

    z = z_from_pca_coord(nx, ny, truncation)
    if z is None:
        return jsonify({'error': 'grid not built yet'}), 400

    img  = z_to_b64(z, truncation, size=512)
    seed = z_to_seed(z)
    return jsonify({'image': img, 'z': z.tolist(), 'seed': seed})


@app.route('/grid/get_z', methods=['POST'])
def grid_get_z():
    """Return the exact z vector of a grid thumbnail by row/col index."""
    data = request.json or {}
    row  = int(data.get('row', 0))
    col  = int(data.get('col', 0))
    d    = _grid_cache.get('data')
    if d is None:
        return jsonify({'error': 'grid not built'}), 400
    gs  = d['grid_size']
    idx = row * gs + col
    if idx >= len(d['grid_zs']):
        return jsonify({'error': 'out of range'}), 400
    z    = np.array(d['grid_zs'][idx])
    seed = z_to_seed(z)
    return jsonify({'z': d['grid_zs'][idx], 'seed': seed})


@app.route('/grid/pin_probe', methods=['POST'])
def grid_pin_probe():
    """
    Pin a z vector from the Browser.
    Sets _pending_z so Walk Explorer gets the exact same image on open.
    """
    global _pending_z
    data = request.json or {}
    z    = data.get('z')
    if z is None:
        return jsonify({'error': 'no z provided'}), 400
    za = np.array(z)
    _walk['z']        = za.copy()
    _walk['z_start']  = za.copy()
    _walk['z_target'] = np.random.randn(Z_DIM)
    _walk['t']        = 0.0
    _walk['pinned_z'] = za.copy()
    _pending_z        = za.copy()   # one-shot for Explorer init
    return jsonify({'status': 'ok'})


@app.route('/grid/pin_infinite', methods=['POST'])
def grid_pin_infinite():
    """
    Pin a z vector as the center point for the Infinite Map.
    Clears tile cache so new tiles are generated from this center.
    """
    global _tile_cache
    data = request.json or {}
    z    = data.get('z')
    if z is None:
        return jsonify({'error': 'no z provided'}), 400
    za = np.array(z)
    _walk['z']               = za.copy()
    _walk['z_start']         = za.copy()
    _walk['z_target']        = np.random.randn(Z_DIM)
    _walk['t']               = 0.0
    _walk['pinned_z']        = za.copy()
    _walk['infinite_center'] = za.copy()
    _tile_cache = {}   # clear so new tiles use this center
    return jsonify({'status': 'ok'})


@app.route('/infinite/center', methods=['GET'])
def infinite_center():
    """Return the pinned center z vector for the Infinite Map, if any."""
    center = _walk.get('infinite_center')
    if center is None:
        return jsonify({'center': None})
    return jsonify({'center': center.tolist()})


@app.route('/infinite/center/clear', methods=['POST'])
def infinite_center_clear():
    """Clear the infinite center after it has been read by the client."""
    _walk.pop('infinite_center', None)
    return jsonify({'status': 'ok'})


# ── Infinite map tiles ────────────────────────────────────────────────────────

@app.route('/infinite/tile', methods=['POST'])
def infinite_tile():
    """
    Generate one tile for the infinite map.
    lx, ly = 2D latent space coordinates.
    Uses two fixed orthogonal basis vectors to map 2D → 512D z vector.
    If center_z is provided, offsets from that point instead of origin.
    """
    if G is None: return jsonify({'error': 'no model'}), 400
    data     = request.json or {}
    lx       = float(data.get('lx', 0))
    ly       = float(data.get('ly', 0))
    trunc    = float(data.get('truncation', 0.7))
    center_z = data.get('center_z', None)

    # Cache key includes center_z hash so different centers don't collide
    center_key = hash(tuple(center_z[:8])) if center_z else 0
    cache_key  = f'{lx:.3f}_{ly:.3f}_{trunc:.2f}_{center_key}'
    if cache_key in _tile_cache:
        return jsonify(_tile_cache[cache_key])

    # Fixed basis vectors — seeded so always the same across sessions
    rng     = np.random.RandomState(42)
    basis_x = rng.randn(Z_DIM).astype(np.float32)
    basis_y = rng.randn(Z_DIM).astype(np.float32)

    # Orthogonalize y against x (Gram-Schmidt)
    basis_x = basis_x / (np.linalg.norm(basis_x) + 1e-8)
    basis_y = basis_y - np.dot(basis_y, basis_x) * basis_x
    basis_y = basis_y / (np.linalg.norm(basis_y) + 1e-8)

    # Scale factor — spreads tiles further apart in latent space
    # StyleGAN2 z vectors typically have norm ~sqrt(Z_DIM) ≈ 22 for 512-dim
    scale = np.sqrt(Z_DIM) * 0.15

    # Reconstruct z — offset from center if provided, otherwise from origin
    if center_z is not None:
        cz = np.array(center_z, dtype=np.float32)
        z  = cz + basis_x * lx * scale + basis_y * ly * scale
    else:
        z = basis_x * lx * scale + basis_y * ly * scale

    img    = z_to_b64(z, trunc, size=128)
    result = {'image': img, 'z': z.tolist()}
    _tile_cache[cache_key] = result
    return jsonify(result)


# ── Recording ─────────────────────────────────────────────────────────────────

@app.route('/record/start', methods=['POST'])
def record_start():
    global _recording, _record_frames
    _recording     = True
    _record_frames = []
    return jsonify({'status': 'recording'})


@app.route('/record/stop', methods=['POST'])
def record_stop():
    global _recording, _record_frames
    _recording = False
    if not _record_frames:
        return jsonify({'error': 'no frames'}), 400

    import subprocess, tempfile

    tmpdir = tempfile.mkdtemp()
    for i, frame_b64 in enumerate(_record_frames):
        img_data = base64.b64decode(frame_b64)
        img      = Image.open(io.BytesIO(img_data))
        img.save(os.path.join(tmpdir, f'frame_{i:05d}.png'))

    output_path = os.path.join(SCRIPT_DIR, f'output_{int(torch.randint(0,9999,(1,)).item())}.mp4')
    subprocess.run([
        r'C:\ffmpeg\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe', '-y',
        '-framerate', '12',
        '-i', os.path.join(tmpdir, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ], check=True)

    _record_frames = []
    return jsonify({'status': 'saved', 'path': output_path})


# ── Serve HTML files ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory(SCRIPT_DIR, 'index.html')

@app.route('/ui')
def serve_explorer():
    return send_from_directory(SCRIPT_DIR, 'latent_explorer.html')

@app.route('/browser')
def serve_browser():
    return send_from_directory(SCRIPT_DIR, 'latent_browser.html')

@app.route('/infinite')
def serve_infinite():
    return send_from_directory(SCRIPT_DIR, 'latent_infinite.html')


# ── Start ─────────────────────────────────────────────────────────────────────

if args.pkl:
    load_model(args.pkl)

if __name__ == '__main__':
    print(f'\n✅  Open http://localhost:{args.port} in your browser\n')
    app.run(port=args.port, debug=False, threaded=True)
