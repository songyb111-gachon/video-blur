#!/usr/bin/env python3
"""vblur.py - blur or pixelate chosen people/objects in a video: SAM tracking -> soft masks -> ffmpeg re-encode.
All coordinates are full-frame pixels in display orientation (after rotation). Frame numbers are 0-based."""
import sys, os, re, json, glob, shutil, struct, mmap, subprocess, argparse, time
import numpy as np, cv2

MODELS = os.environ.get('VBLUR_MODELS', os.path.expanduser('~/.cache/vblur'))
MEAN = np.array([123.675, 116.28, 103.53], np.float32); STD = np.array([58.395, 57.12, 57.375], np.float32)
TONEMAP = "zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
COL = [(0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (0, 128, 255), (255, 255, 0)]

CM = chr(92) + ','                                             # an escaped comma for ffmpeg filter arguments
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
def odd(x): return max(1, int(x) // 2 * 2 + 1)
def ell(d): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(d), odd(d)))
def box_k(d): return np.ones((odd(d), odd(d)), np.uint8)
def live_pts(src):  # presentation timestamps of the frames a decoder actually outputs (edit lists can hide some)
    rows = [l.split(',') for l in sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts,flags', '-of', 'csv=p=0', src]).split()]
    return sorted(int(r[0]) for r in rows if len(r) > 1 and r[0].lstrip('-').isdigit() and 'D' not in r[1])
def packets(src):  # video packets in decode order: (pts, hidden by an edit list)
    rows = [l.split(',') for l in sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts,flags', '-of', 'csv=p=0', src]).split()]
    return [(int(r[0]), 'D' in r[1]) for r in rows if len(r) > 1 and r[0].lstrip('-').isdigit()]

class Ctx:
    def __init__(s, work):
        s.work = work; s.meta = json.load(open(f'{work}/meta.json')); s.W, s.H = s.meta['W'], s.meta['H']
        s.k = min(s.W, s.H) / 1080.0                      # pixel constants below were tuned at 1080p
        s.job = json.load(open(f'{work}/job.json')) if os.path.exists(f'{work}/job.json') else {}
        s.step = s.job.get('step', 2); s.gamma = s.job.get('gamma', 1.0)
        s.x0, s.y0, x1, y1 = s.job.get('crop', [0, 0, s.W, s.H]); s.cw, s.ch = x1 - s.x0, y1 - s.y0
        s.sc = 1024.0 / max(s.cw, s.ch); s.fd = max(1, int(round(4 * s.k)))
        s.lut = (255 * (np.arange(256) / 255.0) ** (1 / s.gamma)).astype(np.uint8); s._c = {}
    def _get(s, key, make):
        if key not in s._c:
            if len(s._c) > 12: s._c.pop(next(iter(s._c)))
            s._c[key] = make()
        return s._c[key]
    def grid(s): a, b = s.job['range']; return list(range(a, b + 1, s.step))
    def img(s, i, d='prev'):
        def make():
            im = cv2.imread(f'{s.work}/{d}/f_{i:05d}.jpg')
            if im is None: sys.exit(f'no preview for frame {i} in {d}/ - run "frames" for that range first')
            return cv2.LUT(im, s.lut)
        return s._get((d, i), make)
    def emb(s, i): return s._get(('e', i), lambda: np.load(f'{s.work}/emb/e_{i:05d}.npy').astype(np.float32))
    def gray(s, i):
        g = lambda: cv2.cvtColor(s.img(i)[s.y0:s.y0 + s.ch, s.x0:s.x0 + s.cw], cv2.COLOR_BGR2GRAY)
        return s._get(('g', i), lambda: cv2.resize(g(), (s.cw // s.fd, s.ch // s.fd), interpolation=cv2.INTER_AREA))
    def flow(s, a, b): return cv2.calcOpticalFlowFarneback(s.gray(a), s.gray(b), None, 0.5, 4, 21, 4, 5, 1.1, 0) * s.fd

# ---------------------------------------------------------------- probe / previews
def probe(src, work):
    os.makedirs(work, exist_ok=True); src = os.path.abspath(src)
    st = json.loads(sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-of', 'json', src]))['streams'][0]
    fr = json.loads(sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-read_intervals', '%+#1', '-of', 'json', src]))['frames'][0]
    frac = lambda v: (lambda p: float(p[0]) / (float(p[1]) if len(p) > 1 and float(p[1]) else 1.0))(str(v).split('/'))
    rot, md, cll, h10 = 0, None, None, False
    for sd in st.get('side_data_list', []) + fr.get('side_data_list', []):
        t = sd.get('side_data_type', '')
        if 'rotation' in sd: rot = int(round(float(sd['rotation'])))
        if t.startswith('Mastering') and 'red_x' in sd:
            c = lambda q: int(round(frac(sd[q]) * 50000)); l = lambda q: int(round(frac(sd[q]) * 10000))
            md = (f"G({c('green_x')},{c('green_y')})B({c('blue_x')},{c('blue_y')})R({c('red_x')},{c('red_y')})"
                  f"WP({c('white_point_x')},{c('white_point_y')})L({l('max_luminance')},{l('min_luminance')})")
        if t.startswith('Content light'): cll = f"{sd['max_content']},{sd['max_average']}"
        if 'HDR10+' in t: h10 = True
    W, H = st['width'], st['height']
    if abs(rot) % 180 == 90: W, H = H, W
    tb = [int(x) for x in st['time_base'].split('/')]; fps = st.get('avg_frame_rate', '30'); ticks = live_pts(src)
    if len(ticks) < 2: sys.exit('no usable frame timestamps in this file - remux it first: ffmpeg -i in -c copy out.mp4')
    m = dict(src=src, W=W, H=H, rot=rot, codec=st['codec_name'], pix=st['pix_fmt'], bits=10 if re.search(r'p1[026](le|be)?$|^p0?1[06]', st['pix_fmt']) else 8,
             trc=st.get('color_transfer'), prim=st.get('color_primaries'), space=st.get('color_space'), range=st.get('color_range'),
             md=md, cll=cll, tb=tb, fps=fps if frac(fps) > 0 else '30', hdr10p=h10, ticks=ticks)
    json.dump(m, open(f'{work}/meta.json', 'w'))
    t = np.array(m['ticks']) * tb[0] / tb[1]; d = np.diff(t)
    print({k: v for k, v in m.items() if k != 'ticks'})
    print(f"frames {len(t)}  duration {t[-1] - t[0]:.2f}s  frame gap min/median/max {d.min() * 1000:.1f}/{np.median(d) * 1000:.1f}/{d.max() * 1000:.1f} ms")

def frames(c, a, b, src=None, d='prev', pick=None, small=0):
    os.makedirs(f'{c.work}/{d}', exist_ok=True)
    hdr = c.meta['trc'] in ('smpte2084', 'arib-std-b67')
    if hdr and 'zscale' not in sh(['ffmpeg', '-hide_banner', '-filters']): hdr = False; print('note: this ffmpeg has no zscale, so HDR previews are not tone-mapped - compensate with a higher gamma')
    sel = '+'.join(f'eq(n{CM}{i})' for i in pick) if pick else f'between(n{CM}{a}{CM}{b})'
    vf = f"select='{sel}'," + (f"scale='if(gt(iw,ih),{small},-2)':'if(gt(iw,ih),-2,{small})'," if small else '') + (TONEMAP if hdr else '') + 'format=yuv420p'
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src or c.meta['src'], '-map', '0:v:0', '-vf', vf, '-fps_mode', 'passthrough',
                    '-frames:v', str(len(pick) if pick else b - a + 1), '-start_number', str(0 if pick else a), '-q:v', '3', f'{c.work}/{d}/f_%05d.jpg'], check=True)

def overview(c, out, n):
    N = len(c.meta['ticks']); idx = sorted(set(np.linspace(0, N - 1, n).astype(int).tolist())); d = f'{c.work}/ov'
    shutil.rmtree(d, ignore_errors=True); frames(c, 0, N - 1, d='ov', pick=idx, small=640)
    for q, i in reversed(list(enumerate(idx))): os.rename(f'{d}/f_{q:05d}.jpg', f'{d}/f_{i:05d}.jpg')
    mean = float(np.mean([cv2.imread(f, 0).mean() for f in glob.glob(f'{d}/*.jpg')])) / 255
    g = float(np.clip(np.log(max(mean, 0.01)) / np.log(0.42), 1.0, 3.0)); print(f'suggested gamma for job.json: {g:.1f}')
    if 'gamma' not in c.job: c.lut = (255 * (np.arange(256) / 255.0) ** (1 / g)).astype(np.uint8)
    sheet(c, out, idx, cols=6, d='ov')

def sheet(c, out, fr, crop=None, cols=4, grid=False, tracks=False, alpha=False, seeds=False, d='prev'):
    tiles = []; tg = c.job.get('targets', []); sec = lambda i: c.meta['ticks'][i] * c.meta['tb'][0] / c.meta['tb'][1]
    for i in fr:
        im = c.img(i, d).copy(); x0, y0, x1, y1 = crop or (0, 0, im.shape[1], im.shape[0]); fs = max(0.5, (x1 - x0) / 500.0); th = max(1, int(fs * 1.5))
        for q, t in enumerate(tg if tracks else []):
            f = f"{c.work}/tracks/{t['name']}_{i:05d}.png"
            if os.path.exists(f):
                cnt, _ = cv2.findContours(cv2.imread(f, 0), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im, cnt, -1, COL[q % 6], th, offset=(c.x0, c.y0))
        f = f'{c.work}/alpha/a_{i:05d}.png'
        if alpha and os.path.exists(f):
            cnt, _ = cv2.findContours((cv2.imread(f, 0) > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(im, cnt, -1, (0, 0, 255), th)
        for q, t in enumerate(tg if seeds else []):
            if t.get('ref') != i: continue
            if t.get('box'): cv2.rectangle(im, tuple(map(int, t['box'][:2])), tuple(map(int, t['box'][2:])), COL[q % 6], th)
            for p, l in zip(t.get('pts', []), t.get('lab', [1] * 99)): cv2.circle(im, tuple(map(int, p)), 4 * th, (0, 0, 255) if l else (255, 0, 0), -1)
            cv2.putText(im, t['name'], (int(t['box'][0]) if t.get('box') else x0, y0 + int(70 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, COL[q % 6], th)
        t = im[y0:y1, x0:x1].copy()
        if grid:
            g = next(v for v in (50, 100, 200, 500) if (x1 - x0) / v <= 12)
            for x in range((x0 // g + 1) * g, x1, g):
                cv2.line(t, (x - x0, 0), (x - x0, y1 - y0), (255, 255, 0), 1); cv2.putText(t, str(x), (x - x0 + 3, int(60 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (255, 255, 0), th)
            for y in range((y0 // g + 1) * g, y1, g):
                cv2.line(t, (0, y - y0), (x1 - x0, y - y0), (0, 255, 255), 1); cv2.putText(t, str(y), (3, y - y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.6, (0, 255, 255), th)
        cv2.rectangle(t, (0, 0), (int(330 * fs), int(44 * fs)), (0, 0, 0), -1)
        cv2.putText(t, f'#{i} {sec(i):.2f}s', (int(8 * fs), int(32 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th + 1)
        cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (255, 255, 255), th); tiles.append(t)
    rows = [np.hstack(tiles[q:q + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[q:q + cols]))) for q in range(0, len(tiles), cols)]
    im = np.vstack(rows); s = min(1.0, 2000 / im.shape[1])
    cv2.imwrite(out, cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, im.shape[1::-1], f'scale {s:.3f}')

# ---------------------------------------------------------------- SAM (quantised ViT-B, CPU)
_sess = {}
def sess(name):
    if name not in _sess:
        p = f'{MODELS}/vit_b-{name}-quant.onnx'
        if not os.path.exists(p): sys.exit(f'missing {p} - download the two SAM files first')
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2
        _sess[name] = ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
    return _sess[name]

def sam_embed(c, i):
    im = c.img(i)[c.y0:c.y0 + c.ch, c.x0:c.x0 + c.cw]; nw, nh = int(round(c.cw * c.sc)), int(round(c.ch * c.sc))
    r = (cv2.resize(im, (nw, nh), interpolation=cv2.INTER_AREA)[:, :, ::-1].astype(np.float32) - MEAN) / STD
    x = np.zeros((1, 3, 1024, 1024), np.float32); x[0, :, :nh, :nw] = r.transpose(2, 0, 1)
    return sess('encoder').run(None, {'x': x})[0]

def sam_predict(c, emb, pts, lab, box=None, mask_in=None):
    """pts/box in full-frame px; lab 1=on the target, 0=not the target. Returns (mask over the crop, score, low-res logits)."""
    P = [[(x - c.x0) * c.sc, (y - c.y0) * c.sc] for x, y in pts]; L = list(lab)
    if box is not None: P += [[(box[0] - c.x0) * c.sc, (box[1] - c.y0) * c.sc], [(box[2] - c.x0) * c.sc, (box[3] - c.y0) * c.sc]]; L += [2, 3]
    else: P += [[0, 0]]; L += [-1]
    _, iou, low = sess('decoder').run(None, {
        'image_embeddings': emb, 'point_coords': np.array([P], np.float32), 'point_labels': np.array([L], np.float32),
        'mask_input': (mask_in if mask_in is not None else np.zeros((1, 1, 256, 256))).astype(np.float32),
        'has_mask_input': np.array([0.0 if mask_in is None else 1.0], np.float32), 'orig_im_size': np.array([1024, 1024], np.float32)})
    # this export's own full-size mask output assumes a fixed input shape and comes out misplaced, so upsample the 256x256 logits ourselves
    nw, nh = int(round(c.cw * c.sc)), int(round(c.ch * c.sc))
    up = cv2.resize(low[0, 0], (1024, 1024), interpolation=cv2.INTER_LINEAR)[:nh, :nw]
    return cv2.resize(up, (c.cw, c.ch), interpolation=cv2.INTER_LINEAR) > 0, float(iou.ravel()[0]), low

def embed(c):
    d = f'{c.work}/emb'; sig = json.dumps([c.job.get('crop'), c.gamma]); t0 = time.time(); G = c.grid()
    if os.path.exists(d) and (not os.path.exists(f'{d}/sig') or open(f'{d}/sig').read() != sig): shutil.rmtree(d); print('crop or gamma changed - recomputing all features')
    os.makedirs(d, exist_ok=True); open(f'{d}/sig', 'w').write(sig)
    for q, i in enumerate(G):
        p = f'{c.work}/emb/e_{i:05d}.npy'
        if not os.path.exists(p): np.save(p, sam_embed(c, i).astype(np.float16)); print(f'{q + 1}/{len(G)} frame {i} {time.time() - t0:.0f}s', flush=True)

# ---------------------------------------------------------------- tracking
def clean(c, m):
    m = m.astype(np.uint8); n, lab, st, _ = cv2.connectedComponentsWithStats(m, 8); out = np.zeros_like(m)
    if n <= 1: return out.astype(bool)
    big = st[1:, cv2.CC_STAT_AREA].max()
    for q in range(1, n):
        if st[q, cv2.CC_STAT_AREA] >= max(0.08 * big, min(1500 * c.k ** 2, big)): out[lab == q] = 1
    return cv2.morphologyEx(out, cv2.MORPH_CLOSE, ell(15 * c.k)).astype(bool)

def interior(c, m, bands):  # a few points deep inside the mask, one per horizontal band
    ys, xs = np.where(m)
    if len(ys) == 0: return []
    d = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5); y0, y1 = ys.min(), ys.max(); pts = []
    for a, b in [(y0 + (y1 - y0) * j // bands, y0 + (y1 - y0) * (j + 1) // bands) for j in range(bands)]:
        if bands > 1 and b - a < 8: continue
        yy, xx = np.unravel_index(np.argmax(d[a:b + 1]), d[a:b + 1].shape)
        if d[a + yy, xx] >= (6 * c.k if bands > 1 else 1.5): pts.append((int(xx + c.x0), int(yy + a + c.y0)))
    return pts

def shift(c, m, dx, dy): return cv2.warpAffine(m.astype(np.uint8), np.float32([[1, 0, dx], [0, 1, dy]]), (c.cw, c.ch), flags=cv2.INTER_NEAREST) > 0

def step(c, prev, low, a, b, amin):
    """carry mask `prev` from frame a to frame b: move it by its own median optical flow, then let SAM redraw it there"""
    f = c.flow(a, b); sub = lambda m: m[::c.fd, ::c.fd][:f.shape[0], :f.shape[1]]
    sel = sub(cv2.erode(prev.astype(np.uint8), box_k(9 * c.k)) > 0)
    if sel.sum() < 20: sel = sub(prev)
    if sel.sum() < 5: return None
    dx, dy = float(np.median(f[..., 0][sel])), float(np.median(f[..., 1][sel]))
    ys, xs = np.where(prev); bx0, by0, bx1, by1 = xs.min() // c.fd, ys.min() // c.fd, xs.max() // c.fd + 1, ys.max() // c.fd + 1
    tpl = c.gray(a)[by0:by1, bx0:bx1]; e = max(2, int(6 * c.k))
    inside = xs.min() > e and ys.min() > e and xs.max() < c.cw - 1 - e and ys.max() < c.ch - 1 - e
    if inside and min(tpl.shape) >= 4 and tpl.std() > 2:           # flow misses small or fast things: look for the target's patch nearby (only while it is wholly in view - a target cut by the edge matches the wrong thing)
        R = int(max(tpl.shape) * 1.5 + 16); sx, sy = max(0, bx0 - R), max(0, by0 - R)
        res = cv2.matchTemplate(c.gray(b)[sy:by1 + R, sx:bx1 + R], tpl, cv2.TM_CCOEFF_NORMED); best, loc = cv2.minMaxLoc(res)[1::2]
        fx, fy = int(round(bx0 + dx / c.fd)) - sx, int(round(by0 + dy / c.fd)) - sy
        here = res[fy, fx] if 0 <= fy < res.shape[0] and 0 <= fx < res.shape[1] else -1
        if best > 0.6 and best > here + 0.2: dx, dy = float((loc[0] + sx - bx0) * c.fd), float((loc[1] + sy - by0) * c.fd)
    pm = shift(c, prev, dx, dy); ys, xs = np.where(pm)
    if len(ys) < amin: return None                                 # nearly out of view
    pad = 18 * c.k
    box = (c.x0 + max(0, xs.min() - pad), c.y0 + max(0, ys.min() - pad), c.x0 + min(c.cw - 1, xs.max() + pad), c.y0 + min(c.ch - 1, ys.max() + pad))
    pts = interior(c, cv2.erode(pm.astype(np.uint8), box_k(7 * c.k)) > 0, 3) or interior(c, pm, 1)
    if not pts: return None
    lin = cv2.warpAffine(low[0, 0], np.float32([[1, 0, dx * c.sc / 4], [0, 1, dy * c.sc / 4]]), (256, 256), flags=cv2.INTER_LINEAR, borderValue=-12.0)[None, None]
    for mi in (lin, None):                                         # with the previous mask as a hint first, then without
        m, _, lo = sam_predict(c, c.emb(b), pts, [1] * len(pts), box=box, mask_in=mi); m = clean(c, m)
        j = (m & pm).sum() / max((m | pm).sum(), 1); ar = m.sum() / max(pm.sum(), 1)
        if j > 0.55 and 0.6 < ar < 1.5: return m, lo, (dx, dy), True
    return pm, lin, (dx, dy), False                                # SAM disagreed: keep the moved mask for this frame

def tfiles(c, name): return [f for f in glob.glob(f'{c.work}/tracks/{name}_*.png') if re.fullmatch(re.escape(name) + '_[0-9]{5}[.]png', os.path.basename(f))]

def track(c, names, seed_only=False):
    G = c.grid(); os.makedirs(f'{c.work}/tracks', exist_ok=True); N = len(c.meta['ticks'])
    for t in c.job['targets']:
        if names and t['name'] not in names: continue
        for f in tfiles(c, t['name']): os.remove(f)
        if 'keys' in t:                                            # hand-placed boxes, interpolated between key frames - no SAM involved
            K = sorted((int(f), b) for f, b in t['keys'].items()); kf = [f for f, _ in K]; n = 0
            for g in [g for g in G if kf[0] <= g <= kf[-1]]:
                x0, y0, x1, y1 = (float(np.interp(g, kf, [b[q] for _, b in K])) - (c.x0, c.y0)[q % 2] for q in range(4)); m = np.zeros((c.ch, c.cw), np.uint8)
                if t.get('shape') == 'rect': cv2.rectangle(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
                else: cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)), (max(1, int((x1 - x0) / 2)), max(1, int((y1 - y0) / 2))), 0, 0, 360, 255, -1)
                cv2.imwrite(f"{c.work}/tracks/{t['name']}_{g:05d}.png", m); n += 1
            print(f"{t['name']}: {n} hand-placed frames {kf[0]}..{kf[-1]}"); continue
        ref = t['ref']; pts = [tuple(p) for p in t.get('pts', [])]
        if ref not in G: sys.exit(f"{t['name']}: ref {ref} is not a sampled frame (range start + multiples of step)")
        m, iou, low = sam_predict(c, c.emb(ref), pts, t.get('lab', [1] * len(pts)), box=t.get('box')); m = clean(c, m)
        res = {ref: m}; weak = []; jump = []; amin = min(800 * c.k ** 2, max(40, 0.05 * m.sum()))
        col = lambda i, mm: c.img(i)[c.y0:c.y0 + c.ch, c.x0:c.x0 + c.cw][mm].mean(axis=0)
        for d in ([] if seed_only else [c.step, -c.step]):
            pm, pl, a, last = m, low, ref, None
            while True:
                b = a + d
                if not t.get('lo', G[0]) <= b <= t.get('hi', G[-1]) or b not in G: break
                r = step(c, pm, pl, a, b, amin)
                if r is None:                                      # leaving at a frame edge: carry the outline two more steps along its last motion
                    for q in (1, 2):
                        if last is None or not 0 <= a + d * q < N: break
                        pm = shift(c, pm, *last)
                        if pm.sum() < 30: break
                        res[a + d * q] = pm
                    break
                if np.abs(col(b, r[0]) - col(a, pm)).max() > 35: jump.append(b)       # the covered area suddenly looks different
                pm, pl, last, ok = r; res[b] = pm; a = b
                if not ok: weak.append(b)
        for i, mm in res.items(): cv2.imwrite(f"{c.work}/tracks/{t['name']}_{i:05d}.png", mm.astype(np.uint8) * 255)
        print(f"{t['name']}: seed score {iou:.2f} area {int(m.sum())}  frames {min(res)}..{max(res)} ({len(res)})  look first at: fallback {sorted(weak)} colour jump {sorted(jump)}")

# ---------------------------------------------------------------- masks
def edge_reach(m8, reach):  # where an outline stops just short of the bottom edge, run it down to the edge
    cols = m8[-reach:].any(axis=0)
    if cols.any():
        low = np.argmax(m8[::-1], axis=0)
        for x in np.where(cols)[0]: m8[m8.shape[0] - 1 - low[x]:, x] = 1
    return m8

def alpha_of(c, m):
    if not m.any(): return np.zeros(m.shape, np.uint8)
    p = int(45 * c.k); m8 = cv2.copyMakeBorder(m.astype(np.uint8), p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    m8 = np.ascontiguousarray(cv2.morphologyEx(m8, cv2.MORPH_CLOSE, ell(41 * c.k))[p:-p, p:-p])       # bridge narrow gaps between outlines
    for r in range(4): m8 = np.ascontiguousarray(np.rot90(edge_reach(np.ascontiguousarray(np.rot90(m8, r)), int(60 * c.k)), -r))
    n, lab, st, _ = cv2.connectedComponentsWithStats((m8 == 0).astype(np.uint8), 4)                   # fill small enclosed pockets
    for q in range(1, n):
        if st[q, cv2.CC_STAT_AREA] < 30000 * c.k ** 2: m8[lab == q] = 1
    m8 = cv2.dilate(m8, ell(33 * c.k)); p = int(40 * c.k)                                             # safety margin, then feather
    a = cv2.GaussianBlur(cv2.copyMakeBorder(m8.astype(np.float32), p, p, p, p, cv2.BORDER_REPLICATE), (0, 0), 8 * c.k)[p:-p, p:-p]
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)

def masks(c):
    names = [t['name'] for t in c.job['targets']]; S = c.step; N = len(c.meta['ticks'])
    idx = sorted({int(f[-9:-4]) for n in names for f in tfiles(c, n)})
    if not idx: sys.exit('no tracks yet')
    raw, out = f'{c.work}/alpha_raw', f'{c.work}/alpha'
    for d in (raw, out): shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    def load(g):                                                   # {target: its outline on the full frame}
        r = {}
        for n in names:
            f = f'{c.work}/tracks/{n}_{g:05d}.png'
            if os.path.exists(f):
                m = np.zeros((c.H, c.W), bool); m[c.y0:c.y0 + c.ch, c.x0:c.x0 + c.cw] = cv2.imread(f, 0) > 127
                if m.any(): r[n] = m
        return r
    uni = lambda r: np.any(list(r.values()), axis=0) if r else np.zeros((c.H, c.W), bool)
    cen = lambda m: np.array(np.nonzero(m)).mean(axis=1)[::-1]
    mv = lambda m, d: cv2.warpAffine(m.astype(np.uint8), np.float32([[1, 0, d[0]], [0, 1, d[1]]]), (c.W, c.H), flags=cv2.INTER_NEAREST) > 0
    def sdf(m): m8 = m.astype(np.uint8); return cv2.distanceTransform(1 - m8, cv2.DIST_L2, 5) - cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    grow = lambda m: cv2.dilate(m.astype(np.uint8), box_k(61 * c.k)).astype(bool)
    put = lambda i, m: cv2.imwrite(f'{raw}/a_{i:05d}.png', alpha_of(c, m))
    G = list(range(idx[0], idx[-1] + 1, S)); A = load(G[0]); ma = uni(A); put(G[0], ma)
    for a, b in zip(G[:-1], G[1:]):                                # frames between samples
        B = load(b); mb = uni(B); both = ma.any() and mb.any()
        if S > 1:
            if both: sa, sb = sdf(ma), sdf(mb); only = (ma & ~grow(mb)) | (mb & ~grow(ma))
            d = {n: cen(B[n]) - cen(A[n]) for n in A if n in B}
            for i in range(a + 1, b):
                w = (i - a) / S; m = (((1 - w) * sa + w * sb) < 0) | only if both else ma | mb   # blend the two outlines' signed distance fields
                for n, dd in d.items():                            # and for anything that moved, slide both outlines to where it should be now
                    if np.abs(dd).max() > 3: m = m | mv(A[n], w * dd) | mv(B[n], (w - 1) * dd)
                put(i, m)
        put(b, mb); A, ma = B, mb
    get = lambda i: cv2.imread(f'{raw}/a_{i:05d}.png', 0).astype(np.float32) if os.path.exists(f'{raw}/a_{i:05d}.png') else np.zeros((c.H, c.W), np.float32)
    lo, hi = max(0, G[0] - 1), min(N - 1, G[-1] + 1)
    for i in range(lo, hi + 1):                                    # never thinner than the neighbouring frames' average
        p, a, n = get(i - 1), get(i), get(i + 1)
        cv2.imwrite(f'{out}/a_{i:05d}.png', np.maximum(a, 0.25 * p + 0.5 * a + 0.25 * n).astype(np.uint8))
    shutil.rmtree(raw); print(f'alpha for frames {lo}..{hi}')

# ---- HDR10+ : copy the per-frame metadata from the source, without re-encoding
T35 = bytes([0xB5, 0x00, 0x3C, 0x00, 0x01, 0x04])  # ITU-T T.35 header of an HDR10+ message
def unesc(b): return b.replace(bytes([0, 0, 3]), bytes([0, 0]))
def esc(b):
    o = bytearray(); z = 0
    for x in b:
        if z >= 2 and x <= 3: o.append(3); z = 0
        o.append(x); z = z + 1 if x == 0 else 0
    return bytes(o)
def sei_msgs(nal):  # (type, payload) of every message in one SEI NAL unit
    r = unesc(bytes(nal[2:])); i = 0; out = []
    while i + 2 <= len(r) and r[i:] != bytes([0x80]):
        t = s = 0
        while r[i] == 255: t += 255; i += 1
        t += r[i]; i += 1
        while r[i] == 255: s += 255; i += 1
        s += r[i]; i += 1
        out.append((t, r[i:i + s])); i += s
    return out
def src_payloads(src, tmp):  # one HDR10+ payload (or None) per coded picture, decode order
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-map', '0:v:0', '-c', 'copy', '-bsf:v', 'hevc_mp4toannexb', '-f', 'hevc', tmp], check=True)
    fh = open(tmp, 'rb'); buf = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ); sc = bytes([0, 0, 1]); aus = []; pend = None; i = buf.find(sc)
    while i >= 0:
        a = i + 3; i = buf.find(sc, a); t = (buf[a] >> 1) & 63
        if t == 39:  # prefix SEI
            e = i if i >= 0 else len(buf)
            while e > a and buf[e - 1] == 0: e -= 1
            for ty, p in sei_msgs(buf[a:e]):
                if ty == 4 and p[:6] == T35: pend = p
        elif t < 32 and buf[a + 2] & 0x80: aus.append(pend); pend = None  # first slice of a picture
    buf.close(); fh.close(); os.remove(tmp); return aus
def boxes(m, a, b):  # mp4 boxes in a byte range: (type, start, header length, end)
    while a + 8 <= b:
        n, t = struct.unpack('>I4s', m[a:a + 8]); h = 8
        if n == 1: n = struct.unpack('>Q', m[a + 8:a + 16])[0]; h = 16
        elif n == 0: n = b - a
        if n < h: break
        yield t, a, h, a + n; a += n
def inside(m, a, b, names):  # content range of a nested box, or None
    for nm in names:
        r = next(((s + h, e) for t, s, h, e in boxes(m, a, b) if t == nm), None)
        if r is None: return None
        a, b = r
    return a, b
def hdr10p_copy(src, dst, out):
    """write a copy of dst (an HEVC .mp4) that carries the HDR10+ message of every source frame; picture, timing and audio stay byte-identical"""
    ps, po = packets(src), packets(dst); aus = src_payloads(src, out + '.tmp.hevc'); shown = [q for q, (p, hid) in enumerate(ps) if not hid]
    if len({p for p, _ in ps}) != len(ps) or len({p for p, _ in po}) != len(po): return 'skipped: duplicate timestamps'
    if len(aus) == len(shown) and len(aus) != len(ps): aus = dict(zip(shown, aus))
    elif len(aus) != len(ps): return f'skipped: {len(aus)} pictures in the source stream but {len(ps)} packets'
    if len(shown) != len(po): return f'skipped: source shows {len(shown)} frames, output has {len(po)}'
    by_rank = [aus[q] for q in sorted(shown, key=lambda q: ps[q][0])]  # payloads in presentation order
    rank = np.argsort(np.argsort([p for p, _ in po]))  # presentation rank of each output sample
    if not any(by_rank): return 'skipped: the source carries no HDR10+ metadata'
    fh = open(dst, 'rb'); m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ); top = list(boxes(m, 0, len(m)))
    if any(t == b'moof' for t, *_ in top) or not any(t == b'moov' for t, *_ in top): return 'skipped: not a plain (unfragmented) mp4'
    moov = next((s + h, e) for t, s, h, e in top if t == b'moov'); tabs = []; vid = None
    for t, s, h, e in boxes(m, *moov):
        if t != b'trak': continue
        hd = inside(m, s + h, e, [b'mdia', b'hdlr']); tb = {t2: (s2 + h2, e2) for t2, s2, h2, e2 in boxes(m, *inside(m, s + h, e, [b'mdia', b'minf', b'stbl']))}
        key = b'co64' if b'co64' in tb else b'stco'; w = 8 if key == b'co64' else 4; a = tb[key][0]; n = struct.unpack('>I', m[a + 4:a + 8])[0]
        co = np.frombuffer(m[a + 8:a + 8 + n * w], '>u8' if w == 8 else '>u4').astype(np.int64); tabs.append((a + 8, w, co))  # chunk offsets of every track move
        if m[hd[0] + 8:hd[0] + 12] == b'vide' and vid is None:
            p = m.find(b'hvcC', *tb[b'stsd'])
            if p < 0 or (m[p + 4 + 21] & 3) != 3: return 'skipped: output is not HEVC with 4-byte NAL lengths'
            a = tb[b'stsz'][0]; fixed, ns = struct.unpack('>II', m[a + 4:a + 12])
            if fixed: return 'skipped: unexpected fixed sample size'
            sz = np.frombuffer(m[a + 12:a + 12 + 4 * ns], '>u4').astype(np.int64); a2 = tb[b'stsc'][0]; ne = struct.unpack('>I', m[a2 + 4:a2 + 8])[0]
            E = [struct.unpack('>III', m[a2 + 8 + 12 * q:a2 + 20 + 12 * q]) for q in range(ne)]; spc = np.zeros(len(co), np.int64)
            for (fc, cnt, _), nx in zip(E, E[1:] + [(len(co) + 1, 0, 0)]): spc[fc - 1:nx[0] - 1] = cnt
            first = np.cumsum(spc) - spc; off = np.empty(ns, np.int64)
            for q in range(len(co)):
                f0 = int(first[q]); r = sz[f0:f0 + int(spc[q])]; off[f0:f0 + len(r)] = co[q] + np.cumsum(r) - r  # file offset of every video sample
            vid = (a + 12, sz, off)
    if vid is None or len(vid[1]) != len(po): return 'skipped: video track not found or sample count differs'
    stsz_at, sz, off = vid; ins = []; add = np.zeros(len(sz), np.int64)
    for j in range(len(sz)):
        pl = by_rank[int(rank[j])]
        if pl is None: continue
        p = int(off[j]); end = p + int(sz[j])
        while p + 6 <= end and ((m[p + 4] >> 1) & 63) >= 32: p += 4 + struct.unpack('>I', m[p:p + 4])[0]  # walk to the first slice of this picture
        if p + 6 > end: continue
        n = len(pl); nal = bytes([39 << 1, m[p + 5]]) + esc(bytes([4] + [255] * (n // 255) + [n % 255]) + pl + bytes([0x80]))  # a prefix SEI NAL holding just this message
        ins.append((p, struct.pack('>I', len(nal)) + nal)); add[j] = 4 + len(nal)
    pos = np.array([p for p, _ in ins], np.int64); cum = np.concatenate([[0], np.cumsum([len(x) for _, x in ins])]).astype(np.int64)
    moved = lambda x: x + cum[np.searchsorted(pos, x, side='left')]  # where a file offset lands after the insertions before it
    patches = [(stsz_at, (sz + add).astype('>u4').tobytes())]
    for at, w, co in tabs:
        nc = moved(co)
        if w == 4 and nc.max() > 0xFFFFFFFF: return 'skipped: file too large for its 32-bit offset table'
        patches.append((at, nc.astype('>u8' if w == 8 else '>u4').tobytes()))
    for t, s, h, e in top:
        if t == b'mdat':
            n = (e - s) + int(cum[np.searchsorted(pos, e, side='left')] - cum[np.searchsorted(pos, s, side='left')])
            if h == 8 and n > 0xFFFFFFFF: return 'skipped: mdat too large for its 32-bit size field'
            patches.append((s, struct.pack('>I', n)) if h == 8 else (s + 8, struct.pack('>Q', n)))
    cur = 0
    with open(out, 'wb') as w:
        for p, kind, x in sorted([(p, 0, x) for p, x in ins] + [(p, 1, x) for p, x in patches]):
            w.write(m[cur:p]); w.write(x); cur = p + len(x) if kind else p
        w.write(m[cur:])
    m.close(); fh.close(); return f'HDR10+ metadata written into {len(ins)} of {len(sz)} frames'
def hdr10p_frames(src):  # the HDR10+ fields of every frame, as ffprobe decodes them
    js = json.loads(sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_frames', '-show_entries', 'frame=pts:frame_side_data', '-of', 'json', src]), object_pairs_hook=lambda p: p)
    return [next((tuple(sd) for sd in dict(f).get('side_data_list', []) if any(q == 'side_data_type' and 'HDR10+' in str(v) for q, v in sd)), None) for f in dict(js)['frames']]

# ---------------------------------------------------------------- render
def pts_expr(ticks, tb):
    """setpts expression giving frame N its original timestamp: piecewise linear, exact at the knots"""
    t = np.asarray(ticks, np.float64); n = len(t)
    if n < 2: return str(float(t[0])), 0.0
    def knots(tol):
        K = [0]; i = 0
        while i < n - 1:
            ok = lambda j: np.abs(t[i] + (t[j] - t[i]) * (np.arange(i, j + 1) - i) / (j - i) - t[i:j + 1]).max() <= tol
            j, s = i + 1, 1
            while j + s < n and ok(j + s): j += s; s *= 2
            while s >= 1:
                if j + s < n and ok(j + s): j += s
                s //= 2
            K.append(j); i = j
        return K
    for tol_ms in (0, 0.25, 1, 3, 10):                             # exact if the timing is regular enough, else the smallest tolerance that keeps the expression short
        K = knots(max(0.01, tol_ms * tb[1] / (tb[0] * 1000.0)))
        if len(K) <= 150: break
    sl = [float((t[b] - t[a]) / (b - a)) for a, b in zip(K[:-1], K[1:])]
    e = f'{float(t[0])!r}+{sl[0]!r}*N' + ''.join(f'+({sl[q] - sl[q - 1]!r})*max(N-{K[q]},0)' for q in range(1, len(sl)))
    return f'floor({e}+0.5)', tol_ms

def enc_args(c, crf, preset):
    m = dict(c.meta); ten = m['bits'] > 8; pix = 'yuv420p10le' if ten else 'yuv420p'; known = lambda q: m.get(q) and m[q] != 'unknown'
    if m['pix'].startswith('yuvj'): m['range'] = 'pc'
    tags = [x for o, q in (('-color_primaries', 'prim'), ('-color_trc', 'trc'), ('-colorspace', 'space'), ('-color_range', 'range')) if known(q) for x in (o, m[q])]
    if not (ten or m['codec'] == 'hevc'): return pix, tags, ['-c:v', 'libx264', '-preset', preset, '-crf', str(crf), '-pix_fmt', pix]
    xp = ['repeat-headers=1', 'keyint=60', 'min-keyint=30']
    if m['trc'] == 'smpte2084': xp += ['hdr10=1', 'hdr10-opt=1']
    if m.get('md'): xp.append('master-display=' + m['md'])
    if m.get('cll'): xp.append('max-cll=' + m['cll'])
    return pix, tags, ['-c:v', 'libx265', '-preset', preset, '-crf', str(crf), '-x265-params', ':'.join(xp), '-tag:v', 'hvc1', '-pix_fmt', pix]

def soften(p, style, s):
    h, w = p.shape
    if style == 'mosaic':
        b = max(2, int(round(s))); q = cv2.resize(p, (max(1, w // b), max(1, h // b)), interpolation=cv2.INTER_AREA)
        return cv2.resize(q, (w, h), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    q = cv2.GaussianBlur(cv2.resize(p, (w // 4, h // 4), interpolation=cv2.INTER_AREA).astype(np.float32), (0, 0), s / 4.0, borderType=cv2.BORDER_REPLICATE)
    return cv2.resize(q, (w, h), interpolation=cv2.INTER_CUBIC)

def render(c, out, style, strength, crf, preset, hdr10p=True):
    m = c.meta; W, H = c.W, c.H; ten = m['bits'] > 8; dt = np.uint16 if ten else np.uint8; mx = 1023 if ten else 255; tb = m['tb']
    s = strength or (42 if style == 'blur' else 36) * c.k; pix, tags, venc = enc_args(c, crf, preset); fs = W * H * 3 // 2 * dt().itemsize
    expr, tol = pts_expr(m['ticks'], tb); log = open(f'{c.work}/render.log', 'w')
    full = ['-vf', 'scale=out_range=pc'] if m['range'] == 'pc' or m['pix'].startswith('yuvj') else []   # keep full-range sources full range through the pipe
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', m['src'], '-map', '0:v:0'] + full + ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', pix, 'pipe:1'], stdout=subprocess.PIPE, stderr=log)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', pix, '-s', f'{W}x{H}', '-framerate', m['fps']] + tags + ['-i', 'pipe:0', '-i', m['src'],
                            '-map', '0:v:0', '-map', '1:a?', '-map_metadata', '1', '-vf', f"settb={tb[0]}/{tb[1]},setpts='{expr}'", '-fps_mode', 'passthrough',
                            '-enc_time_base', f'{tb[0]}:{tb[1]}'] + venc + tags + (['-video_track_timescale', str(tb[1]), '-movflags', '+faststart'] if tb[0] == 1 and out.lower().endswith(('.mp4', '.mov', '.m4v')) else [])
                           + ['-c:a', 'copy', out], stdin=subprocess.PIPE, stderr=log)
    n = done = 0; t0 = time.time(); q = W * H; cw, ch = W // 2, H // 2
    while True:
        buf = dec.stdout.read(fs)
        if len(buf) < fs: break
        ap = f'{c.work}/alpha/a_{n:05d}.png'; a = cv2.imread(ap, 0) if os.path.exists(ap) else None
        if a is not None and a.max() > 2:
            fr = np.frombuffer(buf, dt); A = a.astype(np.float32) / 255; Ac = cv2.resize(A, (cw, ch), interpolation=cv2.INTER_AREA)
            planes = [(fr[:q].reshape(H, W), A, s), (fr[q:q + cw * ch].reshape(ch, cw), Ac, s / 2), (fr[q + cw * ch:].reshape(ch, cw), Ac, s / 2)]
            buf = b''.join(np.clip(p * (1 - al) + soften(p, style, sg) * al + 0.5, 0, mx).astype(dt).tobytes() for p, al, sg in planes); done += 1
        enc.stdin.write(buf); n += 1
        if n % 100 == 0: print(f'{n}/{len(m["ticks"])} frames {time.time() - t0:.0f}s', flush=True)
    enc.stdin.close(); rc = enc.wait(); dec.wait(); log.close()
    print(f'frames {n} (expected {len(m["ticks"])}), altered {done}, encoder exit {rc}, timestamp tolerance {tol} ms, {time.time() - t0:.0f}s')
    if rc or n != len(m['ticks']): sys.exit('render failed or frame count differs - read ' + log.name)
    if hdr10p and m.get('hdr10p') and '-tag:v' in venc:  # the system x265 cannot write HDR10+, so the messages are copied into the finished file
        try: msg = hdr10p_copy(m['src'], out, out + '.h10.mp4')
        except Exception as e: msg = f'skipped: {e!r}'
        if msg.startswith('HDR10+'): os.replace(out + '.h10.mp4', out)
        elif os.path.exists(out + '.h10.mp4'): os.remove(out + '.h10.mp4')
        print(msg)

def check(c, out):
    m = c.meta; W, H = c.W, c.H; src_t = np.array(m['ticks']) * m['tb'][0] / m['tb'][1]
    otb = [int(x) for x in sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=time_base', '-of', 'csv=p=0', out]).strip().strip(',').split('/')]
    out_t = np.array(live_pts(out)) * otb[0] / otb[1]
    print(f'frames: source {len(src_t)}, output {len(out_t)}' + (f', largest timestamp difference {np.abs(src_t - out_t).max() * 1000:.3f} ms' if len(src_t) == len(out_t) else '  !! MISMATCH'))
    ids = sorted(int(f[-9:-4]) for f in glob.glob(f'{c.work}/alpha/a_*.png')); a, b = ids[0], ids[-1]
    rd = lambda f: subprocess.Popen(['ffmpeg', '-v', 'error', '-i', f, '-map', '0:v:0', '-vf', f"select='between(n{CM}{a}{CM}{b})'", '-fps_mode', 'passthrough', '-frames:v', str(b - a + 1),
                                     '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'], stdout=subprocess.PIPE)
    ps, po = rd(m['src']), rd(out); inside = []; psnr = []
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 3 * c.k)        # fine detail = what a small blur removes
    for i in range(a, b + 1):
        bs, bo = ps.stdout.read(W * H), po.stdout.read(W * H)
        if len(bs) < W * H or len(bo) < W * H: print(f'decode stopped early at frame {i}  !! PROBLEM'); break
        s, o = (np.frombuffer(x, np.uint8).reshape(H, W).astype(np.float32) for x in (bs, bo)); al = cv2.imread(f'{c.work}/alpha/a_{i:05d}.png', 0)
        d = (s - o)[al < 3]; psnr.append(10 * np.log10(255 ** 2 / max(float(np.mean(d * d)), 1e-6)))
        core = cv2.erode((al > 250).astype(np.uint8), box_k(9)) > 0
        if core.sum() > 200: hs, ho = hp(s)[core], hp(o)[core]; inside.append((float((hs * ho).sum() / max(float((hs * hs).sum()), 1e-6)), i))
    for p in (ps, po): p.stdout.close(); p.wait()
    if inside:
        print(f'share of the original fine detail still present inside the treated areas (1 = untouched, near 0 = gone): median {np.median([x[0] for x in inside]):.3f} over {len(inside)} frames')
        print('  highest (share, frame):', [(round(x[0], 3), x[1]) for x in sorted(inside, reverse=True)[:5]])
    print(f'outside the treated areas, PSNR vs source: min {min(psnr):.1f} dB, median {np.median(psnr):.1f} dB')
    if m.get('hdr10p'):
        x, y = hdr10p_frames(m['src']), hdr10p_frames(out)
        print(f'HDR10+: source has it on {sum(v is not None for v in x)} frames, output on {sum(v is not None for v in y)}; identical values on {sum(p == q and p is not None for p, q in zip(x, y))} of {len(x)}')

def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('probe'); p.add_argument('src'); p.add_argument('work')
    p = sub.add_parser('frames'); p.add_argument('work'); p.add_argument('a', type=int); p.add_argument('b', type=int); p.add_argument('--src'); p.add_argument('--dir', default='prev')
    p = sub.add_parser('sheet'); p.add_argument('work'); p.add_argument('out'); p.add_argument('--frames', required=True, help='a:b:step or a,b,c'); p.add_argument('--crop', help='x0,y0,x1,y1')
    p.add_argument('--cols', type=int, default=4); p.add_argument('--dir', default='prev')
    for f in ('grid', 'tracks', 'alpha', 'seeds'): p.add_argument('--' + f, action='store_true')
    for name in ('embed', 'masks'): sub.add_parser(name).add_argument('work')
    p = sub.add_parser('overview'); p.add_argument('work'); p.add_argument('out'); p.add_argument('--n', type=int, default=24)
    p = sub.add_parser('track'); p.add_argument('work'); p.add_argument('names', nargs='*'); p.add_argument('--seed-only', action='store_true')
    p = sub.add_parser('render'); p.add_argument('work'); p.add_argument('out'); p.add_argument('--style', choices=['blur', 'mosaic'], default='blur')
    p.add_argument('--strength', type=float, default=0, help='blur sigma or mosaic block in px (default scales with resolution)'); p.add_argument('--crf', type=int, default=16); p.add_argument('--preset', default='medium'); p.add_argument('--no-hdr10plus', action='store_true')
    p = sub.add_parser('check'); p.add_argument('work'); p.add_argument('out')
    a = ap.parse_args()
    if a.cmd == 'probe': return probe(a.src, a.work)
    c = Ctx(a.work)
    if a.cmd == 'frames': frames(c, a.a, a.b, a.src, a.dir)
    elif a.cmd == 'overview': overview(c, a.out, a.n)
    elif a.cmd == 'sheet':
        fr = [int(x) for x in a.frames.split(',')] if ':' not in a.frames else (lambda v: list(range(v[0], v[1] + 1, v[2] if len(v) > 2 else 1)))([int(x) for x in a.frames.split(':')])
        sheet(c, a.out, fr, [int(x) for x in a.crop.split(',')] if a.crop else None, a.cols, a.grid, a.tracks, a.alpha, a.seeds, a.dir)
    elif a.cmd == 'embed': embed(c)
    elif a.cmd == 'track': track(c, a.names, a.seed_only)
    elif a.cmd == 'masks': masks(c)
    elif a.cmd == 'render': render(c, a.out, a.style, a.strength, a.crf, a.preset, not a.no_hdr10plus)
    elif a.cmd == 'check': check(c, a.out)

if __name__ == '__main__': main()
