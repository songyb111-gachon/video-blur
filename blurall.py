#!/usr/bin/env python3
"""blurall.py - hide EVERY person (whole body) or EVERY face in a video.
detect on each frame -> link faces over time -> soft masks -> ffmpeg re-encode -> copy HDR10+ metadata -> checks.
Frame numbers are 0-based. Detection, boxes and masks live at "detection size" (long side <= 1920) and are scaled up when applied."""
import sys, os, re, json, glob, shutil, struct, mmap, subprocess, argparse, time
import numpy as np, cv2

MODELS = os.environ.get('VBLUR_MODELS', os.path.expanduser('~/.cache/vblur'))
TONEMAP = "zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
CM = chr(92) + ','  # an escaped comma for ffmpeg filter arguments
SEQ = 'settb=1/1000,setpts=N*1000,'  # plain increasing timestamps for frames piped out of ffmpeg
def sh(cmd): return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
def odd(x): return max(1, int(x) // 2 * 2 + 1)
def ell(d): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(d), odd(d)))
def box_k(d): return np.ones((odd(d), odd(d)), np.uint8)
def packets(src):  # video packets in decode order: (pts, hidden by an edit list)
    rows = [l.split(',') for l in sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'packet=pts,flags', '-of', 'csv=p=0', src]).split()]
    return [(int(r[0]), 'D' in r[1]) for r in rows if len(r) > 1 and r[0].lstrip('-').isdigit()]
def live_pts(src): return sorted(p for p, d in packets(src) if not d)
def grid(tiles, cols, out):
    rows = [np.hstack(tiles[q:q + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[q:q + cols]))) for q in range(0, len(tiles), cols)]
    im = np.vstack(rows); s = min(1.0, 2000 / im.shape[1])
    cv2.imwrite(out, cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im, [cv2.IMWRITE_JPEG_QUALITY, 88]); print(out, im.shape[1::-1], f'scale {s:.2f}')
def label(t, text):
    fs = max(0.45, t.shape[1] / 520.0); cv2.rectangle(t, (0, 0), (int(len(text) * 19 * fs) + 8, int(40 * fs)), (0, 0, 0), -1)
    cv2.putText(t, text, (4, int(29 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(1, int(fs * 2)))
    cv2.rectangle(t, (0, 0), (t.shape[1] - 1, t.shape[0] - 1), (255, 255, 255), 1); return t

class Ctx:
    def __init__(s, work):
        s.work = work; s.meta = m = json.load(open(f'{work}/meta.json')); s.W, s.H, s.N = m['W'], m['H'], len(m['ticks'])
        f = min(1.0, 1920.0 / max(s.W, s.H)); s.dw, s.dh = int(round(s.W * f / 2)) * 2, int(round(s.H * f / 2)) * 2  # detection size
        s.sx, s.sy = s.W / s.dw, s.H / s.dh; s.k = min(s.W, s.H) / 1080.0; s.kd = min(s.dw, s.dh) / 1080.0  # pixel constants were tuned at 1080p
        s.fw, s.fh = s.dw // 8, s.dh // 8; s.set_gamma(m.get('gamma', 1.0))
    def set_gamma(s, g): s.gamma = g; s.lut = (255 * (np.arange(256) / 255.0) ** (1 / g)).astype(np.uint8)
    def sec(s, i): return s.meta['ticks'][i] * s.meta['tb'][0] / s.meta['tb'][1]
    def stream(s, src, a=0, b=None, size=None, pick=None):
        """yield (frame number, BGR image) from a video: tone-mapped when HDR, scaled to `size`, brightened by the gamma"""
        b = s.N - 1 if b is None else b; w, h = size or (s.W, s.H); idx = list(pick) if pick else list(range(a, b + 1))
        sel = '+'.join(f'eq(n{CM}{i})' for i in idx) if pick else f'between(n{CM}{a}{CM}{b})'
        vf = f"select='{sel}',{SEQ}scale={w}:{h}," + (TONEMAP if s.meta['hdr'] and s.meta['zscale'] else '') + 'format=bgr24'
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-map', '0:v:0', '-vf', vf, '-fps_mode', 'passthrough', '-frames:v', str(len(idx)),
                              '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'], stdout=subprocess.PIPE)
        try:
            for i in idx:
                buf = p.stdout.read(w * h * 3)
                if len(buf) < w * h * 3: break
                yield i, cv2.LUT(np.frombuffer(buf, np.uint8).reshape(h, w, 3), s.lut)
        finally: p.stdout.close(); p.kill(); p.wait()
    def alpha(s, i, full=True):  # the soft mask of frame i (None if that frame is untouched)
        f = f'{s.work}/alpha/a_{i:05d}.png'
        if not os.path.exists(f): return None
        a = cv2.imread(f, 0); return cv2.resize(a, (s.W, s.H), interpolation=cv2.INTER_LINEAR) if full and a.shape != (s.H, s.W) else a

# ---- probe / overview / sheets
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
    tb = [int(x) for x in st['time_base'].split('/')]; fps = st.get('avg_frame_rate', '30'); ticks = live_pts(src); trc = st.get('color_transfer')
    if len(ticks) < 2: sys.exit('no usable frame timestamps in this file - remux it first: ffmpeg -i in -c copy out.mp4')
    m = dict(src=src, W=W, H=H, rot=rot, codec=st['codec_name'], pix=st['pix_fmt'], bits=10 if re.search(r'p1[026](le|be)?$|^p0?1[06]', st['pix_fmt']) else 8,
             trc=trc, prim=st.get('color_primaries'), space=st.get('color_space'), range=st.get('color_range'), md=md, cll=cll, tb=tb,
             fps=fps if frac(fps) > 0 else '30', hdr=trc in ('smpte2084', 'arib-std-b67'), zscale='zscale' in sh(['ffmpeg', '-hide_banner', '-filters']),
             hdr10p=h10, gamma=1.0, ticks=ticks)
    json.dump(m, open(f'{work}/meta.json', 'w')); t = np.array(ticks) * tb[0] / tb[1]; d = np.diff(t)
    print({k: v for k, v in m.items() if k != 'ticks'})
    print(f"frames {len(t)}  duration {t[-1] - t[0]:.2f}s  frame gap min/median/max {d.min() * 1000:.1f}/{np.median(d) * 1000:.1f}/{d.max() * 1000:.1f} ms")
    if m['hdr'] and not m['zscale']: print('note: this ffmpeg has no zscale, so HDR frames are not tone-mapped for detection - use a higher --gamma')

def overview(c, out, n):
    idx = sorted(set(np.linspace(0, c.N - 1, n).astype(int).tolist())); f = 640.0 / max(c.W, c.H); size = (int(c.W * f / 2) * 2, int(c.H * f / 2) * 2)
    c.set_gamma(1.0); fr = list(c.stream(c.meta['src'], pick=idx, size=size))
    mean = float(np.mean([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean() for _, im in fr])) / 255
    g = round(float(np.clip(np.log(max(mean, 0.01)) / np.log(0.42), 1.0, 3.0)), 1); c.meta['gamma'] = g; json.dump(c.meta, open(f'{c.work}/meta.json', 'w')); c.set_gamma(g)
    print(f'gamma {g} saved (brightening used for detection and for these sheets)')
    grid([label(cv2.LUT(im, c.lut), f'#{i} {c.sec(i):.1f}s') for i, im in fr], 6, out)

def sheet(c, out, fr, crop=None, cols=4, boxes=False, alpha=False, src=None, pair=None):
    x0, y0, x1, y1 = crop or (0, 0, c.W, c.H); th = max(1, int(2 * c.k)); rows = []; f = min(1.0, 2000.0 / (cols * (x1 - x0)))
    T = json.load(open(f'{c.work}/tracks.json'))['frames'] if boxes else {}
    for q, video in enumerate([src or c.meta['src']] + ([pair] if pair else [])):
        tiles = []
        for i, im in c.stream(video, pick=fr):
            for bx in (T.get(str(i), []) if q == 0 else []):
                cv2.rectangle(im, (int(bx[0] * c.sx), int(bx[1] * c.sy)), (int(bx[2] * c.sx), int(bx[3] * c.sy)), (0, 255, 255) if bx[4] == 'det' else (255, 128, 0), th)
            a = c.alpha(i) if alpha and q == 0 else None
            if a is not None:
                cnt, _ = cv2.findContours((a > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(im, cnt, -1, (0, 0, 255), th)
            t = im[y0:y1, x0:x1]; tiles.append(label(cv2.resize(t, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else t.copy(), f'#{i} {c.sec(i):.2f}s' + (' out' if q else '')))
        rows.append(tiles)
    tiles = [t for q in range(0, len(fr), cols) for r in rows for t in (r[q:q + cols] + [np.zeros_like(r[0])] * cols)[:cols]]  # with --pair: a row of originals, then the same frames of the output
    grid(tiles, cols, out)

# ---- detection (every frame; resumable)
_fs = None
def face_detect(bgr, thr=0.3):
    """SCRFD-10G faces in a BGR image -> [[x1, y1, x2, y2, score], ...]"""
    global _fs
    if _fs is None:
        p = f'{MODELS}/det_10g.onnx'
        if not os.path.exists(p): sys.exit(f'missing {p} - see the setup section of the README')
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = os.cpu_count() or 2; so.log_severity_level = 3
        _fs = ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
    h, w = bgr.shape[:2]; H, W = (h + 31) // 32 * 32, (w + 31) // 32 * 32
    pad = np.zeros((H, W, 3), np.uint8); pad[:h, :w] = bgr
    x = ((pad[:, :, ::-1].astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
    out = _fs.run(None, {_fs.get_inputs()[0].name: x}); B, S = [], []
    for q, st in enumerate((8, 16, 32)):
        sc = out[q].ravel(); bb = out[q + 3] * st; cy, cx = np.mgrid[:H // st, :W // st]
        c = np.repeat(np.stack([cx, cy], -1).reshape(-1, 2).astype(np.float32) * st, 2, axis=0); m = sc >= thr
        if m.any():
            c, bb, sc = c[m], bb[m], sc[m]; S.append(sc)
            B.append(np.stack([c[:, 0] - bb[:, 0], c[:, 1] - bb[:, 1], c[:, 0] + bb[:, 2], c[:, 1] + bb[:, 3]], 1))
    if not B: return np.zeros((0, 5), np.float32)
    B = np.concatenate(B); S = np.concatenate(S); o = S.argsort()[::-1]; keep = []; x1, y1, x2, y2 = B.T; ar = (x2 - x1) * (y2 - y1)
    while len(o):  # non-maximum suppression
        i = o[0]; keep.append(i); r = o[1:]
        iw = np.clip(np.minimum(x2[i], x2[r]) - np.maximum(x1[i], x1[r]), 0, None); ih = np.clip(np.minimum(y2[i], y2[r]) - np.maximum(y1[i], y1[r]), 0, None)
        o = r[iw * ih / (ar[i] + ar[r] - iw * ih + 1e-9) <= 0.4]
    return np.concatenate([B[keep], S[keep, None]], 1).astype(np.float32)

def detect(c, people, a, b, gamma):
    d = c.work; b = c.N - 1 if b is None else min(b, c.N - 1)
    if gamma: c.meta['gamma'] = gamma; json.dump(c.meta, open(f'{d}/meta.json', 'w')); c.set_gamma(gamma)
    cfg = dict(people=people, a=a, b=b, gamma=c.gamma); fp, ff, fc = f'{d}/dets.jsonl', f'{d}/flow.npy', f'{d}/detect.json'
    if not (os.path.exists(fc) and json.load(open(fc)) == cfg and os.path.exists(fp) and os.path.exists(ff)):  # different settings: start over
        for f in (fp, ff): os.path.exists(f) and os.remove(f)
        shutil.rmtree(f'{d}/pm', ignore_errors=True); json.dump(cfg, open(fc, 'w'))
        np.lib.format.open_memmap(ff, mode='w+', dtype=np.float16, shape=(max(c.N - 1, 1), c.fh, c.fw, 2)).flush()
    os.makedirs(f'{d}/pm', exist_ok=True); FL = np.load(ff, mmap_mode='r+'); start = a + (sum(1 for _ in open(fp)) if os.path.exists(fp) else 0)
    if start > b: print('detection already complete'); return
    if people:
        from ultralytics import YOLO
        yolo = YOLO(f'{MODELS}/yolo11s-seg.pt'); big = min(1920, (max(c.dw, c.dh) + 31) // 32 * 32)
    prev = None; t0 = time.time()
    with open(fp, 'a') as out:
        for i, im in c.stream(c.meta['src'], max(a, start - 1), b, (c.dw, c.dh)):
            g = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (c.dw // 4, c.dh // 4), interpolation=cv2.INTER_AREA)
            if i >= start:
                if prev is not None: FL[i - 1] = (cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 4, 21, 4, 5, 1.1, 0) * 4.0)[::2, ::2][:c.fh, :c.fw]
                faces = face_detect(im)
                if people:
                    u = np.zeros((c.dh, c.dw), bool)
                    for sz, cf in ((big, 0.15), (640, 0.25)):  # a fine pass for small, far people and a coarse pass for big, close ones (the fine pass leaves holes in those)
                        r = yolo.predict(im, imgsz=sz, conf=cf, classes=[0], retina_masks=True, verbose=False, max_det=300)[0]
                        if r.masks is not None: u |= (r.masks.data.cpu().numpy() > 0.5).any(axis=0)
                    cv2.imwrite(f'{d}/pm/tmp.png', u.astype(np.uint8) * 255); os.replace(f'{d}/pm/tmp.png', f'{d}/pm/m_{i:05d}.png')
                out.write(json.dumps([[round(float(v), 2) for v in r] for r in faces]) + chr(10)); out.flush()
                if (i - start) % 20 == 0: FL.flush(); print(f'frame {i} ({i - a + 1}/{b - a + 1})  faces {len(faces)}  {time.time() - t0:.0f}s', flush=True)
            prev = g
    FL.flush(); print(f'detection done: frames {a}..{b}  {time.time() - t0:.0f}s')

# ---- link face detections into tracks
def link(c):
    d = c.work; cfg = json.load(open(f'{d}/detect.json')); a0, people, kd = cfg['a'], cfg['people'], c.kd; K, E = 10, 8  # bridge gaps up to K frames; extend each track E frames before and after
    D = [np.array(json.loads(l), np.float32).reshape(-1, 5) for l in open(f'{d}/dets.jsonl')]; n = len(D); FL = np.load(f'{d}/flow.npy', mmap_mode='r'); pmc = {}
    if n != cfg['b'] - a0 + 1: print(f'note: detection has only reached frame {a0 + n - 1} of {cfg["b"]} - run detect again to finish')
    def mot(i, bx):  # median motion (frame i -> i+1) around a box
        w, h = bx[2] - bx[0], bx[3] - bx[1]; x1, y1, x2, y2 = bx[0] - 0.25 * w, bx[1] - 0.25 * h, bx[2] + 0.25 * w, bx[3] + 0.25 * h
        r = FL[i][max(0, int(y1 // 8)):max(0, int(y2 // 8)) + 1, max(0, int(x1 // 8)):max(0, int(x2 // 8)) + 1].reshape(-1, 2).astype(np.float32)
        return (0.0, 0.0) if len(r) == 0 else (float(np.median(r[:, 0])), float(np.median(r[:, 1])))
    def on_person(i, bx):  # share of a face box that the person segmentation also calls "person"
        if not people: return 1.0
        if i not in pmc: pmc.clear(); pmc[i] = cv2.imread(f'{d}/pm/m_{i:05d}.png', 0)
        r = pmc[i][max(0, int(bx[1])):max(1, int(bx[3])) + 1, max(0, int(bx[0])):max(1, int(bx[2])) + 1]
        return float((r > 127).mean()) if r.size else 0.0
    cs = lambda bx: ((bx[0] + bx[2]) / 2, (bx[1] + bx[3]) / 2, max(((bx[2] - bx[0]) + (bx[3] - bx[1])) / 2, 10.0 * kd))
    T = []; active = []
    for i in range(a0, a0 + n):
        dets = D[i - a0]; pred = {}
        for ti in active:
            bx = T[ti]['cur']; dx, dy = mot(i - 1, bx); pred[ti] = [bx[0] + dx, bx[1] + dy, bx[2] + dx, bx[3] + dy]
        pairs = []
        for ti, pb in pred.items():
            px, py, ps = cs(pb)
            for j, dt in enumerate(dets):
                qx, qy, qs = cs(dt); dist = float(np.hypot(px - qx, py - qy))
                if dist < 0.7 * max(ps, qs) + 4 * kd * T[ti]['miss'] and 0.5 < qs / ps < 2.0: pairs.append((dist / max(ps, qs), ti, j))
        ut, ud = {}, set()
        for _, ti, j in sorted(pairs):
            if ti not in ut and j not in ud: ut[ti] = j; ud.add(j)
        for ti in list(active):
            t = T[ti]
            if ti in ut:
                dt = dets[ut[ti]]; t['cur'] = [float(v) for v in dt[:4]]; t['box'][i] = t['cur'] + ['det']; t['last'] = i; t['miss'] = 0
                t['ndet'] += 1; t['smax'] = max(t['smax'], float(dt[4])); t['op'].append(on_person(i, dt)); t['w'].append(float(dt[2] - dt[0]))
            else:
                t['miss'] += 1; t['cur'] = pred[ti]; x, y, _ = cs(pred[ti])
                if t['miss'] > K or not (0 <= x < c.dw and 0 <= y < c.dh): active.remove(ti)
                else: t['box'][i] = pred[ti] + ['fill']
        for j, dt in enumerate(dets):
            if j not in ud:
                bx = [float(v) for v in dt[:4]]; active.append(len(T))
                T.append(dict(box={i: bx + ['det']}, cur=bx, first=i, last=i, miss=0, ndet=1, smax=float(dt[4]), op=[on_person(i, dt)], w=[float(dt[2] - dt[0])]))
    frames = {i: [] for i in range(a0, a0 + n)}; info = {}
    for tid, t in enumerate(T):
        if t['ndet'] == 1 and t['smax'] < 0.45: continue  # a single weak hit: most likely not a face
        bxs = {f: bx for f, bx in t['box'].items() if f <= t['last'] + E}; bx = t['box'][t['first']][:4]
        for e in range(1, E + 1):  # walk E frames back before the first detection
            j = t['first'] - e
            if j < a0: break
            dx, dy = mot(j, bx); bx = [bx[0] - dx, bx[1] - dy, bx[2] - dx, bx[3] - dy]; bxs[j] = bx + ['fill']
        op = float(np.median(t['op'])); sus = bool(np.median(t['w']) >= 28 * kd and t['smax'] < 0.6)  # big but weak: could be a lantern or an ornament rather than a face
        info[tid] = dict(first=t['first'], last=t['last'], ndet=t['ndet'], smax=round(t['smax'], 2), suspect=sus, auto=bool(sus and people and op < 0.15),
                         box=[int(round(v)) for v in t['box'][t['first']][:4]])
        for f, v in bxs.items(): frames[f].append([round(float(x), 1) for x in v[:4]] + [v[4], tid])
    json.dump(dict(people=people, frames={str(i): v for i, v in frames.items()}, tracks={str(t): v for t, v in info.items()}), open(f'{d}/tracks.json', 'w'))
    nd = sum(1 for v in frames.values() for bx in v if bx[4] == 'det'); nf = sum(1 for v in frames.values() for bx in v if bx[4] == 'fill')
    print(f'tracks {len(info)} (from {sum(len(x) for x in D)} detections); boxes: detected {nd}, filled in {nf}')
    print('suspect tracks (look at them with "suspects"):', [t for t, v in info.items() if v['suspect']])
    print('  of those, dropped automatically because no person is there:', [t for t, v in info.items() if v['auto']])

def suspects(c, out):
    T = json.load(open(f'{c.work}/tracks.json'))['tracks']; by = {}
    for t, v in T.items():
        if v['suspect']: by.setdefault(v['first'], []).append(t)
    if not by: print('no suspect tracks'); return
    tiles = []
    for f, im in c.stream(c.meta['src'], pick=sorted(by), size=(c.dw, c.dh)):
        for t in by[f]:
            v = T[t]; bx = v['box']; r = int(max(1.6 * max(bx[2] - bx[0], bx[3] - bx[1]), 60 * c.kd)); cx, cy = (bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2
            x0, y0 = max(0, min(c.dw - 2 * r, cx - r)), max(0, min(c.dh - 2 * r, cy - r)); cr = im[y0:y0 + 2 * r, x0:x0 + 2 * r].copy()
            cv2.rectangle(cr, (bx[0] - x0, bx[1] - y0), (bx[2] - x0, bx[3] - y0), (0, 255, 255), 1)
            tiles.append(label(cv2.resize(cr, (240, 240), interpolation=cv2.INTER_CUBIC), f"id{t} #{v['first']}-{v['last']} {v['smax']}" + (' AUTO' if v['auto'] else '')))
    grid(tiles, 8, out)

# ---- masks
def stamp(A, cx, cy, ax, ay, sig):  # a soft ellipse, kept local so thousands of small faces stay cheap
    H, W = A.shape; r = int(3 * sig) + 2; x0, x1, y0, y1 = int(cx - ax) - r, int(cx + ax) + r + 1, int(cy - ay) - r, int(cy + ay) + r + 1
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if xb <= xa or yb <= ya: return
    p = np.zeros((y1 - y0, x1 - x0), np.float32); cv2.ellipse(p, (int(round(cx - x0)), int(round(cy - y0))), (int(round(ax)), int(round(ay))), 0, 0, 360, 1.0, -1)
    p = cv2.GaussianBlur(p, (0, 0), sig); A[ya:yb, xa:xb] = np.maximum(A[ya:yb, xa:xb], p[ya - y0:yb - y0, xa - x0:xb - x0])

def edge_reach(m8, reach):  # where an outline stops just short of the bottom edge, run it down to the edge
    cols = m8[-reach:].any(axis=0)
    if cols.any():
        low = np.argmax(m8[::-1], axis=0)
        for x in np.where(cols)[0]: m8[m8.shape[0] - 1 - low[x]:, x] = 1
    return m8

def alpha_of(k, m):  # outline -> soft mask with a safety margin (values 0..255)
    if not m.any(): return np.zeros(m.shape, np.float32)
    p = int(45 * k); m8 = cv2.copyMakeBorder(m.astype(np.uint8), p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    m8 = np.ascontiguousarray(cv2.morphologyEx(m8, cv2.MORPH_CLOSE, ell(41 * k))[p:-p, p:-p])  # bridge narrow gaps between outlines
    for r in range(4): m8 = np.ascontiguousarray(np.rot90(edge_reach(np.ascontiguousarray(np.rot90(m8, r)), int(60 * k)), -r))
    n, lab, st, _ = cv2.connectedComponentsWithStats((m8 == 0).astype(np.uint8), 4)  # fill small enclosed pockets
    for q in range(1, n):
        if st[q, cv2.CC_STAT_AREA] < 30000 * k ** 2: m8[lab == q] = 1
    m8 = cv2.dilate(m8, ell(33 * k)); p = int(40 * k)  # safety margin, then feather
    return np.clip(cv2.GaussianBlur(cv2.copyMakeBorder(m8.astype(np.float32), p, p, p, p, cv2.BORDER_REPLICATE), (0, 0), 8 * k)[p:-p, p:-p], 0, 1) * 255

def masks(c, mode, drop, keep):
    d = c.work; T = json.load(open(f'{d}/tracks.json')); kd = c.kd; H, W = c.dh, c.dw
    if mode == 'people' and not T['people']: sys.exit('whole-body masks need "detect --people"')
    dead = ({int(t) for t, v in T['tracks'].items() if v['auto']} | set(drop)) - set(keep)
    per = {int(i): [bx for bx in v if bx[5] not in dead] for i, v in T['frames'].items()}; ids = sorted(per); zero = np.zeros((H, W), np.float32)
    def faces(i):
        A = np.zeros((H, W), np.float32)
        for x1, y1, x2, y2, kind, _ in per.get(i, []):
            w, h = x2 - x1, y2 - y1; s = 1.5 if kind == 'det' else 1.9  # filled-in boxes are less certain: cover more
            stamp(A, (x1 + x2) / 2, (y1 + y2) / 2, max(0.5 * w * s, 7 * kd), max(0.5 * h * s, 8 * kd), float(np.clip(0.12 * w, 2.5 * kd, 14 * kd)))
        return A * 255
    cache = {}
    def person(i):
        if i not in per: return None
        if i not in cache:
            for q in [q for q in cache if q < i - 5]: del cache[q]
            f = f'{d}/pm/m_{i:05d}.png'; m = (cv2.imread(f, 0) > 127).astype(np.uint8) if os.path.exists(f) else np.zeros((H, W), np.uint8)
            for x1, y1, x2, y2, kind, _ in per[i]:  # every face belongs to a person: cover the head even where the body was missed
                w, h = x2 - x1, y2 - y1; cv2.ellipse(m, (int((x1 + x2) / 2), int((y1 + y2) / 2 + 0.15 * h)), (int(max(1.1 * w, 8 * kd)), int(max(1.25 * h, 10 * kd))), 0, 0, 360, 1, -1)
            cache[i] = m > 0
        return cache[i]
    def body(i):
        u = np.zeros((H, W), bool)
        for j in range(i - 2, i + 3):  # union over +-2 frames: a person missed on one frame is still covered
            r = person(j)
            if r is not None: u |= r
        return alpha_of(kd, u)
    raw = (lambda i: (faces(i) if mode == 'faces' else body(i)) if i in per else zero)
    shutil.rmtree(f'{d}/alpha', ignore_errors=True); os.makedirs(f'{d}/alpha'); p, a = zero, raw(ids[0]); cover = []; t0 = time.time()
    for i in ids:
        n = raw(i + 1); o = np.clip(np.maximum(a, 0.25 * p + 0.5 * a + 0.25 * n) + 0.5, 0, 255).astype(np.uint8)  # never thinner than the neighbouring frames' average
        cv2.imwrite(f'{d}/alpha/a_{i:05d}.png', o); cover.append(float((o > 127).mean())); p, a = a, n
    print(f'{mode} masks for frames {ids[0]}..{ids[-1]}; dropped tracks {sorted(dead)}; share of the picture covered: median {np.median(cover) * 100:.0f}%, max {max(cover) * 100:.0f}%; {time.time() - t0:.0f}s')

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

# ---- render
def pts_expr(ticks, tb):
    """setpts expression giving frame N its original timestamp: piecewise linear, exact at the knots"""
    t = np.asarray(ticks, np.float64); n = len(t)
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
    for tol_ms in (0, 0.25, 1, 3, 10):  # exact if the timing is regular enough, else the smallest tolerance that keeps the expression short
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

def render(c, out, style, strength, crf, preset, hdr10p):
    m = c.meta; W, H = c.W, c.H; ten = m['bits'] > 8; dt = np.uint16 if ten else np.uint8; mx = 1023 if ten else 255; tb = m['tb']
    s = strength or (42 if style == 'blur' else 36) * c.k; pix, tags, venc = enc_args(c, crf, preset); fs = W * H * 3 // 2 * dt().itemsize
    expr, tol = pts_expr(m['ticks'], tb); log = open(f'{c.work}/render.log', 'w')
    full = ['-vf', 'scale=out_range=pc'] if m['range'] == 'pc' or m['pix'].startswith('yuvj') else []  # keep full-range sources full range through the pipe
    dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', m['src'], '-map', '0:v:0'] + full + ['-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', pix, 'pipe:1'], stdout=subprocess.PIPE, stderr=log)
    enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', pix, '-s', f'{W}x{H}', '-framerate', m['fps']] + tags + ['-i', 'pipe:0', '-i', m['src'],
                            '-map', '0:v:0', '-map', '1:a?', '-map_metadata', '1', '-vf', f"settb={tb[0]}/{tb[1]},setpts='{expr}'", '-fps_mode', 'passthrough',
                            '-enc_time_base', f'{tb[0]}:{tb[1]}'] + venc + tags + (['-video_track_timescale', str(tb[1]), '-movflags', '+faststart'] if tb[0] == 1 and out.lower().endswith(('.mp4', '.mov', '.m4v')) else [])
                           + ['-c:a', 'copy', out], stdin=subprocess.PIPE, stderr=log)
    n = done = 0; t0 = time.time(); q = W * H; cw, ch = W // 2, H // 2
    while True:
        buf = dec.stdout.read(fs)
        if len(buf) < fs: break
        a = c.alpha(n)
        if a is not None and a.max() > 2:
            fr = np.frombuffer(buf, dt); A = a.astype(np.float32) / 255; Ac = cv2.resize(A, (cw, ch), interpolation=cv2.INTER_AREA)
            planes = [(fr[:q].reshape(H, W), A, s), (fr[q:q + cw * ch].reshape(ch, cw), Ac, s / 2), (fr[q + cw * ch:].reshape(ch, cw), Ac, s / 2)]
            buf = b''.join(np.clip(p * (1 - al) + soften(p, style, sg) * al + 0.5, 0, mx).astype(dt).tobytes() for p, al, sg in planes); done += 1
        enc.stdin.write(buf); n += 1
        if n % 100 == 0: print(f'{n}/{c.N} frames {time.time() - t0:.0f}s', flush=True)
    enc.stdin.close(); rc = enc.wait(); dec.wait(); log.close()
    print(f'frames {n} (expected {c.N}), altered {done}, encoder exit {rc}, timestamp tolerance {tol} ms, {time.time() - t0:.0f}s')
    if rc or n != c.N: sys.exit('render failed or frame count differs - read ' + log.name)
    if hdr10p and m['hdr10p'] and '-tag:v' in venc:  # the system x265 cannot write HDR10+, so the messages are copied into the finished file
        try: msg = hdr10p_copy(m['src'], out, out + '.h10.mp4')
        except Exception as e: msg = f'skipped: {e!r}'
        if msg.startswith('HDR10+'): os.replace(out + '.h10.mp4', out)
        elif os.path.exists(out + '.h10.mp4'): os.remove(out + '.h10.mp4')
        print(msg)

# ---- checks
def check(c, out):
    m = c.meta; W, H = c.W, c.H; src_t = np.array(m['ticks']) * m['tb'][0] / m['tb'][1]
    otb = [int(x) for x in sh(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=time_base', '-of', 'csv=p=0', out]).strip().strip(',').split('/')]
    out_t = np.array(live_pts(out)) * otb[0] / otb[1]
    print(f'frames: source {len(src_t)}, output {len(out_t)}' + (f', largest timestamp difference {np.abs(src_t - out_t).max() * 1000:.3f} ms' if len(src_t) == len(out_t) else '  !! MISMATCH'))
    ids = sorted(int(f[-9:-4]) for f in glob.glob(f'{c.work}/alpha/a_*.png')); a, b = ids[0], ids[-1]
    rd = lambda f: subprocess.Popen(['ffmpeg', '-v', 'error', '-i', f, '-map', '0:v:0', '-vf', f"select='between(n{CM}{a}{CM}{b})',{SEQ}format=gray", '-fps_mode', 'passthrough', '-frames:v', str(b - a + 1),
                                     '-f', 'rawvideo', '-pix_fmt', 'gray', 'pipe:1'], stdout=subprocess.PIPE)
    ps, po = rd(m['src']), rd(out); inside_ = []; psnr = []
    hp = lambda x: x - cv2.GaussianBlur(x, (0, 0), 3 * c.k)  # fine detail = what a small blur removes
    for i in range(a, b + 1):
        bs, bo = ps.stdout.read(W * H), po.stdout.read(W * H)
        if len(bs) < W * H or len(bo) < W * H: print(f'decode stopped early at frame {i}  !! PROBLEM'); break
        s, o = (np.frombuffer(x, np.uint8).reshape(H, W).astype(np.float32) for x in (bs, bo)); al = c.alpha(i)
        d = (s - o)[al < 3]; psnr.append(10 * np.log10(255 ** 2 / max(float(np.mean(d * d)), 1e-6)) if d.size else 99.0)
        core = cv2.erode((al > 250).astype(np.uint8), box_k(9)) > 0
        if core.sum() > 200: hs, ho = hp(s)[core], hp(o)[core]; inside_.append((float((hs * ho).sum() / max(float((hs * hs).sum()), 1e-6)), i))
    for p in (ps, po): p.stdout.close(); p.wait()
    if inside_:
        print(f'share of the original fine detail still present inside the treated areas (1 = untouched, near 0 = gone): median {np.median([x[0] for x in inside_]):.3f} over {len(inside_)} frames')
        print('  highest (share, frame):', [(round(x[0], 3), x[1]) for x in sorted(inside_, reverse=True)[:5]])
    print(f'outside the treated areas, PSNR vs source: min {min(psnr):.1f} dB, median {np.median(psnr):.1f} dB')
    if m['hdr10p']:
        x, y = hdr10p_frames(m['src']), hdr10p_frames(out)
        print(f'HDR10+: source has it on {sum(v is not None for v in x)} frames, output on {sum(v is not None for v in y)}; identical values on {sum(p == q and p is not None for p, q in zip(x, y))} of {len(x)}')

def residual(c, video, out, step):
    """run the face detector on the finished video: a face it still finds on an untouched area was not hidden"""
    kd = c.kd; G = []; n_in = 0; t0 = time.time(); a, b = (lambda v: (v['a'], v['b']))(json.load(open(f'{c.work}/detect.json')))
    for (i, o), (_, s) in zip(c.stream(video, a, b, (c.dw, c.dh)), c.stream(c.meta['src'], a, b, (c.dw, c.dh))):
        if (i - a) % step: continue
        al = c.alpha(i, False)
        for x1, y1, x2, y2, sc in face_detect(o):
            r = al[max(0, int(y1)):max(1, int(y2)) + 1, max(0, int(x1)):max(1, int(x2)) + 1] if al is not None else np.zeros(0)
            if r.size and r.mean() / 255 >= 0.6: n_in += 1; continue  # sits on an area that is already blurred
            cx, cy, w = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
            g = next((g for g in G if i - g['last'] <= 12 * step and abs(cx - g['cx']) < 60 * kd and abs(cy - g['cy']) < 60 * kd), None)
            if g: g.update(last=i, cx=cx, cy=cy, n=g['n'] + 1, smax=max(g['smax'], float(sc))); continue
            rr = int(max(2.2 * w, 45 * kd)); x0, y0 = int(max(0, min(c.dw - 2 * rr, cx - rr))), int(max(0, min(c.dh - 2 * rr, cy - rr)))
            cut = lambda im: cv2.resize(im[y0:y0 + 2 * rr, x0:x0 + 2 * rr], (200, 200), interpolation=cv2.INTER_CUBIC)
            G.append(dict(first=i, last=i, cx=cx, cy=cy, n=1, smax=float(sc), tile=np.vstack([cut(s), cut(o)])))
        if (i - a) % (20 * step) == 0: print(f'frame {i}  {time.time() - t0:.0f}s', flush=True)
    print(f'detections left in the output: {n_in} on already-blurred areas (expected: a blurred head still looks head-shaped), {sum(g["n"] for g in G)} on untouched areas = {len(G)} distinct things')
    if G: grid([label(g['tile'], f"#{g['first']}-{g['last']} n{g['n']} {g['smax']:.2f}") for g in G], 10, out); print('each tile: original on top, output below - look at every one')

def main():
    ap = argparse.ArgumentParser(description=__doc__); sub = ap.add_subparsers(dest='cmd', required=True); ints = lambda v: [int(x) for x in v.split(',') if x]
    p = sub.add_parser('probe'); p.add_argument('src'); p.add_argument('work')
    p = sub.add_parser('overview'); p.add_argument('work'); p.add_argument('out'); p.add_argument('--n', type=int, default=24)
    p = sub.add_parser('detect'); p.add_argument('work'); p.add_argument('--people', action='store_true'); p.add_argument('--a', type=int, default=0); p.add_argument('--b', type=int); p.add_argument('--gamma', type=float, default=0)
    for name in ('link',): sub.add_parser(name).add_argument('work')
    p = sub.add_parser('suspects'); p.add_argument('work'); p.add_argument('out')
    p = sub.add_parser('masks'); p.add_argument('work'); p.add_argument('--mode', choices=['faces', 'people'], required=True); p.add_argument('--drop', type=ints, default=[]); p.add_argument('--keep', type=ints, default=[])
    p = sub.add_parser('render'); p.add_argument('work'); p.add_argument('out'); p.add_argument('--style', choices=['blur', 'mosaic'], default='blur')
    p.add_argument('--strength', type=float, default=0, help='blur sigma or mosaic block in px (default scales with resolution)'); p.add_argument('--crf', type=int, default=16); p.add_argument('--preset', default='medium')
    p.add_argument('--no-hdr10plus', action='store_true')
    p = sub.add_parser('check'); p.add_argument('work'); p.add_argument('out')
    p = sub.add_parser('residual'); p.add_argument('work'); p.add_argument('video'); p.add_argument('out'); p.add_argument('--step', type=int, default=2)
    p = sub.add_parser('sheet'); p.add_argument('work'); p.add_argument('out'); p.add_argument('--frames', required=True, help='a:b:step or a,b,c'); p.add_argument('--crop', type=ints, help='x0,y0,x1,y1 in full-size pixels')
    p.add_argument('--cols', type=int, default=4); p.add_argument('--src'); p.add_argument('--pair'); p.add_argument('--boxes', action='store_true'); p.add_argument('--alpha', action='store_true')
    a = ap.parse_args()
    if a.cmd == 'probe': return probe(a.src, a.work)
    c = Ctx(a.work)
    if a.cmd == 'overview': overview(c, a.out, a.n)
    elif a.cmd == 'detect': detect(c, a.people, a.a, a.b, a.gamma)
    elif a.cmd == 'link': link(c)
    elif a.cmd == 'suspects': suspects(c, a.out)
    elif a.cmd == 'masks': masks(c, a.mode, a.drop, a.keep)
    elif a.cmd == 'render': render(c, a.out, a.style, a.strength, a.crf, a.preset, not a.no_hdr10plus)
    elif a.cmd == 'check': check(c, a.out)
    elif a.cmd == 'residual': residual(c, a.video, a.out, a.step)
    elif a.cmd == 'sheet':
        fr = ints(a.frames) if ':' not in a.frames else (lambda v: list(range(v[0], v[1] + 1, v[2] if len(v) > 2 else 1)))([int(x) for x in a.frames.split(':')])
        sheet(c, a.out, sorted(set(fr)), a.crop, a.cols, a.boxes, a.alpha, a.src, a.pair)

if __name__ == '__main__': main()
