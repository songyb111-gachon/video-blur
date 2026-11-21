#!/usr/bin/env python3
"""app.py - a local web app around blurall.py and vblur.py.

    python app.py            opens http://127.0.0.1:8765 in the browser of this computer
    python app.py --lan      also reachable from phones on the same Wi-Fi (asks for the PIN it prints)

Videos, work files and results are kept in the "jobs" folder next to this file."""
import sys, os, re, json, time, math, shutil, queue, socket, secrets, zipfile, argparse, threading, subprocess, urllib.request, urllib.parse, importlib.util
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); PY = sys.executable
BLURALL, VBLUR = os.path.join(HERE, 'blurall.py'), os.path.join(HERE, 'vblur.py')
MODELS = os.environ.get('VBLUR_MODELS', os.path.expanduser('~/.cache/vblur'))
DATA = os.path.join(HERE, 'jobs')
TONEMAP = "zscale=t=linear:npl=200,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,"
MODEL_URLS = {
    'det_10g.onnx': dict(url='https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip', member='det_10g.onnx'),
    'yolo11s-seg.pt': dict(url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt'),
    'vit_b-encoder-quant.onnx': dict(url='https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-encoder-quant.onnx'),
    'vit_b-decoder-quant.onnx': dict(url='https://github.com/danielgatis/rembg/releases/download/v0.0.0/vit_b-decoder-quant.onnx'),
}
NEED = {'faces': ['det_10g.onnx'], 'people': ['det_10g.onnx', 'yolo11s-seg.pt'], 'pick': ['vit_b-encoder-quant.onnx', 'vit_b-decoder-quant.onnx']}
LABEL = dict(probe='영상 읽기', overview='미리보기 만들기', models='모델 준비', detect='얼굴·사람 찾기', link='프레임 잇기', masks='가릴 영역 만들기',
             render='영상 만들기', check='결과 확인', residual='남은 얼굴 찾기', compare='비교 이미지 만들기', frames='구간 불러오기', embed='화면 분석', track='대상 따라가기')
SUFFIX = dict(faces='faces_blur', people='people_blur', pick='blur')
VIDEO_EXT = ('.mp4', '.mov', '.m4v', '.mkv', '.webm', '.avi', '.mts', '.ts')

JOBS = {}; LOCK = threading.RLock(); Q = queue.Queue(); CUR = dict(job=None, proc=None); SAM = threading.Lock()
AUTH = dict(pin=None, token=secrets.token_hex(16), fails=0)

class Cancel(Exception): pass
class Fail(Exception): pass

# ---- small helpers
def jdir(j, *p): return os.path.join(DATA, j['id'], *p)
def save(j):
    with LOCK:
        d = {k: v for k, v in j.items() if not k.startswith('_')}; tmp = jdir(j, 'job.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f: json.dump(d, f, ensure_ascii=False)
        os.replace(tmp, jdir(j, 'job.json'))
def meta(j): return json.load(open(jdir(j, 'wb', 'meta.json')))
def have(name): p = os.path.join(MODELS, name); return os.path.exists(p) and os.path.getsize(p) > 100000
def even(x): return max(2, int(round(x / 2)) * 2)
def find(lines, pat):
    for l in lines:
        m = re.search(pat, l)
        if m: return m
def nlines(p):
    if not os.path.exists(p): return 0
    with open(p, 'rb') as f: return sum(1 for _ in f)
_TOOLS = {}
def tools():
    """what this computer has: ffmpeg and its encoders/filters, optional packages, model files"""
    if _TOOLS and time.time() - _TOOLS['t'] < 60: return dict(_TOOLS['v'], models={n: have(n) for n in MODEL_URLS})
    r = dict(ffmpeg=False, x265=False, x264=False, zscale=False, ultralytics=importlib.util.find_spec('ultralytics') is not None, models={n: have(n) for n in MODEL_URLS})
    try:
        e = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
        f = subprocess.run(['ffmpeg', '-hide_banner', '-filters'], capture_output=True, text=True).stdout
        r.update(ffmpeg=True, x265='libx265' in e, x264='libx264' in e, zscale=' zscale ' in f)
        subprocess.run(['ffprobe', '-version'], capture_output=True)
    except OSError: r['ffmpeg'] = False
    _TOOLS.update(t=time.time(), v=r); return r

def tool_path():
    """if ffmpeg is not on PATH, use the copy launch.py installed (tools/ next to this file, or winget's folders on Windows)"""
    la = os.environ.get('LOCALAPPDATA', ''); cand = [os.path.join(HERE, 'tools', 'ffmpeg', 'bin')]
    if os.name == 'nt' and la:
        import glob; cand += [os.path.join(la, 'Microsoft', 'WinGet', 'Links')] + sorted({os.path.dirname(p) for p in glob.glob(os.path.join(la, 'Microsoft', 'WinGet', 'Packages', 'Gyan.FFmpeg*', '*', 'bin', 'ffmpeg.exe'))})
    for d in cand:
        if shutil.which('ffmpeg') and shutil.which('ffprobe'): return
        if shutil.which('ffmpeg', path=d) and shutil.which('ffprobe', path=d): os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

def grab(src, idx, size, hdr):
    """decode the listed frame numbers of a video -> {frame: BGR image}, scaled to size and tone-mapped when HDR"""
    idx = sorted(set(idx)); w, h = size; out = {}
    if not idx: return out
    vf = "select='" + '+'.join(f'eq(n\\,{i})' for i in idx) + f"',settb=1/1000,setpts=N*1000,scale={w}:{h}," + (TONEMAP if hdr else '') + 'format=bgr24'
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-map', '0:v:0', '-vf', vf, '-fps_mode', 'passthrough', '-frames:v', str(len(idx)),
                          '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'], stdout=subprocess.PIPE)
    try:
        for i in idx:
            buf = p.stdout.read(w * h * 3)
            if len(buf) < w * h * 3: break
            out[i] = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally: p.stdout.close(); p.kill(); p.wait()
    return out

# ---- tasks: every task is a list of steps; each step runs one command of blurall.py / vblur.py
def begin(j, kind, keys):
    j['_cancel'] = False
    j['task'] = dict(kind=kind, state='queued', error=None, t0=time.time(), log=[], steps=[dict(key=k, label=LABEL[k], state='wait', pct=0) for k in keys])
    save(j); Q.put(j['id'])

def do(j, key, fn):
    s = next(x for x in j['task']['steps'] if x['key'] == key)
    if j.get('_cancel'): raise Cancel()
    s.update(state='run', t0=time.time(), pct=0); s.pop('pct0', None); save(j)
    r = fn(s)
    if j.get('_cancel'): raise Cancel()
    s.update(state='skip' if r == 'skip' else 'done', pct=1, took=round(time.time() - s['t0'], 1)); s.pop('note', None); save(j)
    return r

def sh(j, s, cmd, prog=None, poll=None):
    """run one command, keep its output, and move the step's progress from its printed lines (prog) or from files it writes (poll)"""
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', VBLUR_MODELS=MODELS)
    kw = dict(creationflags=0x00000200) if os.name == 'nt' else dict(start_new_session=True)
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', bufsize=1, cwd=HERE, env=env, **kw)
    CUR['proc'] = p; lines = []; stop = threading.Event()
    def setp(v):
        if v is None: return
        v = max(0.0, min(1.0, float(v))); s.setdefault('pct0', v); s['pct'] = v
    def tick():
        while not stop.wait(1.5):
            try: setp(poll())
            except Exception: pass
    if poll: threading.Thread(target=tick, daemon=True).start()
    for line in p.stdout:
        line = line.rstrip()
        if not line: continue
        lines.append(line); j['task']['log'] = (j['task']['log'] + [line])[-60:]
        if prog:
            try: setp(prog(line))
            except Exception: pass
    rc = p.wait(); stop.set(); CUR['proc'] = None
    if j.get('_cancel'): raise Cancel()
    if rc: raise Fail(' / '.join(lines[-3:]) or f'exit code {rc}')
    return lines

def kill(p):
    if p is None or p.poll() is not None: return
    try:
        if os.name == 'nt': subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)], capture_output=True)
        else: os.killpg(p.pid, 9)
    except Exception: p.kill()

def models(j, kinds):
    def fn(s):
        need = [n for k in kinds for n in NEED[k] if not have(n)]
        if not need: return 'skip'
        os.makedirs(MODELS, exist_ok=True)
        for q, name in enumerate(need):
            spec = MODEL_URLS[name]; dst = os.path.join(MODELS, name); tmp = dst + '.part'
            with urllib.request.urlopen(urllib.request.Request(spec['url'], headers={'User-Agent': 'video-blur-app'}), timeout=60) as r, open(tmp, 'wb') as f:
                total = int(r.headers.get('Content-Length') or 0); got = 0
                while True:
                    if j.get('_cancel'): raise Cancel()
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b); got += len(b); s['note'] = f'{name} {got >> 20}' + (f'/{total >> 20} MB' if total else ' MB')
                    if total: s['pct'] = (q + got / total) / len(need)
            if spec.get('member'):
                with zipfile.ZipFile(tmp) as z, z.open(next(n for n in z.namelist() if n.endswith(spec['member']))) as a, open(dst + '.x', 'wb') as b: shutil.copyfileobj(a, b)
                os.remove(tmp); os.replace(dst + '.x', dst)
            else: os.replace(tmp, dst)
    do(j, 'models', fn)

def t_analyze(j):
    wb = jdir(j, 'wb')
    do(j, 'probe', lambda s: sh(j, s, [PY, BLURALL, 'probe', j['src'], wb]))
    def ov(s):
        sh(j, s, [PY, BLURALL, 'overview', wb, os.path.join(wb, 'ov.jpg')])
        m = meta(j); N = len(m['ticks']); k = max(1, math.ceil(N / 150)); d = jdir(j, 'scrub'); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
        vf = f"select='not(mod(n\\,{k}))',scale='if(gt(iw,ih),360,-2)':'if(gt(iw,ih),-2,360)'," + (TONEMAP if m['hdr'] and m['zscale'] else '') + 'format=yuvj420p'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', j['src'], '-map', '0:v:0', '-vf', vf, '-fps_mode', 'passthrough', '-start_number', '0', '-q:v', '5',
                        os.path.join(d, 's_%04d.jpg')], check=True)
        dur = (m['ticks'][-1] - m['ticks'][0]) * m['tb'][0] / m['tb'][1]
        j['info'] = dict(W=m['W'], H=m['H'], frames=N, duration=round(dur, 2), dt=dur / max(1, N - 1), hdr=bool(m['hdr']), hdr10p=bool(m['hdr10p']), codec=m['codec'], bits=m['bits'],
                         gamma=m['gamma'], scrub=dict(k=k, n=len(os.listdir(d))))
    do(j, 'overview', ov)

def strength(j, o):
    k = min(j['info']['W'], j['info']['H']) / 1080.0; f = dict(low=0.6, mid=1.0, high=1.7).get(o.get('level', 'mid'), 1.0)
    return [] if f == 1.0 else ['--strength', f'{f * (42 if o.get("style", "blur") == "blur" else 36) * k:.1f}']
def enc_opts(j, o):
    crf = dict(high=16, mid=20, small=24).get(o.get('quality', 'high'), 16)
    r = ['--style', 'mosaic' if o.get('style') == 'mosaic' else 'blur', '--crf', str(crf), '--preset', 'fast' if max(j['info']['W'], j['info']['H']) > 2000 else 'medium'] + strength(j, o)
    return r + ([] if o.get('hdr10p', True) else ['--no-hdr10plus'])

def checks(lines_render, lines_check):
    r = {}; m = find(lines_render, r'altered (\d+)'); r['altered'] = int(m.group(1)) if m else None
    m = find(lines_render, r'^(HDR10\+ metadata written into (\d+) of (\d+) frames|skipped: .*)$'); r['hdr10p_copy'] = m.group(1) if m else None
    m = find(lines_check, r'frames: source (\d+), output (\d+)(?:, largest timestamp difference ([\d.]+) ms)?')
    if m: r.update(frames_src=int(m.group(1)), frames_out=int(m.group(2)), ts_diff_ms=float(m.group(3)) if m.group(3) else None)
    m = find(lines_check, r'median (-?[\d.]+) over (\d+) frames'); r['detail'] = max(0.0, float(m.group(1))) if m else None
    m = find(lines_check, r'highest \(share, frame\): \[\((-?[\d.]+)'); r['detail_max'] = float(m.group(1)) if m else None
    m = find(lines_check, r'PSNR vs source: min ([\d.]+) dB, median ([\d.]+) dB'); r['psnr_min'], r['psnr_med'] = (float(m.group(1)), float(m.group(2))) if m else (None, None)
    m = find(lines_check, r'HDR10\+: source has it on (\d+) frames, output on (\d+); identical values on (\d+) of (\d+)')
    if m: r['hdr10p'] = dict(src=int(m.group(1)), out=int(m.group(2)), same=int(m.group(3)), n=int(m.group(4)))
    r['problem'] = bool(find(lines_check, r'!! (MISMATCH|PROBLEM)'))
    return r

def compare(j, out, a, b, hdr):
    """a few frames of the source and of the result at the same moments, as separate small images"""
    i = j['info']; f = 720.0 / max(i['W'], i['H']); size = (even(i['W'] * f), even(i['H'] * f)); d = jdir(j, 'cmp'); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    idx = sorted(set(np.linspace(a, b, 8).astype(int).tolist())); A, B = grab(j['src'], idx, size, hdr), grab(out, idx, size, hdr); r = []
    for q, n in enumerate(idx):
        if n in A and n in B:
            cv2.imwrite(os.path.join(d, f'a_{q:02d}.jpg'), A[n], [cv2.IMWRITE_JPEG_QUALITY, 85]); cv2.imwrite(os.path.join(d, f'b_{q:02d}.jpg'), B[n], [cv2.IMWRITE_JPEG_QUALITY, 85])
            r.append(dict(frame=n, t=round(n * i['dt'], 2), a=f'cmp/a_{q:02d}.jpg', b=f'cmp/b_{q:02d}.jpg'))
    return r

def tiles(j):
    """one small picture per doubtful face candidate, so each can be switched on or off"""
    m = meta(j); T = json.load(open(jdir(j, 'wb', 'tracks.json')))['tracks']; d = jdir(j, 'tiles'); shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    f = min(1.0, 1920.0 / max(m['W'], m['H'])); dw, dh = int(round(m['W'] * f / 2)) * 2, int(round(m['H'] * f / 2)) * 2; kd = min(dw, dh) / 1080.0
    sus = {t: v for t, v in T.items() if v['suspect']}; F = grab(j['src'], [v['first'] for v in sus.values()], (dw, dh), m['hdr'] and m['zscale'])
    lut = (255 * (np.arange(256) / 255.0) ** (1 / m.get('gamma', 1.0))).astype(np.uint8); r = []
    for t, v in sus.items():
        if v['first'] not in F: continue
        im = cv2.LUT(F[v['first']], lut); bx = v['box']; rr = int(max(1.6 * max(bx[2] - bx[0], bx[3] - bx[1]), 60 * kd)); cx, cy = (bx[0] + bx[2]) // 2, (bx[1] + bx[3]) // 2
        x0, y0 = max(0, min(dw - 2 * rr, cx - rr)), max(0, min(dh - 2 * rr, cy - rr)); cr = im[y0:y0 + 2 * rr, x0:x0 + 2 * rr].copy()
        cv2.rectangle(cr, (bx[0] - x0, bx[1] - y0), (bx[2] - x0, bx[3] - y0), (0, 255, 255), 1)
        cv2.imwrite(os.path.join(d, f's_{t}.jpg'), cv2.resize(cr, (220, 220), interpolation=cv2.INTER_CUBIC), [cv2.IMWRITE_JPEG_QUALITY, 85])
        r.append(dict(id=int(t), first=v['first'], last=v['last'], score=v['smax'], auto=bool(v['auto']), t=round(v['first'] * j['info']['dt'], 2)))
    return sorted(r, key=lambda x: x['first'])

def t_blur(j):
    o = j['opts']; mode = o['mode']; wb = jdir(j, 'wb'); N = j['info']['frames']; t0 = time.time()
    a = max(0, min(N - 1, int(o.get('a', 0)))); b = max(a, min(N - 1, int(o.get('b', N - 1)))); g = float(o.get('gamma') or j['info']['gamma'])
    if mode == 'people' and importlib.util.find_spec('ultralytics') is None: raise Fail('"모든 사람" 방식에 필요한 ultralytics 패키지가 없습니다. 앱을 끄고 앱 폴더의 .venv 폴더를 지운 뒤 run.bat(맥·리눅스는 run.sh)을 다시 실행하면 설치합니다.')
    next(x for x in j['task']['steps'] if x['key'] == 'detect')['label'] = '사람 찾기' if mode == 'people' else '얼굴 찾기'
    models(j, [mode]); dets = os.path.join(wb, 'dets.jsonl'); cfgp = os.path.join(wb, 'detect.json'); ran = []
    def detect(s):
        if os.path.exists(cfgp):
            c = json.load(open(cfgp))
            if c['a'] == a and c['b'] == b and abs(c['gamma'] - g) < 1e-6 and (c['people'] or mode == 'faces') and nlines(dets) == b - a + 1: return 'skip'
        ran.append(1)
        sh(j, s, [PY, BLURALL, 'detect', wb, '--a', str(a), '--b', str(b), '--gamma', str(g)] + (['--people'] if mode == 'people' else []), poll=lambda: nlines(dets) / (b - a + 1))
    do(j, 'detect', detect)
    def link(s):
        tr = os.path.join(wb, 'tracks.json')
        if not ran and os.path.exists(tr) and os.path.getmtime(tr) >= os.path.getmtime(dets) and j.get('suspects') is not None: return 'skip'
        sh(j, s, [PY, BLURALL, 'link', wb]); j['suspects'] = tiles(j)
    do(j, 'link', link)
    hide = {} if ran else {int(k): bool(v) for k, v in (o.get('hide') or {}).items() if str(k).isdigit()}; sus = {x['id']: x for x in j['suspects']}   # after a new search the candidates are numbered afresh
    drop = [t for t, h in hide.items() if t in sus and not h and not sus[t]['auto']]; keep = [t for t, h in hide.items() if t in sus and h and sus[t]['auto']]
    L = {}
    def masks(s):
        L['masks'] = sh(j, s, [PY, BLURALL, 'masks', wb, '--mode', mode] + (['--drop', ','.join(map(str, drop))] if drop else []) + (['--keep', ','.join(map(str, keep))] if keep else []),
                        poll=lambda: len(os.listdir(os.path.join(wb, 'alpha'))) / (b - a + 1) if os.path.isdir(os.path.join(wb, 'alpha')) else 0)
    do(j, 'masks', masks)
    os.makedirs(jdir(j, 'out'), exist_ok=True); out = jdir(j, 'out', mode + '.mp4'); tmp = jdir(j, 'out', mode + '.new.mp4')
    do(j, 'render', lambda s: L.update(render=sh(j, s, [PY, BLURALL, 'render', wb, tmp] + enc_opts(j, o), prog=lambda l: (lambda m: int(m.group(1)) / int(m.group(2)) if m else None)(re.match(r'(\d+)/(\d+) frames', l)))))
    os.replace(tmp, out)
    do(j, 'check', lambda s: L.update(check=sh(j, s, [PY, BLURALL, 'check', wb, out])))
    res = os.path.join(wb, 'res.jpg')
    def residual(s):
        if os.path.exists(res): os.remove(res)
        L['res'] = sh(j, s, [PY, BLURALL, 'residual', wb, out, res], prog=lambda l: (lambda m: (int(m.group(1)) - a) / max(1, b - a) if m else None)(re.match(r'frame (\d+)', l)))
    do(j, 'residual', residual)
    m = meta(j); cmp = do(j, 'compare', lambda s: compare(j, out, a, b, m['hdr'] and m['zscale']))
    r = checks(L['render'], L['check']); mm = find(L['masks'], r'median (\d+)%, max (\d+)%'); mr = find(L['res'], r'(\d+) on already-blurred areas.*?(\d+) on untouched areas = (\d+) distinct')
    r.update(mode=mode, a=a, b=b, size=os.path.getsize(out), cover=[int(mm.group(1)), int(mm.group(2))] if mm else None, cmp=cmp, took=round(time.time() - t0),
             residual=dict(blurred=int(mr.group(1)), left=int(mr.group(3)), image=os.path.exists(res)) if mr else None, run=int(time.time()),
             hidden=[x['id'] for x in j['suspects'] if (x['id'] in keep) or (not x['auto'] and x['id'] not in drop)])
    j['result'] = r

def vjob(j, targets=None, crop=None):
    p = j['pick']; d = dict(range=[p['a'], p['b']], step=p['step'], gamma=p['gamma'], targets=targets or [])
    if crop: d['crop'] = [int(v) for v in crop]
    with open(jdir(j, 'wv', 'job.json'), 'w', encoding='utf-8') as f: json.dump(d, f)

def t_prepare(j):
    o = j['opts']; wv = jdir(j, 'wv'); N = j['info']['frames']; a = max(0, min(N - 1, int(o['a']))); b = max(a, min(N - 1, int(o['b']))); st = max(1, min(5, int(o.get('step', 2))))
    models(j, ['pick']); old = j.get('pick') or {}; prev = os.path.join(wv, 'prev')
    def fr(s):
        same = old.get('a') == a and old.get('b') == b and os.path.isdir(prev) and len(os.listdir(prev)) >= b - a + 1
        for d in () if same else ('prev', 'tracks', 'alpha', 'alpha_raw'): shutil.rmtree(os.path.join(wv, d), ignore_errors=True)   # features (emb) stay valid: they are per frame
        if not os.path.exists(os.path.join(wv, 'meta.json')): sh(j, s, [PY, VBLUR, 'probe', j['src'], wv])
        j['pick'] = dict(a=a, b=b, step=st, gamma=float(j['info']['gamma']), ready=False); vjob(j)
        if not same: sh(j, s, [PY, VBLUR, 'frames', wv, str(a), str(b)], poll=lambda: len(os.listdir(prev)) / (b - a + 1) if os.path.isdir(prev) else 0)
        j['pick']['ready'] = True
    do(j, 'frames', fr)

def t_pick(j):
    o = j['opts']; wv = jdir(j, 'wv'); p = j['pick']; a, b = p['a'], p['b']; t0 = time.time(); G = len(range(a, b + 1, p['step'])); L = {}
    tg = [dict(name=f'T{q + 1}', ref=int(t['ref']), pts=[[float(x), float(y)] for x, y in t.get('pts', [])], lab=[int(v) for v in t.get('lab', [])],
               **({'box': [float(v) for v in t['box']]} if t.get('box') else {})) for q, t in enumerate(o['targets'])]
    if not tg: raise Fail('가릴 대상을 하나 이상 골라 주세요.')
    models(j, ['pick'])
    with SAM: vjob(j, tg, o.get('crop'))
    emb = os.path.join(wv, 'emb')
    do(j, 'embed', lambda s: sh(j, s, [PY, VBLUR, 'embed', wv], poll=lambda: max(0, len(os.listdir(emb)) - 1) / G if os.path.isdir(emb) else 0))
    do(j, 'track', lambda s: L.update(track=sh(j, s, [PY, VBLUR, 'track', wv])))
    do(j, 'masks', lambda s: sh(j, s, [PY, VBLUR, 'masks', wv]))
    os.makedirs(jdir(j, 'out'), exist_ok=True); out = jdir(j, 'out', 'pick.mp4'); tmp = jdir(j, 'out', 'pick.new.mp4')
    do(j, 'render', lambda s: L.update(render=sh(j, s, [PY, VBLUR, 'render', wv, tmp] + enc_opts(j, o), prog=lambda l: (lambda m: int(m.group(1)) / int(m.group(2)) if m else None)(re.match(r'(\d+)/(\d+) frames', l)))))
    os.replace(tmp, out)
    do(j, 'check', lambda s: L.update(check=sh(j, s, [PY, VBLUR, 'check', wv, out])))
    m = meta(j); cmp = do(j, 'compare', lambda s: compare(j, out, a, b, m['hdr'] and m['zscale']))
    r = checks(L['render'], L['check']); tr = []
    for l in L['track']:
        q = re.match(r'(T\d+): seed score ([\d.]+) area (\d+)\s+frames (\d+)\.\.(\d+) \((\d+)\)\s+look first at: fallback \[(.*?)\] colour jump \[(.*?)\]', l)
        if q: tr.append(dict(name=q.group(1), score=float(q.group(2)), lo=int(q.group(4)), hi=int(q.group(5)), n=int(q.group(6)),
                             weak=[int(x) for x in q.group(7).split(',') if x.strip()], jump=[int(x) for x in q.group(8).split(',') if x.strip()]))
    r.update(mode='pick', a=a, b=b, size=os.path.getsize(out), cmp=cmp, took=round(time.time() - t0), tracks=tr, run=int(time.time())); j['result'] = r

def seed(j, q):
    """outline of one target on its reference frame, straight away; SAM runs in a short-lived helper process so its memory is given back afterwards"""
    p = j.get('pick') or {}
    if not p.get('ready'): raise Fail('구간을 먼저 불러와 주세요.')
    if (j.get('task') or {}).get('state') in ('queued', 'running'): raise Fail('작업이 진행 중입니다.')
    if [n for n in NEED['pick'] if not have(n)]: raise Fail('SAM 모델 파일이 없습니다.')
    if not q.get('pts') and not q.get('box'): raise Fail('박스를 그리거나 점을 찍어 주세요.')
    with SAM:
        vjob(j, [], q.get('crop'))
        r = subprocess.run([PY, os.path.abspath(__file__), '--seed-worker', jdir(j, 'wv')], input=json.dumps(q), capture_output=True, text=True, encoding='utf-8', errors='replace',
                           env=dict(os.environ, PYTHONIOENCODING='utf-8', VBLUR_MODELS=MODELS))
    try: d = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception: raise Fail((r.stderr or r.stdout or '윤곽을 계산하지 못했습니다.').strip().splitlines()[-1])
    if d.get('error'): raise Fail(d['error'])
    return d

def seed_worker(wv):
    q = json.loads(sys.stdin.read()); sys.path.insert(0, HERE); import vblur as V
    c = V.Ctx(wv); ref = int(q['ref'])
    if ref not in c.grid(): return dict(error='이 프레임은 기준 프레임으로 쓸 수 없습니다.')
    d = os.path.join(wv, 'emb'); sig = json.dumps([c.job.get('crop'), c.gamma])     # the same bookkeeping as "vblur.py embed", so its later run reuses this frame
    if os.path.exists(d) and (not os.path.exists(os.path.join(d, 'sig')) or open(os.path.join(d, 'sig')).read() != sig): shutil.rmtree(d)
    os.makedirs(d, exist_ok=True); open(os.path.join(d, 'sig'), 'w').write(sig); e = os.path.join(d, f'e_{ref:05d}.npy')
    if not os.path.exists(e): np.save(e, V.sam_embed(c, ref).astype(np.float16))
    pts = [(float(x), float(y)) for x, y in q.get('pts') or []]; lab = [int(v) for v in q.get('lab') or [1] * len(pts)]
    m, iou, _ = V.sam_predict(c, c.emb(ref), pts, lab, box=q.get('box')); m = V.clean(c, m)
    cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [[[int(x) + c.x0, int(y) + c.y0] for x, y in cv2.approxPolyDP(k, 1.5, True).reshape(-1, 2)] for k in cnt if cv2.contourArea(k) > 20]
    return dict(polys=polys, score=round(iou, 2), area=int(m.sum()))

TASKS = dict(analyze=t_analyze, blur=t_blur, prepare=t_prepare, pick=t_pick)
def worker():
    while True:
        j = JOBS.get(Q.get())
        if not j or j['task']['state'] != 'queued': continue
        t = j['task']; CUR['job'] = j['id']
        try:
            if j.get('_cancel'): raise Cancel()
            t['state'] = 'running'; save(j); TASKS[t['kind']](j); t['state'] = 'done'
        except Cancel: t['state'] = 'canceled'
        except Exception as e:
            t['state'] = 'error'; t['error'] = str(e) if isinstance(e, (Fail, subprocess.CalledProcessError)) else f'{type(e).__name__}: {e}'
            if not isinstance(e, Fail): import traceback; traceback.print_exc()
        for s in t['steps']:
            if s['state'] == 'run': s['state'] = 'error' if t['state'] == 'error' else 'wait'
        t['t1'] = time.time(); CUR['job'] = None; save(j)

def new_job(name, src=None):
    jid = time.strftime('%m%d-%H%M%S-') + secrets.token_hex(2); j = dict(id=jid, name=os.path.basename(name) or 'video.mp4', created=time.time(), src=src, info=None, opts={}, result=None, pick=None, suspects=None)
    os.makedirs(os.path.join(DATA, jid, 'src'), exist_ok=True)
    with LOCK: JOBS[jid] = j
    return j

def load_jobs():
    os.makedirs(DATA, exist_ok=True)
    for d in sorted(os.listdir(DATA)):
        p = os.path.join(DATA, d, 'job.json')
        try: j = json.load(open(p, encoding='utf-8'))
        except Exception: continue
        t = j.get('task')
        if t and t['state'] in ('queued', 'running'):
            t['state'] = 'canceled'
            for s in t['steps']:
                if s['state'] == 'run': s['state'] = 'wait'
        JOBS[j['id']] = j

def public(j, full=True):
    d = {k: v for k, v in j.items() if not k.startswith('_') and k != 'src'}; d['now'] = time.time()
    if not full:
        t = j.get('task'); d = dict(id=j['id'], name=j['name'], created=j['created'], info=j.get('info'), has_result=bool(j.get('result')),
                                    task=dict(kind=t['kind'], state=t['state'], step=next((s['label'] for s in t['steps'] if s['state'] == 'run'), None)) if t else None)
    return d

# ---- HTTP
class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *a): pass
    def local(self): return self.client_address[0] in ('127.0.0.1', '::1', '::ffff:127.0.0.1')
    def authed(self): return self.local() or (AUTH['pin'] and f"vbauth={AUTH['token']}" in (self.headers.get('Cookie') or ''))
    def out(self, code, body, ctype='application/json', extra=None):
        if isinstance(body, (dict, list)): body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str): body = body.encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', ctype + ('; charset=utf-8' if ctype.startswith('text/') or ctype == 'application/json' else ''))
        self.send_header('Content-Length', str(len(body))); self.send_header('Cache-Control', 'no-store')
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)
    def err(self, msg, code=400): self.out(code, dict(error=msg))
    def body(self):
        n = int(self.headers.get('Content-Length') or 0); return json.loads(self.rfile.read(n) or b'{}') if n else {}
    def file(self, path, name=None):
        if not os.path.isfile(path): return self.err('파일이 없습니다.', 404)
        size = os.path.getsize(path); a, b, code = 0, size - 1, 200; m = re.match(r'bytes=(\d*)-(\d*)$', self.headers.get('Range') or '')
        if m and (m.group(1) or m.group(2)):
            if m.group(1): a = int(m.group(1)); b = min(int(m.group(2)), b) if m.group(2) else b
            else: a = max(0, size - int(m.group(2)))
            if a > b: return self.out(416, b'', 'text/plain', {'Content-Range': f'bytes */{size}'})
            code = 206
        ext = os.path.splitext(path)[1].lower(); ctype = {'.jpg': 'image/jpeg', '.png': 'image/png', '.mp4': 'video/mp4'}.get(ext, 'application/octet-stream')
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(b - a + 1)); self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Cache-Control', 'no-store')
        if code == 206: self.send_header('Content-Range', f'bytes {a}-{b}/{size}')
        if name: self.send_header('Content-Disposition', "attachment; filename*=UTF-8''" + urllib.parse.quote(name))
        self.end_headers()
        try:
            with open(path, 'rb') as f:
                f.seek(a); left = b - a + 1
                while left > 0:
                    buf = f.read(min(1 << 20, left))
                    if not buf: break
                    self.wfile.write(buf); left -= len(buf)
        except (BrokenPipeError, ConnectionError): pass

    def do_GET(self):
        u = urllib.parse.urlsplit(self.path); p = u.path; q = urllib.parse.parse_qs(u.query)
        if p == '/favicon.ico': return self.out(204, b'', 'image/x-icon')
        if p == '/login':
            if AUTH['fails'] > 30: return self.out(403, 'PIN을 너무 많이 틀렸습니다. 앱을 다시 켜 주세요.', 'text/plain')
            if AUTH['pin'] and q.get('pin', [''])[0] == AUTH['pin']: return self.out(303, b'', 'text/plain', {'Location': '/', 'Set-Cookie': f"vbauth={AUTH['token']}; Path=/; HttpOnly; SameSite=Strict"})
            AUTH['fails'] += 1; time.sleep(1); return self.out(303, b'', 'text/plain', {'Location': '/?wrong=1'})
        if not self.authed(): return self.out(200, LOGIN, 'text/html')
        if p == '/': return self.out(200, open(os.path.join(HERE, 'index.html'), 'rb').read(), 'text/html')
        if p == '/api/status':
            with LOCK: jobs = [public(j, False) for j in sorted(JOBS.values(), key=lambda x: -x['created'])]
            return self.out(200, dict(tools=tools(), jobs=jobs, local=self.local(), busy=CUR['job'], cores=os.cpu_count() or 1))
        m = re.match(r'/api/job/([\w-]+)(?:/(.*))?$', p); j = JOBS.get(m.group(1)) if m else None
        if not j: return self.err('없는 주소입니다.', 404)
        rest = m.group(2) or ''
        if rest == '': return self.out(200, public(j))
        if rest == 'download':
            mode = q.get('mode', [(j.get('result') or {}).get('mode', '')])[0]
            if mode not in SUFFIX: return self.err('결과가 없습니다.', 404)
            return self.file(jdir(j, 'out', mode + '.mp4'), f"{os.path.splitext(j['name'])[0]}_{SUFFIX[mode]}.mp4")
        if rest == 'frame':
            f = jdir(j, 'wv', 'prev', f"f_{int(q.get('i', ['0'])[0]):05d}.jpg"); im = cv2.imread(f) if os.path.exists(f) else None
            if im is None: return self.err('이 프레임의 미리보기가 없습니다.', 404)
            w = max(160, min(1600, int(q.get('w', ['960'])[0]))); g = float((j.get('pick') or {}).get('gamma') or 1.0)
            if im.shape[1] > w: im = cv2.resize(im, (w, int(round(im.shape[0] * w / im.shape[1]))), interpolation=cv2.INTER_AREA)
            im = cv2.LUT(im, (255 * (np.arange(256) / 255.0) ** (1 / g)).astype(np.uint8))
            return self.out(200, cv2.imencode('.jpg', im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), 'image/jpeg')
        if rest.startswith('f/'):
            rel = rest[2:]
            if not re.fullmatch(r'(scrub|cmp|tiles|wb)/[\w.]+\.jpg', rel) or '..' in rel: return self.err('없는 파일입니다.', 404)
            return self.file(jdir(j, *rel.split('/')))
        return self.err('없는 주소입니다.', 404)

    def do_POST(self):
        u = urllib.parse.urlsplit(self.path); p = u.path; q = urllib.parse.parse_qs(u.query)
        if not self.authed() or self.headers.get('X-VB') != '1':
            self.close_connection = True; return self.err('권한이 없습니다.', 403)
        try:
            if p == '/api/upload':
                name = q.get('name', ['video.mp4'])[0]; ext = os.path.splitext(name)[1].lower(); n = int(self.headers.get('Content-Length') or 0)
                if ext not in VIDEO_EXT or n <= 0:
                    self.close_connection = True; return self.err('영상 파일(mp4, mov 등)만 올릴 수 있습니다.')
                j = new_job(name); j['src'] = jdir(j, 'src', 'input' + ext); left = n
                with open(j['src'], 'wb') as f:
                    while left > 0:
                        buf = self.rfile.read(min(1 << 20, left))
                        if not buf: break
                        f.write(buf); left -= len(buf)
                if left:
                    JOBS.pop(j['id'], None); shutil.rmtree(jdir(j), ignore_errors=True); self.close_connection = True; return self.err('업로드가 중간에 끊겼습니다.')
                begin(j, 'analyze', ['probe', 'overview']); return self.out(200, dict(id=j['id']))
            d = self.body()
            if p == '/api/open':
                path = os.path.abspath(os.path.expanduser(str(d.get('path', '')).strip().strip('"')))
                if not self.local(): return self.err('파일 경로로 열기는 이 PC에서만 쓸 수 있습니다.', 403)
                if not os.path.isfile(path) or os.path.splitext(path)[1].lower() not in VIDEO_EXT: return self.err('영상 파일을 찾을 수 없습니다: ' + path)
                j = new_job(path, path); begin(j, 'analyze', ['probe', 'overview']); return self.out(200, dict(id=j['id']))
            m = re.match(r'/api/job/([\w-]+)/(\w+)$', p); j = JOBS.get(m.group(1)) if m else None
            if not j: return self.err('없는 주소입니다.', 404)
            act = m.group(2); busy = (j.get('task') or {}).get('state') in ('queued', 'running')
            if act == 'cancel':
                j['_cancel'] = True
                if CUR['job'] == j['id']: kill(CUR['proc'])
                elif busy: j['task']['state'] = 'canceled'; save(j)
                return self.out(200, dict(ok=True))
            if act == 'seed':
                try: return self.out(200, seed(j, d))
                except Fail as e: return self.err(str(e))
            if busy: return self.err('이 영상은 지금 작업 중입니다. 끝나거나 취소한 뒤에 다시 해 주세요.', 409)
            if act == 'delete':
                with LOCK: JOBS.pop(j['id'], None)
                shutil.rmtree(jdir(j), ignore_errors=True); return self.out(200, dict(ok=True))
            if not j.get('info'): return self.err('영상을 아직 읽는 중이거나 읽지 못했습니다.')
            if act == 'blur':
                if d.get('mode') not in ('faces', 'people'): return self.err('방식을 골라 주세요.')
                j['opts'] = d; begin(j, 'blur', ['models', 'detect', 'link', 'masks', 'render', 'check', 'residual', 'compare']); return self.out(200, dict(ok=True))
            if act == 'prepare': j['opts'] = dict(j.get('opts') or {}, **d); begin(j, 'prepare', ['models', 'frames']); return self.out(200, dict(ok=True))
            if act == 'pick':
                if not (j.get('pick') or {}).get('ready'): return self.err('구간을 먼저 불러와 주세요.')
                j['opts'] = dict(d, mode='pick'); begin(j, 'pick', ['models', 'embed', 'track', 'masks', 'render', 'check', 'compare']); return self.out(200, dict(ok=True))
            return self.err('없는 주소입니다.', 404)
        except (BrokenPipeError, ConnectionError): pass
        except Exception as e: return self.err(f'{type(e).__name__}: {e}', 500)

LOGIN = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>video-blur</title>
<body style="font:16px system-ui;margin:0;display:grid;place-items:center;height:100vh;background:#e9edf1;color:#18212c">
<form action=/login style="background:#fff;padding:28px;border-radius:14px;width:260px"><p style="margin:0 0 14px;font-weight:700">PC 화면에 나온 PIN을 입력하세요</p>
<input name=pin inputmode=numeric autofocus style="font:24px system-ui;width:100%;box-sizing:border-box;padding:8px;letter-spacing:.2em">
<button style="margin-top:14px;width:100%;padding:10px;font:16px system-ui;font-weight:700;background:#f2b705;border:0;border-radius:8px">들어가기</button>
<p id=w style="color:#b93a26;margin:12px 0 0;font-size:14px" hidden>PIN이 맞지 않습니다.</p></form><script>if(location.search.includes('wrong'))document.getElementById('w').hidden=false</script>"""

def main():
    global DATA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8765); ap.add_argument('--lan', action='store_true', help='let other devices on the same network connect (PIN required)')
    ap.add_argument('--data', default=DATA, help='folder for videos and work files'); ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--seed-worker', help=argparse.SUPPRESS)
    a = ap.parse_args(); DATA = os.path.abspath(a.data)
    if a.seed_worker:
        try: print(json.dumps(seed_worker(a.seed_worker)))
        except SystemExit as x: print(json.dumps(dict(error=str(x))))
        return
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass
    tool_path(); load_jobs(); threading.Thread(target=worker, daemon=True).start()
    srv = ThreadingHTTPServer(('0.0.0.0' if a.lan else '127.0.0.1', a.port), H); srv.daemon_threads = True; url = f'http://127.0.0.1:{a.port}'
    print(f'video-blur 앱이 켜졌습니다. 이 PC에서: {url}')
    if a.lan:
        AUTH['pin'] = f'{secrets.randbelow(10 ** 6):06d}'
        try: s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(('10.255.255.255', 1)); ip = s.getsockname()[0]; s.close()
        except Exception: ip = '(이 PC의 IP 주소)'
        print(f'같은 와이파이의 다른 기기에서: http://{ip}:{a.port}   PIN: {AUTH["pin"]}')
    t = tools()
    if not t['ffmpeg']: print('!! ffmpeg를 찾을 수 없습니다. ffmpeg를 설치하고 PATH에 넣은 뒤 다시 켜 주세요.')
    print('끄려면 이 창에서 Ctrl+C')
    if not a.no_browser:
        try: import webbrowser; threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        except Exception: pass
    try: srv.serve_forever()
    except KeyboardInterrupt: kill(CUR['proc']); print('\n앱을 껐습니다.')

if __name__ == '__main__': main()
