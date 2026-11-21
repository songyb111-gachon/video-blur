#!/usr/bin/env python3
"""imgblur.py - hide every face, every person, or chosen targets in a photo. It uses the same models as blurall.py and vblur.py.

    python imgblur.py blur IN OUT --mode faces|people [--style blur|mosaic] [--strength px]

The result keeps the picture's size, orientation and colour profile. Other metadata is NOT copied (EXIF: place, time, camera, and the
small preview picture stored inside many photos, which would still show the original).
The other commands work on a work folder W and are what app.py runs step by step: info, detect, seed, render."""
import sys, os, json, argparse, shutil, tempfile
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import blurall as B

HI_SIDE = 2560 if os.environ.get('VBLUR_LOWMEM') else 3840      # the near-full-size face search needs about 2 GB of memory at 3840
MAX_PIXELS = 120_000_000      # the whole picture is held in memory several times over while it is blurred
def even(x): return max(2, int(round(x / 2)) * 2)
def lut(g): return (255 * (np.arange(256) / 255.0) ** (1 / g)).astype(np.uint8)
def fit(im, side):
    f = side / max(im.shape[:2]); return cv2.resize(im, (even(im.shape[1] * f), even(im.shape[0] * f)), interpolation=cv2.INTER_AREA) if f < 1 else im

def load(path):
    """-> (BGR picture turned upright, its transparency channel or None, its colour profile or None)"""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    try: im = Image.open(path); n = im.size[0] * im.size[1]; im.seek(0); im.load()
    except Image.DecompressionBombError: sys.exit('사진이 너무 큽니다. %d만 화소까지 다룰 수 있습니다.' % (MAX_PIXELS // 10000))
    except Exception as e: sys.exit('이 사진을 읽지 못했습니다. 파일이 손상됐거나 지원하지 않는 형식입니다. (%s)' % type(e).__name__)
    if n > MAX_PIXELS: sys.exit('사진이 너무 큽니다. %d만 화소까지 다룰 수 있습니다.' % (MAX_PIXELS // 10000))
    im = ImageOps.exif_transpose(im)
    if im.mode == 'P': im = im.convert('RGBA' if 'transparency' in im.info else 'RGB')
    if im.mode in ('I;16', 'I;16B', 'I;16L', 'I', 'F'):      # 16-bit grey: Pillow would clip it, so scale it down by hand
        x = np.asarray(im, np.float64); im = Image.fromarray(np.clip(x / (257.0 if x.max() > 255 else 1.0), 0, 255).astype(np.uint8))
    icc = im.info.get('icc_profile') if im.mode in ('RGB', 'RGBA') else None
    a = np.array(im.getchannel('A')) if im.mode in ('RGBA', 'LA') else None
    return np.ascontiguousarray(np.array(im.convert('RGB'))[:, :, ::-1]), a, icc

def save(path, bgr, a=None, icc=None):
    from PIL import Image
    ext = os.path.splitext(path)[1].lower(); im = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
    if a is not None and ext in ('.png', '.webp'): im.putalpha(Image.fromarray(a))
    kw = dict(quality=95, subsampling=0, optimize=True) if ext in ('.jpg', '.jpeg') else dict(quality=95) if ext == '.webp' else {}
    if icc: kw['icc_profile'] = icc
    im.save(path, **kw)

def meta(W): return json.load(open(os.path.join(W, 'img.json'), encoding='utf-8'))

def cmd_info(src, W):
    os.makedirs(W, exist_ok=True); bgr, a, icc = load(src); h, w = bgr.shape[:2]; view = fit(bgr, 1600)
    mean = float(cv2.cvtColor(fit(view, 640), cv2.COLOR_BGR2GRAY).mean()) / 255; g = round(float(np.clip(np.log(max(mean, 0.01)) / np.log(0.42), 1.0, 3.0)), 1)
    cv2.imwrite(os.path.join(W, 'view.jpg'), view, [cv2.IMWRITE_JPEG_QUALITY, 88]); cv2.imwrite(os.path.join(W, 'thumb.jpg'), fit(bgr, 360), [cv2.IMWRITE_JPEG_QUALITY, 80])
    m = dict(src=os.path.abspath(src), W=w, H=h, gamma=g, alpha=a is not None, icc=bool(icc))
    with open(os.path.join(W, 'img.json'), 'w', encoding='utf-8') as f: json.dump(m, f)
    print(json.dumps(m))

def det_faces(bgr, g):
    """faces at one or two scales (big photos are also searched near full size, for small faces) -> [[x1, y1, x2, y2, score]] in full-size pixels"""
    h, w = bgr.shape[:2]; out = []
    for side in sorted({min(max(h, w), 1920), min(max(h, w), HI_SIDE) if max(h, w) > 2400 else min(max(h, w), 1920)}):
        im = fit(bgr, side); sx, sy = w / im.shape[1], h / im.shape[0]
        out += [[x1 * sx, y1 * sy, x2 * sx, y2 * sy, s] for x1, y1, x2, y2, s in B.face_detect(cv2.LUT(im, lut(g)))]
    out.sort(key=lambda r: -r[4]); keep = []
    for r in out:      # the same face found at both scales: keep the stronger hit
        for q in keep:
            iw = min(r[2], q[2]) - max(r[0], q[0]); ih = min(r[3], q[3]) - max(r[1], q[1])
            if iw > 0 and ih > 0 and iw * ih / ((r[2] - r[0]) * (r[3] - r[1]) + (q[2] - q[0]) * (q[3] - q[1]) - iw * ih) > 0.4: break
        else: keep.append(r)
    return keep

def det_size(w, h): f = min(1.0, 1920.0 / max(w, h)); return even(w * f), even(h * f)

def cmd_detect(W, people, gamma):
    m = meta(W); g = gamma or m['gamma']; bgr, _, _ = load(m['src']); h, w = bgr.shape[:2]; k = min(w, h) / 1080.0; dw, dh = det_size(w, h); person = None
    if people:
        from ultralytics import YOLO
        im = cv2.LUT(cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA) if (dw, dh) != (w, h) else bgr, lut(g)); person = np.zeros((dh, dw), bool)
        yolo = YOLO(f'{B.MODELS}/yolo11s-seg.pt'); big = min(1920, (max(dw, dh) + 31) // 32 * 32)
        for sz, cf in ((big, 0.15), (640, 0.25)):      # as in blurall.py: a fine pass for small, far people and a coarse pass for big, close ones
            r = yolo.predict(im, imgsz=sz, conf=cf, classes=[0], retina_masks=True, verbose=False, max_det=300)[0]
            if r.masks is not None: person |= (cv2.resize((r.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8), (dw, dh), interpolation=cv2.INTER_NEAREST) > 0)
        cv2.imwrite(os.path.join(W, 'person.png'), person.astype(np.uint8) * 255)
    faces = []; d = os.path.join(W, 'tiles'); shutil.rmtree(d, ignore_errors=True); os.makedirs(d); bright = lut(g)
    for q, (x1, y1, x2, y2, s) in enumerate(det_faces(bgr, g)):
        sus = bool(x2 - x1 >= 28 * k and s < 0.6); auto = False
        if sus and person is not None:      # big but weak, and no person there: most likely a lamp or an ornament
            r = person[max(0, int(y1 * dh / h)):max(1, int(y2 * dh / h)) + 1, max(0, int(x1 * dw / w)):max(1, int(x2 * dw / w)) + 1]; auto = bool(r.size == 0 or r.mean() < 0.15)
        faces.append(dict(id=q, box=[round(float(v), 1) for v in (x1, y1, x2, y2)], score=round(float(s), 2), suspect=sus, auto=auto))
        rr = int(max(1.6 * max(x2 - x1, y2 - y1), 60 * k)); cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2); x0, y0 = max(0, min(w - 2 * rr, cx - rr)), max(0, min(h - 2 * rr, cy - rr))
        cr = cv2.resize(cv2.LUT(bgr[y0:y0 + 2 * rr, x0:x0 + 2 * rr], bright), (220, 220), interpolation=cv2.INTER_AREA); f = 220.0 / max(1, min(2 * rr, w - x0))
        cv2.rectangle(cr, (int((x1 - x0) * f), int((y1 - y0) * f)), (int((x2 - x0) * f), int((y2 - y0) * f)), (0, 255, 255), 1)
        cv2.imwrite(os.path.join(d, f's_{q}.jpg'), cr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    with open(os.path.join(W, 'faces.json'), 'w') as f: json.dump(dict(people=bool(people), gamma=g, faces=faces), f)
    print(f'faces {len(faces)}, of those doubtful {sum(x["suspect"] for x in faces)}, dropped automatically {sum(x["auto"] for x in faces)}')

class SamCtx:
    """what vblur.py's SAM functions need to know about one still picture"""
    def __init__(s, bgr, crop, g):
        s.H, s.W = bgr.shape[:2]; s.k = min(s.W, s.H) / 1080.0; s.x0, s.y0, x1, y1 = [int(v) for v in crop] if crop else (0, 0, s.W, s.H)
        s.cw, s.ch = x1 - s.x0, y1 - s.y0; s.sc = 1024.0 / max(s.cw, s.ch); s._im = cv2.LUT(bgr, lut(g))
    def img(s, i, d='prev'): return s._im

def sam_masks(W, bgr, g, crop, targets):
    """-> (context, [(outline over the crop, score)] for each target); the picture's SAM features are kept in W for the next call"""
    import vblur as V
    c = SamCtx(bgr, crop, g); ef, sf = os.path.join(W, 'emb.npy'), os.path.join(W, 'emb.sig'); sig = json.dumps([crop, g])
    if not (os.path.exists(ef) and os.path.exists(sf) and open(sf).read() == sig):
        np.save(ef, V.sam_embed(c, 0).astype(np.float16)); open(sf, 'w').write(sig)
    emb = np.load(ef).astype(np.float32); out = []
    for t in targets:
        pts = [(float(x), float(y)) for x, y in t.get('pts') or []]; lab = [int(v) for v in t.get('lab') or [1] * len(pts)]
        m, iou, _ = V.sam_predict(c, emb, pts, lab, box=t.get('box')); out.append((V.clean(c, m), iou))
    return c, out

def cmd_seed(W):
    q = json.loads(sys.stdin.read()); m = meta(W); bgr, _, _ = load(m['src'])
    if not q.get('pts') and not q.get('box'): return print(json.dumps(dict(error='no box and no points')))
    c, r = sam_masks(W, bgr, m['gamma'], q.get('crop'), [q]); mask, iou = r[0]
    cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = [[[int(x) + c.x0, int(y) + c.y0] for x, y in cv2.approxPolyDP(k, 1.5, True).reshape(-1, 2)] for k in cnt if cv2.contourArea(k) > 20]
    print(json.dumps(dict(polys=polys, score=round(float(iou), 2), area=int(mask.sum()))))

def cmd_render(W, out, mode, hide, style, strength, targets_file):
    m = meta(W); bgr, a, icc = load(m['src']); h, w = bgr.shape[:2]; k = min(w, h) / 1080.0; dw, dh = det_size(w, h); kd = min(dw, dh) / 1080.0; fx, fy = dw / w, dh / h
    F = json.load(open(os.path.join(W, 'faces.json'))) if mode != 'pick' else dict(faces=[], people=False)
    hidden = [f for f in F['faces'] if (f['id'] in hide if hide is not None else not f['auto'])]
    if mode == 'faces':
        A = np.zeros((h, w), np.float32)
        for f in hidden:
            x1, y1, x2, y2 = f['box']; bw, bh = x2 - x1, y2 - y1
            B.stamp(A, (x1 + x2) / 2, (y1 + y2) / 2, max(0.5 * bw * 1.5, 7 * k), max(0.5 * bh * 1.5, 8 * k), float(np.clip(0.12 * bw, 2.5 * k, 14 * k)))
    else:
        if mode == 'people':
            if not F['people']: sys.exit('whole-body masks need "detect --people"')
            M = (cv2.imread(os.path.join(W, 'person.png'), 0) > 127).astype(np.uint8)
            for f in hidden:      # every face belongs to a person: cover the head even where the body was missed
                x1, y1, x2, y2 = f['box']; bw, bh = (x2 - x1) * fx, (y2 - y1) * fy
                cv2.ellipse(M, (int((x1 + x2) / 2 * fx), int((y1 + y2) / 2 * fy + 0.15 * bh)), (int(max(1.1 * bw, 8 * kd)), int(max(1.25 * bh, 10 * kd))), 0, 0, 360, 1, -1)
        else:
            q = json.load(open(targets_file, encoding='utf-8')); c, r = sam_masks(W, bgr, m['gamma'], q.get('crop'), q['targets']); full = np.zeros((h, w), np.uint8)
            for mask, _ in r: full[c.y0:c.y0 + c.ch, c.x0:c.x0 + c.cw] |= mask.astype(np.uint8)
            M = cv2.resize(full, (dw, dh), interpolation=cv2.INTER_AREA if (dw, dh) != (w, h) else cv2.INTER_NEAREST)
        A = cv2.resize(B.alpha_of(kd, M > 0) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR) if (dw, dh) != (w, h) else B.alpha_of(kd, M > 0) / 255.0
    s = strength or (42 if style == 'blur' else 36) * k; res = np.empty_like(bgr)
    for ch in range(3):
        p = bgr[:, :, ch].astype(np.float32); res[:, :, ch] = np.clip(p * (1 - A) + B.soften(p, style, s) * A + 0.5, 0, 255).astype(np.uint8)
    save(out, res, a, icc); cv2.imwrite(os.path.join(W, 'out_view.jpg'), fit(res, 1600), [cv2.IMWRITE_JPEG_QUALITY, 88])
    untouched = bool(np.array_equal(res[A == 0], bgr[A == 0])); del res
    # the saved file, read back: outside the hidden areas a lossless format gives the source exactly, a lossy one (jpg, webp) differs slightly
    back = load(out)[0]; se = 0.0; n = 0
    for y in range(0, h, 512):
        keep = A[y:y + 512] == 0; d = back[y:y + 512][keep].astype(np.int32) - bgr[y:y + 512][keep]; se += float((d * d).sum()); n += d.size
    psnr = None if se == 0 else round(float(10 * np.log10(255.0 * 255.0 * n / se)), 1)
    print(json.dumps(dict(cover=round(float((A > 0.5).mean()) * 100, 1), hidden=[f['id'] for f in hidden], size=os.path.getsize(out), untouched=untouched, exact=se == 0, psnr=psnr)))

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest='cmd', required=True)
    ints = lambda v: [int(x) for x in v.split(',') if x.strip()]
    def common(p):
        p.add_argument('--style', choices=['blur', 'mosaic'], default='blur'); p.add_argument('--strength', type=float, default=0, help='blur sigma or mosaic block in px (default scales with the picture size)')
    p = sub.add_parser('blur'); p.add_argument('src'); p.add_argument('out'); p.add_argument('--mode', choices=['faces', 'people'], default='faces'); common(p)
    p = sub.add_parser('info'); p.add_argument('src'); p.add_argument('work')
    p = sub.add_parser('detect'); p.add_argument('work'); p.add_argument('--people', action='store_true'); p.add_argument('--gamma', type=float, default=0)
    p = sub.add_parser('seed'); p.add_argument('work')
    p = sub.add_parser('render'); p.add_argument('work'); p.add_argument('out'); p.add_argument('--mode', choices=['faces', 'people', 'pick'], required=True)
    p.add_argument('--hide', type=ints, help='ids of the faces to hide (default: all but the automatically dropped ones)'); p.add_argument('--targets'); common(p)
    a = ap.parse_args()
    if a.cmd == 'blur':
        W = tempfile.mkdtemp(prefix='imgblur-')
        try: cmd_info(a.src, W); cmd_detect(W, a.mode == 'people', 0); cmd_render(W, a.out, a.mode, None, a.style, a.strength, None)
        finally: shutil.rmtree(W, ignore_errors=True)
    elif a.cmd == 'info': cmd_info(a.src, a.work)
    elif a.cmd == 'detect': cmd_detect(a.work, a.people, a.gamma)
    elif a.cmd == 'seed': cmd_seed(a.work)
    elif a.cmd == 'render': cmd_render(a.work, a.out, a.mode, a.hide, a.style, a.strength, a.targets)

if __name__ == '__main__': main()
