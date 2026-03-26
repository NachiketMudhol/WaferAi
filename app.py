"""
WaferAI v4 — Combined: FP16 ONNX + FP32 GradCAM + Smart Grid Auto-Splitter
═══════════════════════════════════════════════════════════════════════════════
COMBINED FROM:
  • app.py (v2)        → FP16 ONNX classification + FP32 GradCAM dual-pipeline
  • waferai_v3.py (v3) → Smart grid auto-detection & cell splitting

WHAT'S INCLUDED:
  • Upload a single SEM image → FP16 ONNX classifies, FP32 GradCAM visualises
  • Upload a phone photo of a SEM grid sheet → auto-detects grid, crops every
    cell, runs FP16+FP32 pipeline on each cell individually
  • Batch upload: multiple images, each auto-detected as single or grid
  • GradCAM per image/cell, Wafer Map, full score bars, history

HOW TO RUN:
  1. pip install flask flask-cors torch torchvision
         opencv-python-headless Pillow numpy matplotlib onnxruntime
  2. Ensure these files are in Sem_defect_Ai folder:
       best_model.pth           (FP32, used for GradCAM)
       defect_model_fp16.onnx   (FP16, used for classification)
  3. python waferai_v4_combined.py
  4. Open http://localhost:5000
"""

import os, io, base64
import numpy as np
import cv2
from PIL import Image
import torch, torch.nn as nn
from torchvision import models, transforms
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# ── Model file paths ───────────────────────────────────────────────────────
MODEL_PATH_FP32 = os.environ.get("MODEL_PATH",      "C:/Users/nachi/Downloads/Sem_defect_Ai/best_model.pth")
MODEL_PATH_ONNX = os.environ.get("MODEL_PATH_ONNX", "C:/Users/nachi/Downloads/Sem_defect_Ai/defect_model_fp16.onnx")

NUM_CLASSES = 8
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Bridge","Clean","CMP Scratches","Crack","LER","Open","Other","Vias"]

app = Flask(__name__)
CORS(app)

# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_fp32_model(path):
    """Load best_model.pth — full precision, used for GradCAM."""
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES)
    if os.path.exists(path):
        state = torch.load(path, map_location=DEVICE, weights_only=False)
        if isinstance(state, dict):
            m.load_state_dict({k: v.float() for k, v in state.items()}, strict=False)
        else:
            m = state; m.float()
        print(f"✅  FP32 model loaded (GradCAM)  →  {path}")
    else:
        print(f"⚠️   {path} not found — GradCAM using random weights")
    return m.to(DEVICE).eval()

def load_onnx_model(path):
    """Load defect_model_fp16.onnx — REQUIRED for classification (no FP32 fallback)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n❌  FP16 ONNX model not found: '{path}'\n"
            f"    Place 'defect_model_fp16.onnx' in the Sem_defect_Ai folder,\n"
            f"    or set the MODEL_PATH_ONNX environment variable to its full path.\n"
            f"    e.g.  set MODEL_PATH_ONNX=C:\\path\\to\\defect_model_fp16.onnx"
        )
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "❌  onnxruntime is not installed.\n"
            "    Run:  pip install onnxruntime"
        )
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    print(f"✅  ONNX FP16 model loaded (classify)  →  {path}")
    return sess

model     = load_fp32_model(MODEL_PATH_FP32)
onnx_sess = load_onnx_model(MODEL_PATH_ONNX)

def classify(tensor):
    """Classify via ONNX FP16 model only (no FP32 fallback)."""
    inp  = tensor.cpu().numpy().astype(np.float32)
    name = onnx_sess.get_inputs()[0].name
    raw  = onnx_sess.run(None, {name: inp})[0][0]
    e    = np.exp(raw - raw.max())
    return (e / e.sum()), "ONNX FP16 ✓"

infer_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# ══════════════════════════════════════════════════════════════
#  GRADCAM
# ══════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, mdl, layer):
        self.mdl=mdl; self.grads=self.acts=None
        layer.register_forward_hook(lambda m,i,o: setattr(self,'acts',o.detach()))
        layer.register_backward_hook(lambda m,i,o: setattr(self,'grads',o[0].detach()))
    def run(self, tensor):
        self.mdl.zero_grad(); out=self.mdl(tensor)
        idx=out.argmax(1).item(); out[0,idx].backward()
        w=self.grads.mean([0,2,3])
        cam=(self.acts[0]*w[:,None,None]).mean(0).cpu().numpy()
        cam=np.maximum(cam,0); cam=cam/cam.max() if cam.max()>0 else cam
        return cv2.resize(cam,(IMG_SIZE,IMG_SIZE)), idx, out

gradcam = GradCAM(model, model.features[-1][0])

# ══════════════════════════════════════════════════════════════
#  WAFER MAP
# ══════════════════════════════════════════════════════════════

def make_wafer_map(cls_name):
    fig,ax=plt.subplots(figsize=(6,6),facecolor="#050a14")
    ax.set_facecolor("#050a14"); ax.set_aspect("equal"); ax.axis("off")
    dw,dh,gap=0.10,0.10,0.012; total=defect=0
    for ry in np.arange(-1.0,1.05,dh+gap):
        for rx in np.arange(-1.0,1.05,dw+gap):
            cx,cy=rx+dw/2,ry+dh/2; d=np.hypot(cx,cy)
            if d+dw/2>1.0: continue
            total+=1; bad=False
            if   cls_name=="Bridge":        bad=abs(cy)<0.18 or abs(cx)<0.18
            elif cls_name=="Clean":         bad=False
            elif cls_name=="CMP Scratches": bad=abs(np.sin(np.arctan2(cy,cx)*2))<0.18
            elif cls_name=="Crack":         bad=abs(cx-cy)<0.14 or abs(cx+cy)<0.14
            elif cls_name=="LER":           bad=d>0.62 and d<0.88
            elif cls_name=="Open":          bad=d<0.38
            elif cls_name=="Other":         bad=(int(rx*100)+int(ry*100))%5==0
            elif cls_name=="Vias":          bad=0.25<d<0.52 and (int(rx*100))%4==0
            fc="#ff4444" if bad else "#0d2a3a"; ec="#ff7777" if bad else "#0a4060"
            ax.add_patch(plt.Rectangle((rx,ry),dw*.9,dh*.9,lw=.3,ec=ec,fc=fc,alpha=.92))
            if bad: defect+=1
    ax.add_patch(plt.Circle((0,0),1.0,fill=False,ec="#00d4ff",lw=2,alpha=.8))
    ax.plot([-.12,.12],[1,1],c="#00d4ff",lw=4,solid_capstyle="round",alpha=.9)
    for r in [.3,.55,.75,.9]: ax.add_patch(plt.Circle((0,0),r,fill=False,ec="#00d4ff",lw=.5,alpha=.14))
    yld=(total-defect)/total*100 if total else 0
    ax.text(0,-1.14,f"Class: {cls_name}   Dies: {total}   Defective: {defect}   Yield: {yld:.1f}%",
        ha="center",va="center",color="#00d4ff",fontsize=7.5,fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=.3",fc="#0a1628",ec="#00d4ff",alpha=.85))
    ax.set_xlim(-1.2,1.2); ax.set_ylim(-1.28,1.18)
    ax.set_title("Wafer Map — Die Distribution",color="#00d4ff",fontsize=11,fontfamily="monospace",pad=8)
    buf=io.BytesIO()
    plt.savefig(buf,format="png",dpi=150,bbox_inches="tight",facecolor="#050a14")
    plt.close(fig); buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ══════════════════════════════════════════════════════════════
#  GRID SPLITTER  (from waferai_v3)
# ══════════════════════════════════════════════════════════════

def find_grid_lines(gray_np):
    """Detect grid line positions using gradient analysis."""
    H, W = gray_np.shape
    blurred = cv2.GaussianBlur(gray_np, (3,3), 0)
    grad_y = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3))
    grad_x = np.abs(cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3))

    row_g = grad_y.mean(axis=1)
    col_g = grad_x.mean(axis=0)
    if row_g.max()>0: row_g /= row_g.max()
    if col_g.max()>0: col_g /= col_g.max()

    def lines_from_signal(signal, length, thresh=0.45, min_gap_frac=0.04):
        above = signal > thresh
        lines=[]; in_band=False; bs=0
        min_gap=int(length*min_gap_frac)
        for i,v in enumerate(above):
            if v and not in_band: in_band=True; bs=i
            elif not v and in_band:
                in_band=False; c=(bs+i)//2
                if length*0.02 < c < length*0.98:
                    if not lines or c-lines[-1]>min_gap: lines.append(c)
        return lines

    return (lines_from_signal(row_g, H, 0.45, 0.04),
            lines_from_signal(col_g, W, 0.45, 0.06))


def is_grid_image(pil_img):
    """Returns True if image appears to be an N×M grid sheet."""
    gray = np.array(pil_img.convert("L"))
    rows, cols = find_grid_lines(gray)
    return len(rows) >= 1 and len(cols) >= 1


def split_grid_image(pil_img):
    """Auto-split a grid image into individual cells."""
    gray = np.array(pil_img.convert("L"))
    H, W = gray.shape
    row_lines, col_lines = find_grid_lines(gray)

    if len(row_lines) < 1 or len(col_lines) < 1:
        return [{"image": pil_img, "row": 0, "col": 0, "label": "Image"}]

    rows = [0] + row_lines + [H]
    cols = [0] + col_lines + [W]

    widths  = [cols[i+1]-cols[i] for i in range(len(cols)-1)]
    heights = [rows[i+1]-rows[i] for i in range(len(rows)-1)]
    modal_w = sorted(widths)[len(widths)//2]
    modal_h = sorted(heights)[len(heights)//2]

    cells = []
    cell_num = 1
    for ri in range(len(rows)-1):
        for ci in range(len(cols)-1):
            y0,y1 = rows[ri], rows[ri+1]
            x0,x1 = cols[ci], cols[ci+1]
            cw, ch = x1-x0, y1-y0
            if cw < modal_w * 0.55 or ch < modal_h * 0.55:
                continue
            bx = max(2, int(cw * 0.03))
            by = max(2, int(ch * 0.03))
            cell = pil_img.crop((x0+bx, y0+by, x1-bx, y1-by))
            cells.append({"image": cell, "row": ri, "col": ci, "label": f"Cell {cell_num}"})
            cell_num += 1

    return cells if cells else [{"image": pil_img, "row": 0, "col": 0, "label": "Image"}]


def make_grid_overview(cell_results, n_cols):
    """Create a visual overview image showing all cells with their predicted class."""
    CLASS_COLORS_BGR = {
        "Bridge":       (107,107,255), "Clean":        (52,211,153),
        "CMP Scratches":(50,163,251),  "Crack":        (0,170,255),
        "LER":          (0,255,136),   "Open":         (255,212,0),
        "Other":        (184,118,244), "Vias":         (250,139,168),
    }
    n = len(cell_results)
    n_rows = (n + n_cols - 1) // n_cols
    thumb = 150; pad = 4; label_h = 28
    canvas_w = n_cols * (thumb + pad) + pad
    canvas_h = n_rows * (thumb + label_h + pad) + pad
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (14, 20, 10)

    for idx, r in enumerate(cell_results):
        row = idx // n_cols
        col = idx % n_cols
        x = col * (thumb + pad) + pad
        y = row * (thumb + label_h + pad) + pad
        cell_np = np.array(r["cell_orig"])
        if cell_np.ndim == 2:
            cell_np = cv2.cvtColor(cell_np, cv2.COLOR_GRAY2BGR)
        elif cell_np.shape[2] == 4:
            cell_np = cv2.cvtColor(cell_np, cv2.COLOR_RGBA2BGR)
        else:
            cell_np = cv2.cvtColor(cell_np, cv2.COLOR_RGB2BGR)
        cell_resized = cv2.resize(cell_np, (thumb, thumb))
        canvas[y:y+thumb, x:x+thumb] = cell_resized
        color = CLASS_COLORS_BGR.get(r["predicted_class"], (200,200,200))
        cv2.rectangle(canvas, (x,y), (x+thumb,y+thumb), color, 2)
        label_y = y + thumb
        cv2.rectangle(canvas, (x, label_y), (x+thumb, label_y+label_h), (20,30,20), -1)
        cls_text  = r["predicted_class"][:11]
        conf_text = f"{r['confidence']:.0f}%"
        cv2.putText(canvas, cls_text,  (x+3, label_y+12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, conf_text, (x+3, label_y+24), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1, cv2.LINE_AA)

    _, buf = cv2.imencode(".png", canvas)
    return base64.b64encode(buf).decode()

# ══════════════════════════════════════════════════════════════
#  CORE INFERENCE  (FP16 ONNX classify + FP32 GradCAM)
# ══════════════════════════════════════════════════════════════

def run_inference(pil_img, filename=""):
    """Run dual-model pipeline on a single PIL image. Returns result dict."""
    t = infer_tf(pil_img).unsqueeze(0).to(DEVICE)

    # Stage 1: Classification via FP16 ONNX
    probs, classify_src = classify(t)
    idx   = int(probs.argmax())
    pred  = CLASS_NAMES[idx]
    conf  = round(float(probs[idx]) * 100, 2)
    scores = {CLASS_NAMES[i]: round(float(probs[i]) * 100, 2) for i in range(NUM_CLASSES)}

    # Stage 2: GradCAM via FP32 PyTorch
    t32 = t.clone().requires_grad_(True)
    cam, _, _ = gradcam.run(t32)

    orig_np = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE)))
    if orig_np.ndim == 2: orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    _, obuf = cv2.imencode(".png", cv2.cvtColor(orig_np, cv2.COLOR_RGB2BGR))

    gray_rgb = cv2.cvtColor(cv2.cvtColor(orig_np, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    hmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(gray_rgb, 0.45, hmap, 0.55, 0)
    _, gbuf = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    small_np = np.array(pil_img.resize((150, 150)))
    if small_np.ndim == 2: small_np = cv2.cvtColor(small_np, cv2.COLOR_GRAY2RGB)

    return dict(
        predicted_class = pred,
        confidence      = conf,
        scores          = scores,
        original_image  = base64.b64encode(obuf).decode(),
        gradcam_image   = base64.b64encode(gbuf).decode(),
        wafer_map       = make_wafer_map(pred),
        _cell_orig_np   = small_np,
        _classify_src   = classify_src,
    )

# ══════════════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/health")
def health():
    return jsonify(
        status      = "ok",
        model       = "MobileNetV3-Small (Dual FP16+FP32)",
        model_fp32  = str(os.path.exists(MODEL_PATH_FP32)),
        model_onnx  = str(os.path.exists(MODEL_PATH_ONNX)),
        pipeline    = "ONNX FP16 classify → FP32 GradCAM",
        grid_split  = "enabled",
        classes     = NUM_CLASSES,
        device      = str(DEVICE)
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Smart endpoint:
    - Single SEM image  → FP16 classify + FP32 GradCAM, returns single result
    - Grid sheet photo  → auto-splits cells, runs pipeline on each cell,
                          returns array of results + grid overview image
    """
    if "image" not in request.files:
        return jsonify(error="No image field"), 400

    f    = request.files["image"]
    data = f.read()

    try:
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return jsonify(error=f"Cannot open image: {e}"), 400

    try:
        grid_detected = is_grid_image(pil)

        if not grid_detected:
            # ── Single image path ──────────────────────────────
            r = run_inference(pil, f.filename)
            return jsonify(
                mode            = "single",
                is_grid         = False,
                success         = True,
                predicted_class = r["predicted_class"],
                confidence      = r["confidence"],
                scores          = r["scores"],
                original_image  = r["original_image"],
                gradcam_image   = r["gradcam_image"],
                wafer_map       = r["wafer_map"],
                metadata        = dict(
                    filename   = f.filename,
                    device     = str(DEVICE),
                    model_fp16 = r["_classify_src"],
                    model_fp32 = "FP32 ✓" if os.path.exists(MODEL_PATH_FP32) else "FP32 (demo)",
                    pipeline   = "ONNX FP16 classify → FP32 GradCAM"
                )
            )

        # ── Grid path ──────────────────────────────────────────
        cells = split_grid_image(pil)
        gray  = np.array(pil.convert("L"))
        row_l, col_l = find_grid_lines(gray)
        unique_cols  = sorted(set(c["col"] for c in cells))
        n_display_cols = max(len(unique_cols), 1)

        cell_results = []
        for c in cells:
            r = run_inference(c["image"])
            cell_results.append({
                "label":           c["label"],
                "row":             c["row"],
                "col":             c["col"],
                "predicted_class": r["predicted_class"],
                "confidence":      r["confidence"],
                "scores":          r["scores"],
                "original_image":  r["original_image"],
                "gradcam_image":   r["gradcam_image"],
                "wafer_map":       r["wafer_map"],
                "cell_orig":       Image.fromarray(r["_cell_orig_np"]),
            })

        classes_found = [cr["predicted_class"] for cr in cell_results]
        from collections import Counter
        class_counts = dict(Counter(classes_found))
        defective    = [c for c in classes_found if c != "Clean"]
        avg_conf     = round(sum(cr["confidence"] for cr in cell_results) / len(cell_results), 1)

        overview_b64 = make_grid_overview(cell_results, n_display_cols)

        for cr in cell_results:
            del cr["cell_orig"]

        return jsonify(
            mode             = "grid",
            is_grid          = True,
            success          = True,
            grid_rows        = len(row_l) + 1,
            grid_cols        = n_display_cols,
            total_cells      = len(cell_results),
            defective_count  = len(defective),
            clean_count      = len(cell_results) - len(defective),
            avg_confidence   = avg_conf,
            class_summary    = class_counts,
            grid_overview    = overview_b64,
            cells            = cell_results,
            metadata         = dict(
                filename   = f.filename,
                device     = str(DEVICE),
                pipeline   = "ONNX FP16 classify → FP32 GradCAM"
            )
        )

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify(error=str(e)), 500


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """Multi-file batch — each file goes through smart predict logic."""
    files = request.files.getlist("images")
    if not files: return jsonify(error="No images uploaded"), 400
    results = []
    for f in files:
        try:
            pil  = Image.open(io.BytesIO(f.read())).convert("RGB")
            grid = is_grid_image(pil)
            if grid:
                cells = split_grid_image(pil)
                crs   = []
                for c in cells:
                    r = run_inference(c["image"])
                    crs.append({
                        "label":           c["label"],
                        "predicted_class": r["predicted_class"],
                        "confidence":      r["confidence"],
                        "scores":          r["scores"],
                        "original_image":  r["original_image"],
                        "gradcam_image":   r["gradcam_image"],
                        "wafer_map":       r["wafer_map"],
                    })
                results.append(dict(
                    success=True, mode="grid", filename=f.filename,
                    cells=crs, total_cells=len(crs)
                ))
            else:
                r = run_inference(pil, f.filename)
                results.append(dict(
                    success=True, mode="single",
                    filename=f.filename,
                    predicted_class=r["predicted_class"],
                    confidence=r["confidence"],
                    scores=r["scores"],
                    original_image=r["original_image"],
                    gradcam_image=r["gradcam_image"],
                    wafer_map=r["wafer_map"],
                    metadata=dict(
                        filename=f.filename, device=str(DEVICE),
                        model_fp16=r["_classify_src"],
                        model_fp32="FP32 ✓" if os.path.exists(MODEL_PATH_FP32) else "FP32 (demo)",
                        pipeline="ONNX FP16 classify → FP32 GradCAM"
                    )
                ))
        except Exception as e:
            results.append(dict(success=False, metadata={"filename": f.filename}, error=str(e)))
    return jsonify(results=results, total=len(results))


# ══════════════════════════════════════════════════════════════
#  FRONTEND  (Combined v2 UI + v3 Grid View tab)
# ══════════════════════════════════════════════════════════════

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>WaferAI v4 — FP16 + Grid Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet"/>
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#050a14;--blue:#00d4ff;--green:#00ff88;--amber:#ffaa00;--red:#ff6b6b;
  --card:rgba(255,255,255,.03);--border:rgba(255,255,255,.08)}
body{background:var(--bg);color:#fff;font-family:'Syne',sans-serif;min-height:100vh}
.mono{font-family:'Space Mono',monospace}
body::before{content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background-image:linear-gradient(rgba(0,212,255,.03) 1px,transparent 1px),
  linear-gradient(90deg,rgba(0,212,255,.03) 1px,transparent 1px);background-size:44px 44px}
#root{position:relative;z-index:1}
::-webkit-scrollbar{width:4px}::-webkit-scrollbar-thumb{background:rgba(0,212,255,.3);border-radius:2px}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.3;transform:scale(.6)}}
@keyframes radar{to{transform:rotate(360deg)}}
@keyframes sonar{0%{transform:scale(.4);opacity:.9}100%{transform:scale(2.6);opacity:0}}
@keyframes scan{0%{top:-2px}100%{top:100%}}
@keyframes fadeUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes glow{0%,100%{box-shadow:0 0 10px rgba(0,212,255,.25)}50%{box-shadow:0 0 28px rgba(0,212,255,.6)}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-7px)}}
.pulse-dot{animation:pulse 2s ease-in-out infinite}
.float-anim{animation:float 4s ease-in-out infinite}
.fade-up{animation:fadeUp .5s ease forwards}
/* Nav */
nav{position:sticky;top:0;z-index:100;background:rgba(5,10,20,.9);
  backdrop-filter:blur(20px);border-bottom:1px solid rgba(0,212,255,.12)}
.nav-in{max-width:1400px;margin:auto;padding:0 24px;height:64px;
  display:flex;align-items:center;justify-content:space-between}
.logo{display:flex;align-items:center;gap:12px;cursor:pointer}
.logo-icon{width:38px;height:38px;border-radius:10px;background:rgba(0,212,255,.14);
  border:1px solid rgba(0,212,255,.4);display:flex;align-items:center;justify-content:center}
.logo-name{font-size:1.2rem;font-weight:800;letter-spacing:-.02em}
.logo-name span{color:var(--blue)}
.logo-tag{font-size:.52rem;color:rgba(255,255,255,.28);letter-spacing:.12em;
  font-family:'Space Mono',monospace;margin-top:-3px}
.nav-links{display:flex;gap:2px}
.nb{padding:8px 14px;border:none;background:transparent;color:rgba(255,255,255,.42);
  cursor:pointer;font-family:'Syne',sans-serif;font-size:.86rem;font-weight:600;
  border-radius:8px;border-bottom:2px solid transparent;transition:all .25s}
.nb:hover{color:#fff}.nb.active{color:var(--blue);border-bottom-color:var(--blue)}
.mbadge{display:flex;align-items:center;gap:8px;padding:6px 14px;border-radius:99px;
  background:rgba(0,255,136,.07);border:1px solid rgba(0,255,136,.22)}
.mbadge span{font-family:'Space Mono',monospace;font-size:.7rem;color:var(--green)}
.hamburger{display:none;background:none;border:none;color:rgba(255,255,255,.6);cursor:pointer;padding:8px}
.mob-menu{display:none;flex-direction:column;gap:4px;padding:8px 16px 16px;border-top:1px solid rgba(255,255,255,.06)}
.mob-menu.open{display:flex}
/* Layout */
main{max-width:1400px;margin:auto;padding:0 24px 80px}
.sec{display:none}.sec.active{display:block}
.card{background:var(--card);border:1px solid var(--border);border-radius:18px;backdrop-filter:blur(12px)}
.btn{border:none;cursor:pointer;font-family:'Syne',sans-serif;font-weight:700;
  border-radius:12px;transition:all .3s;display:inline-flex;align-items:center;gap:8px}
.btn-p{background:linear-gradient(135deg,#00d4ff,#0070f3);color:#fff}
.btn-p:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(0,212,255,.38);filter:brightness(1.08)}
.btn-p:disabled{opacity:.35;cursor:not-allowed;transform:none;box-shadow:none}
.btn-g{background:var(--card);border:1px solid var(--border);color:rgba(255,255,255,.55)}
.btn-g:hover{border-color:rgba(0,212,255,.35);color:#fff;background:rgba(0,212,255,.08)}
.btn-r{background:rgba(255,107,107,.12);border:1px solid rgba(255,107,107,.3);color:var(--red)}
.btn-sm{padding:8px 16px;font-size:.82rem}.btn-lg{padding:15px 40px;font-size:1.05rem}
/* Hero */
.hero{padding:42px 0 32px;text-align:center}
.chip{display:inline-flex;align-items:center;gap:8px;padding:5px 14px;border-radius:99px;
  background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.25);margin-bottom:16px}
.chip span{font-size:.72rem;color:var(--blue);font-family:'Space Mono',monospace}
h1{font-size:clamp(1.8rem,5vw,3.2rem);font-weight:800;letter-spacing:-.03em;line-height:1.12;margin-bottom:12px}
.gt{background:linear-gradient(90deg,var(--blue),var(--green));-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero p{color:rgba(255,255,255,.4);max-width:580px;margin:0 auto;font-size:.9rem;line-height:1.65}
/* Upload */
.upload-zone{border:2px dashed rgba(0,212,255,.28);border-radius:22px;min-height:230px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:pointer;position:relative;overflow:hidden;transition:all .3s;padding:32px 24px;text-align:center}
.upload-zone:hover,.upload-zone.drag{border-color:var(--blue);background:rgba(0,212,255,.04);animation:glow 1.8s infinite}
.scanline{position:absolute;left:0;right:0;height:2px;
  background:linear-gradient(90deg,transparent,var(--blue),transparent);animation:scan 2.2s linear infinite}
.upload-icon{width:72px;height:72px;border-radius:18px;background:rgba(0,212,255,.1);
  border:1px solid rgba(0,212,255,.3);display:flex;align-items:center;justify-content:center;margin-bottom:14px}
.pills{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:10px}
.pill{padding:3px 10px;border-radius:6px;font-size:.7rem;font-family:'Space Mono',monospace;
  background:rgba(0,212,255,.1);color:var(--blue);border:1px solid rgba(0,212,255,.22)}
/* Overlay */
#overlay{position:fixed;inset:0;z-index:200;background:rgba(5,10,20,.97);
  backdrop-filter:blur(18px);display:none;flex-direction:column;align-items:center;justify-content:center}
#overlay.show{display:flex}
.radar-wrap{position:relative;width:128px;height:128px;margin-bottom:28px}
.radar-ring{position:absolute;inset:0;border:2px solid transparent;
  border-top-color:var(--blue);border-radius:50%;animation:radar 1.8s linear infinite}
.sonar-ring{position:absolute;inset:0;border-radius:50%;
  border:1px solid rgba(0,212,255,.3);animation:sonar 2.2s ease-out infinite}
.sonar-ring:nth-child(2){animation-delay:.55s}.sonar-ring:nth-child(3){animation-delay:1.1s}
.radar-icon{position:absolute;inset:0;display:flex;align-items:center;justify-content:center}
/* Error */
.err-bar{padding:11px 16px;border-radius:12px;display:none;align-items:center;gap:12px;
  background:rgba(255,100,100,.1);border:1px solid rgba(255,100,100,.3);margin-bottom:16px}
.err-bar.show{display:flex}
.err-bar span{font-size:.84rem;color:var(--red);flex:1}
/* Results */
.res-hero{padding:22px 26px;margin-bottom:16px;border:1px solid rgba(0,255,136,.18);
  box-shadow:0 0 40px rgba(0,255,136,.05)}
.conf-num{font-size:clamp(2.8rem,8vw,4.8rem);font-weight:700;
  font-family:'Space Mono',monospace;color:var(--green);letter-spacing:-.04em;line-height:1}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.ch{padding:12px 16px;display:flex;align-items:center;justify-content:space-between;
  border-bottom:1px solid rgba(255,255,255,.06)}
.cl{font-size:.65rem;font-family:'Space Mono',monospace;color:rgba(255,255,255,.3);letter-spacing:.12em}
.img-panel{height:200px;overflow:hidden;cursor:zoom-in}
.img-panel img{width:100%;height:100%;object-fit:cover;transition:transform .4s;display:block}
.img-panel:hover img{transform:scale(1.04)}
.meta-rows{padding:10px 14px}
.mr{display:flex;justify-content:space-between;padding:3px 0;
  border-bottom:1px solid rgba(255,255,255,.04);font-size:.76rem}
.mr span:first-child{color:rgba(255,255,255,.3)}
.mr span:last-child{font-family:'Space Mono',monospace;color:rgba(255,255,255,.75)}
.cam-ctrl{padding:10px 14px;display:flex;flex-direction:column;gap:8px}
.mode-tabs{display:flex;gap:4px}
.mt{flex:1;padding:5px 0;border:1px solid rgba(255,255,255,.07);border-radius:7px;
  background:rgba(255,255,255,.04);cursor:pointer;font-family:'Space Mono',monospace;
  font-size:.7rem;color:rgba(255,255,255,.36);transition:all .25s}
.mt.active{background:rgba(0,212,255,.18);border-color:rgba(0,212,255,.4);color:var(--blue)}
.sl-row{display:flex;align-items:center;gap:10px;font-size:.76rem}
.sl-row span:first-child{color:rgba(255,255,255,.3)}
.sl-row input{flex:1;accent-color:var(--blue);height:4px}
.sl-row .vl{font-family:'Space Mono',monospace;color:var(--blue);width:36px;text-align:right}
.score-bars{padding:12px 14px}
.score-row{display:flex;align-items:center;gap:8px;margin-bottom:7px}
.sn{width:90px;text-align:right;font-size:.72rem;font-family:'Space Mono',monospace;flex-shrink:0}
.st{flex:1;height:18px;border-radius:99px;overflow:hidden;background:rgba(255,255,255,.05)}
.sf{height:100%;border-radius:99px;transition:width 1s cubic-bezier(.4,0,.2,1)}
.sp{width:46px;font-size:.72rem;font-family:'Space Mono',monospace}
.stat-grid{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-top:16px}
.stat-card{border-radius:14px;padding:14px;background:var(--card);border:1px solid var(--border);transition:all .3s}
.stat-lbl{font-size:.6rem;font-family:'Space Mono',monospace;color:rgba(255,255,255,.3);letter-spacing:.1em}
.stat-val{font-size:1.5rem;font-weight:700;font-family:'Space Mono',monospace;margin-top:7px}
.stat-card:hover{transform:translateY(-2px)}
/* Grid overview */
.batch-stats{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px}
.bst{padding:12px 18px;border-radius:12px;background:var(--card);border:1px solid var(--border);flex:1;min-width:85px}
.bst-val{font-size:1.35rem;font-weight:700;font-family:'Space Mono',monospace}
.bst-lbl{font-size:.6rem;color:rgba(255,255,255,.3);font-family:'Space Mono',monospace;margin-top:3px}
.overview-img-wrap{border-radius:16px;overflow:hidden;border:1px solid rgba(0,212,255,.2)}
.overview-img-wrap img{width:100%;display:block}
.cells-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px;margin-top:16px}
.cell-card{border-radius:14px;overflow:hidden;border:1px solid var(--border);
  background:var(--card);cursor:pointer;transition:all .3s}
.cell-card:hover{transform:translateY(-3px);border-color:rgba(0,212,255,.4);
  box-shadow:0 8px 24px rgba(0,212,255,.1)}
.cell-card.sel{border-color:var(--blue);box-shadow:0 0 18px rgba(0,212,255,.2)}
.cell-card img{width:100%;height:115px;object-fit:cover;display:block}
.cell-card-body{padding:9px 12px}
.cc-cls{font-weight:700;font-size:.88rem;margin-bottom:2px}
.cc-conf{font-family:'Space Mono',monospace;font-size:.76rem}
.cc-lbl{font-size:.64rem;color:rgba(255,255,255,.3);margin-top:3px}
/* Batch */
.bc{border-radius:14px;overflow:hidden;border:1px solid var(--border);background:var(--card);cursor:pointer;transition:all .3s}
.bc:hover{transform:translateY(-3px);border-color:rgba(0,212,255,.4)}
.bc.sel{border-color:var(--blue)}
.bc img{width:100%;height:125px;object-fit:cover;display:block}
.bc-body{padding:9px 12px}
.bc-cls{font-weight:700;font-size:.88rem;margin-bottom:2px}
.bc-conf{font-family:'Space Mono',monospace;font-size:.76rem}
.bc-file{font-size:.64rem;color:rgba(255,255,255,.3);margin-top:3px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
/* Queue */
.qi{padding:8px 14px;border-radius:10px;background:var(--card);border:1px solid var(--border);
  display:flex;align-items:center;gap:10px;font-size:.82rem}
.qs{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.qs.pending{background:rgba(255,255,255,.2)}.qs.running{background:var(--amber);animation:pulse 1s infinite}
.qs.done{background:var(--green)}.qs.error{background:var(--red)}
/* Wafer */
.wafer-img{width:100%;max-height:460px;object-fit:contain;border-radius:14px}
.mini-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:13px;margin-top:16px}
.mc{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:15px;transition:all .3s}
.mc:hover{transform:translateY(-2px);border-color:rgba(0,212,255,.3)}
.mc .mv{font-size:1.3rem;font-weight:700;font-family:'Space Mono',monospace}
/* History */
.hh{display:grid;padding:10px 18px;
  grid-template-columns:2.5rem 1fr 130px 90px 90px 90px;
  font-size:.65rem;font-family:'Space Mono',monospace;color:rgba(255,255,255,.27);
  letter-spacing:.08em;border-bottom:1px solid rgba(255,255,255,.06)}
.hr{display:grid;padding:12px 18px;
  grid-template-columns:2.5rem 1fr 130px 90px 90px 90px;
  align-items:center;border-bottom:1px solid rgba(255,255,255,.04);cursor:pointer;transition:background .2s}
.hr:hover{background:rgba(0,212,255,.05)}.hr:last-child{border-bottom:none}
.ht{width:32px;height:32px;border-radius:6px;object-fit:cover;border:1px solid rgba(0,212,255,.2)}
.hf{display:flex;align-items:center;gap:9px}
.hn{font-size:.84rem;font-weight:600;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.sp-pill{padding:3px 10px;border-radius:99px;font-size:.67rem;font-family:'Space Mono',monospace;display:inline-block}
.sp-def{background:rgba(255,107,107,.12);color:var(--red);border:1px solid rgba(255,107,107,.25)}
.sp-ok{background:rgba(0,255,136,.12);color:var(--green);border:1px solid rgba(0,255,136,.25)}
.sp-grid{background:rgba(0,212,255,.12);color:var(--blue);border:1px solid rgba(0,212,255,.25)}
@media(max-width:900px){
  .g3{grid-template-columns:1fr}.stat-grid{grid-template-columns:repeat(3,1fr)}
  .mini-grid{grid-template-columns:repeat(2,1fr)}
  .hh,.hr{grid-template-columns:2.5rem 1fr 100px 80px}
  .hh span:nth-child(n+5),.hr>*:nth-child(n+5){display:none}}
@media(max-width:640px){
  .nav-links,.mbadge{display:none}.hamburger{display:block}
  .stat-grid{grid-template-columns:repeat(2,1fr)}
  .cells-grid{grid-template-columns:repeat(2,1fr)}
  main{padding:0 14px 60px}}
footer{border-top:1px solid rgba(255,255,255,.06);background:rgba(5,10,20,.82);padding:20px 24px}
.fi{max-width:1400px;margin:auto;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px}
</style>
</head>
<body>
<nav>
<div class="nav-in">
  <div class="logo" onclick="S('analyze')">
    <div class="logo-icon">
      <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
        <circle cx="11" cy="11" r="9" stroke="#00d4ff" stroke-width="1.5"/>
        <circle cx="11" cy="11" r="5" stroke="#00d4ff" stroke-width="1" stroke-dasharray="2 2"/>
        <circle cx="11" cy="11" r="2" fill="#00d4ff"/>
        <line x1="11" y1="2" x2="11" y2="5" stroke="#00d4ff" stroke-width="1.5"/>
        <line x1="11" y1="17" x2="11" y2="20" stroke="#00d4ff" stroke-width="1.5"/>
        <line x1="2" y1="11" x2="5" y2="11" stroke="#00d4ff" stroke-width="1.5"/>
        <line x1="17" y1="11" x2="20" y2="11" stroke="#00d4ff" stroke-width="1.5"/>
      </svg>
    </div>
    <div><div class="logo-name">Wafer<span>AI</span></div><div class="logo-tag">FP16 DUAL-MODEL + GRID DETECTION v4</div></div>
  </div>
  <div class="nav-links">
    <button class="nb active" onclick="S('analyze')">Analyze</button>
    <button class="nb" onclick="S('results')">Dashboard</button>
    <button class="nb" onclick="S('grid')">Grid View</button>
    <button class="nb" onclick="S('batch')">Batch</button>
    <button class="nb" onclick="S('wafer')">Wafer Map</button>
    <button class="nb" onclick="S('history')">Reports</button>
  </div>
  <div class="mbadge">
    <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);flex-shrink:0" class="pulse-dot"></span>
    <span>v4 &bull; FP16 &bull; Grid Split</span>
  </div>
  <button class="hamburger" onclick="document.getElementById('mmenu').classList.toggle('open')">
    <svg width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="3" y1="6" x2="19" y2="6"/><line x1="3" y1="11" x2="19" y2="11"/><line x1="3" y1="16" x2="19" y2="16"/>
    </svg>
  </button>
</div>
<div class="mob-menu" id="mmenu">
  <button class="nb active" onclick="S('analyze');document.getElementById('mmenu').classList.remove('open')">Analyze</button>
  <button class="nb" onclick="S('results');document.getElementById('mmenu').classList.remove('open')">Dashboard</button>
  <button class="nb" onclick="S('grid');document.getElementById('mmenu').classList.remove('open')">Grid View</button>
  <button class="nb" onclick="S('batch');document.getElementById('mmenu').classList.remove('open')">Batch</button>
  <button class="nb" onclick="S('history');document.getElementById('mmenu').classList.remove('open')">Reports</button>
</div>
</nav>

<div id="overlay">
  <div class="radar-wrap">
    <div class="radar-ring"></div>
    <div class="sonar-ring"></div><div class="sonar-ring"></div><div class="sonar-ring"></div>
    <div class="radar-icon">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><circle cx="11" cy="11" r="3"/>
      </svg>
    </div>
  </div>
  <p id="ovMode" style="font-size:.82rem;font-family:'Space Mono',monospace;color:var(--blue);margin-bottom:8px;
    padding:4px 12px;border-radius:8px;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.25)"></p>
  <h2 style="color:#fff;font-size:1.2rem;margin-bottom:5px" id="ovTitle">Processing...</h2>
  <p class="mono" style="color:rgba(255,255,255,.36);font-size:.76rem;margin-bottom:8px" id="ovSub"></p>
  <div id="ovProgress" style="width:260px;height:5px;border-radius:99px;background:rgba(255,255,255,.08);margin:12px 0;display:none">
    <div id="ovFill" style="height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .3s;width:0%"></div>
  </div>
</div>

<div id="root"><main>

<!-- ANALYZE -->
<div class="sec active" id="sec-analyze">
  <div class="hero fade-up">
    <div class="chip">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
      <span>WAFERAI v4 — FP16 ONNX + FP32 GRADCAM + AUTO GRID SPLIT</span>
    </div>
    <h1>SEM Defect Analysis<br><span class="gt">Dual-Model · Smart Grid</span></h1>
    <p>
      Upload a <strong style="color:var(--blue)">single SEM image</strong> or a
      <strong style="color:var(--green)">grid sheet photo</strong> —
      FP16 ONNX classifies every die, FP32 GradCAM visualises attention,
      and grid sheets are auto-detected &amp; split per cell.
    </p>
  </div>

  <!-- Pipeline banner -->
  <div class="card" style="padding:18px 22px;margin-bottom:24px;display:flex;gap:24px;flex-wrap:wrap;align-items:center">
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;min-width:120px">
      <div style="width:44px;height:44px;border-radius:12px;background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.3);display:flex;align-items:center;justify-content:center">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
      </div>
      <span style="font-size:.78rem;font-weight:700;color:var(--blue)">Grid Auto-Split</span>
      <span style="font-size:.7rem;color:rgba(255,255,255,.38);text-align:center">Cells detected &amp; cropped</span>
    </div>
    <div style="color:rgba(255,255,255,.2);font-size:1.5rem">→</div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;min-width:120px">
      <div style="width:44px;height:44px;border-radius:12px;background:rgba(255,170,0,.1);border:1px solid rgba(255,170,0,.3);display:flex;align-items:center;justify-content:center">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#ffaa00" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
      </div>
      <span style="font-size:.78rem;font-weight:700;color:var(--amber)">FP16 ONNX</span>
      <span style="font-size:.7rem;color:rgba(255,255,255,.38);text-align:center">Fast classification</span>
    </div>
    <div style="color:rgba(255,255,255,.2);font-size:1.5rem">→</div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;min-width:120px">
      <div style="width:44px;height:44px;border-radius:12px;background:rgba(0,255,136,.1);border:1px solid rgba(0,255,136,.3);display:flex;align-items:center;justify-content:center">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00ff88" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
      </div>
      <span style="font-size:.78rem;font-weight:700;color:var(--green)">FP32 GradCAM</span>
      <span style="font-size:.7rem;color:rgba(255,255,255,.38);text-align:center">Attention heatmap</span>
    </div>
    <div style="color:rgba(255,255,255,.2);font-size:1.5rem">→</div>
    <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex:1;min-width:120px">
      <div style="width:44px;height:44px;border-radius:12px;background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.3);display:flex;align-items:center;justify-content:center">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
      </div>
      <span style="font-size:.78rem;font-weight:700;color:#a78bfa">Wafer Map</span>
      <span style="font-size:.7rem;color:rgba(255,255,255,.38);text-align:center">Die-level yield view</span>
    </div>
  </div>

  <div class="err-bar" id="errBar">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ff6b6b" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="#ff6b6b"/></svg>
    <span id="errMsg"></span>
    <button onclick="hideErr()" style="background:none;border:none;cursor:pointer;color:var(--red);margin-left:auto">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
    </button>
  </div>

  <!-- Single upload -->
  <div class="upload-zone" id="upZone"
    ondragover="event.preventDefault();this.classList.add('drag')"
    ondragleave="this.classList.remove('drag')"
    ondrop="onDrop(event)"
    onclick="document.getElementById('fi').click()">
    <div class="scanline" id="scanLine" style="display:none"></div>
    <div id="upIcon" class="upload-icon float-anim">
      <svg width="34" height="34" viewBox="0 0 24 24" fill="none" stroke="#00d4ff" stroke-width="1.5">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
        <polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>
      </svg>
    </div>
    <div id="upPreviewWrap" style="display:none;margin-bottom:14px;position:relative">
      <img id="upPreview" style="max-height:160px;max-width:240px;border-radius:12px;
        border:2px solid rgba(0,255,136,.5);box-shadow:0 0 20px rgba(0,255,136,.25)" alt="preview"/>
      <div style="position:absolute;top:-8px;right:-8px;width:26px;height:26px;border-radius:50%;
        background:var(--green);display:flex;align-items:center;justify-content:center">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#050a14" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>
      </div>
    </div>
    <p id="upLabel" style="font-size:1rem;font-weight:700;margin-bottom:6px">Drop SEM Image or Click to Upload</p>
    <p id="upSub" style="font-size:.82rem;color:rgba(255,255,255,.35);margin-bottom:12px">Single die OR full grid sheet — auto-detected &amp; split</p>
    <div class="pills">
      <span class="pill">PNG</span><span class="pill">JPG</span><span class="pill">BMP</span><span class="pill">TIF</span>
      <span class="pill" style="background:rgba(255,170,0,.1);color:var(--amber);border-color:rgba(255,170,0,.3)">FP16 ONNX ✓</span>
      <span class="pill" style="background:rgba(0,255,136,.1);color:var(--green);border-color:rgba(0,255,136,.3)">GRID AUTO-SPLIT ✓</span>
    </div>
    <input type="file" id="fi" accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" style="display:none" onchange="onFileSel(event)"/>
  </div>

  <div style="display:flex;justify-content:center;gap:14px;margin-top:24px">
    <button class="btn btn-p btn-lg" id="runBtn" disabled onclick="runAnalysis()">
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      Analyze Image
    </button>
    <button class="btn btn-g btn-sm" id="resetBtn" style="display:none" onclick="resetAll()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-4.95"/></svg>Reset
    </button>
  </div>

  <!-- Batch queue -->
  <div class="card" style="margin-top:32px;padding:20px 24px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:10px">
      <div>
        <div style="font-weight:700;font-size:1rem">Batch Queue</div>
        <div style="font-size:.78rem;color:rgba(255,255,255,.35);margin-top:2px">Add multiple images for batch analysis</div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap">
        <button class="btn btn-g btn-sm" onclick="document.getElementById('bfi').click()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>Add Images
        </button>
        <button class="btn btn-p btn-sm" id="batchRunBtn" disabled onclick="runBatch()">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg>Run Batch
        </button>
        <button class="btn btn-r btn-sm" id="clearQBtn" style="display:none" onclick="clearQ()">Clear</button>
      </div>
      <input type="file" id="bfi" multiple accept=".png,.jpg,.jpeg,.bmp,.tif,.tiff" style="display:none" onchange="addToQ(event)"/>
    </div>
    <div id="queueList" style="display:flex;flex-direction:column;gap:6px"></div>
    <div id="queueEmpty" style="padding:20px;text-align:center;color:rgba(255,255,255,.25);font-size:.84rem">No images in queue</div>
    <div id="batchProgress" style="display:none;margin-top:14px">
      <div style="height:5px;border-radius:99px;background:rgba(255,255,255,.08)">
        <div id="pfill" style="height:100%;border-radius:99px;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .3s;width:0%"></div>
      </div>
    </div>
  </div>
</div>

<!-- DASHBOARD -->
<div class="sec" id="sec-results">
  <div id="noRes" style="display:flex;flex-direction:column;align-items:center;padding:80px 0">
    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
    <p style="color:rgba(255,255,255,.36);font-size:1rem;font-weight:600;margin-bottom:8px">No Analysis Yet</p>
    <button class="btn btn-p btn-sm" onclick="S('analyze')">Go to Analyze</button>
  </div>
  <div id="resContent" style="display:none">
    <div id="gridBackBtn" style="display:none;margin:18px 0 6px">
      <button class="btn btn-g btn-sm" onclick="S('grid')">&#8592; Back to Grid Overview</button>
      <span class="mono" style="font-size:.75rem;color:rgba(255,255,255,.3);margin-left:12px" id="cellLabel"></span>
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;padding:20px 0 14px">
      <h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Result Dashboard</h2>
      <div id="resNav" style="display:none;align-items:center;gap:8px">
        <button class="btn btn-g btn-sm" id="btnPrev" onclick="prevR()">&#8249;</button>
        <span class="mono" style="font-size:.82rem" id="navLbl"></span>
        <button class="btn btn-g btn-sm" id="btnNext" onclick="nextR()">&#8250;</button>
      </div>
    </div>

    <div class="card res-hero fade-up" style="margin-bottom:16px">
      <div style="display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:14px">
        <div>
          <div class="mono" style="font-size:.64rem;color:rgba(255,255,255,.28);letter-spacing:.12em;margin-bottom:6px">PRIMARY DETECTION</div>
          <div style="font-size:clamp(1.4rem,4vw,2.5rem);font-weight:800;letter-spacing:-.02em;margin-bottom:8px">
            <span id="rCls">—</span> <span style="color:rgba(255,255,255,.24)">Defect</span>
          </div>
          <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
            <span class="conf-num" id="rConf">—</span>
            <div>
              <div style="font-size:.76rem;color:rgba(255,255,255,.34)">Confidence</div>
              <div class="mono" id="r2nd" style="font-size:.68rem;margin-top:4px;padding:3px 10px;border-radius:8px;
                background:rgba(255,170,0,.14);color:var(--amber);border:1px solid rgba(255,170,0,.28);display:inline-block">2nd: —</div>
            </div>
          </div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <button class="btn btn-p btn-sm" onclick="S('wafer')">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>Wafer Map
          </button>
          <a id="dlGC" class="btn btn-g btn-sm" download="gradcam.png">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>GradCAM
          </a>
        </div>
      </div>
    </div>

    <div class="g3" style="margin-bottom:16px">
      <div class="card fade-up">
        <div class="ch"><span class="cl">ORIGINAL IMAGE</span></div>
        <div class="img-panel"><img id="rOrig" src="" alt="original"/></div>
        <div class="meta-rows">
          <div class="mr"><span>File / Cell</span><span id="mFile">—</span></div>
          <div class="mr"><span>Device</span><span id="mDev">cpu</span></div>
          <div class="mr"><span>FP16 Model</span><span id="mFp16">—</span></div>
          <div class="mr"><span>FP32 Model</span><span id="mFp32">—</span></div>
          <div class="mr"><span>Pipeline</span><span id="mPipeline">FP16→FP32</span></div>
        </div>
      </div>
      <div class="card fade-up" style="animation-delay:.1s">
        <div class="ch"><span class="cl">GRADCAM THERMAL</span></div>
        <div class="img-panel"><img id="rGrad" src="" alt="gradcam"/></div>
        <div class="cam-ctrl">
          <div class="mode-tabs">
            <button class="mt active" onclick="setMode('overlay',this)">overlay</button>
            <button class="mt"        onclick="setMode('blend',this)">blend</button>
            <button class="mt"        onclick="setMode('split',this)">split</button>
          </div>
          <div class="sl-row">
            <span>Opacity</span>
            <input type="range" min="20" max="100" value="100" oninput="setOp(this.value)"/>
            <span class="vl" id="opVal">100%</span>
          </div>
        </div>
      </div>
      <div class="card fade-up" style="animation-delay:.2s">
        <div class="ch"><span class="cl">CLASS PROBABILITIES</span></div>
        <div class="score-bars" id="scoreBars"></div>
      </div>
    </div>
    <div class="stat-grid" id="statGrid"></div>
  </div>
</div>

<!-- GRID VIEW -->
<div class="sec" id="sec-grid">
  <div style="padding:24px 0 16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
    <div>
      <h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Grid Analysis</h2>
      <p style="color:rgba(255,255,255,.36);font-size:.87rem;margin-top:4px" id="gridSubtitle">No grid analysed yet</p>
    </div>
    <a id="dlOverview" class="btn btn-p btn-sm" download="grid_overview.png" style="display:none">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>Export Overview
    </a>
  </div>
  <div id="gridEmpty" class="card" style="padding:70px;display:flex;flex-direction:column;align-items:center">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/></svg>
    <p style="color:rgba(255,255,255,.34)">Upload a grid sheet image to see the auto-split analysis</p>
    <button class="btn btn-p btn-sm" style="margin-top:16px" onclick="S('analyze')">Upload Grid Image</button>
  </div>
  <div id="gridContent" style="display:none">
    <div class="batch-stats" id="gridStats"></div>
    <div style="margin-bottom:20px">
      <div class="mono" style="font-size:.7rem;color:rgba(255,255,255,.28);letter-spacing:.1em;margin-bottom:10px">GRID OVERVIEW — ALL CELLS WITH PREDICTED CLASS</div>
      <div class="overview-img-wrap"><img id="overviewImg" src="" alt="grid overview"/></div>
    </div>
    <div class="mono" style="font-size:.7rem;color:rgba(255,255,255,.28);letter-spacing:.1em;margin-bottom:12px">CLICK ANY CELL FOR FULL DETAILS (GradCAM + Wafer Map)</div>
    <div class="cells-grid" id="cellsGrid"></div>
  </div>
</div>

<!-- BATCH RESULTS -->
<div class="sec" id="sec-batch">
  <div style="padding:24px 0 16px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px">
    <div>
      <h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Batch Results</h2>
      <p style="color:rgba(255,255,255,.36);font-size:.87rem;margin-top:4px" id="bSubtitle">No batch run yet</p>
    </div>
  </div>
  <div id="bEmpty" class="card" style="padding:70px;display:flex;flex-direction:column;align-items:center">
    <p style="color:rgba(255,255,255,.34)">Add images to the batch queue and run batch analysis.</p>
    <button class="btn btn-p btn-sm" style="margin-top:16px" onclick="S('analyze')">Go to Analyze</button>
  </div>
  <div id="bContent" style="display:none">
    <div class="batch-stats" id="bStats"></div>
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:12px;margin-top:4px" id="bGrid"></div>
  </div>
</div>

<!-- WAFER MAP -->
<div class="sec" id="sec-wafer">
  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:14px;padding:24px 0 16px">
    <div>
      <h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Wafer Map</h2>
      <p style="color:rgba(255,255,255,.36);font-size:.87rem;margin-top:4px">
        Die-level defect distribution — <span id="wCls" style="color:var(--blue)">run analysis first</span>
      </p>
    </div>
    <a id="dlW" class="btn btn-p btn-sm" download="wafer_map.png" style="display:none">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>Export PNG
    </a>
  </div>
  <div id="wEmpty" class="card" style="padding:70px;display:flex;flex-direction:column;align-items:center">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="3"/></svg>
    <p style="color:rgba(255,255,255,.34)">Run an analysis to generate the wafer map</p>
    <button class="btn btn-p btn-sm" style="margin-top:16px" onclick="S('analyze')">Analyze Image</button>
  </div>
  <div id="wContent" style="display:none">
    <div class="card" style="padding:16px;display:flex;justify-content:center"><img id="wImg" class="wafer-img" src="" alt="wafer"/></div>
    <div class="mini-grid">
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">CLASS</div><div class="mv" id="wStatCls" style="color:var(--blue)">—</div></div>
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">CONFIDENCE</div><div class="mv" id="wStatConf" style="color:var(--green)">—</div></div>
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">YIELD</div><div class="mv" style="color:var(--amber)">—</div></div>
      <div class="mc"><div class="mono" style="font-size:.6rem;color:rgba(255,255,255,.26);margin-bottom:7px">STATUS</div><div class="mv" id="wStatSt" style="color:var(--red)">—</div></div>
    </div>
  </div>
</div>

<!-- HISTORY -->
<div class="sec" id="sec-history">
  <div style="padding:24px 0 16px">
    <h2 style="font-size:1.8rem;font-weight:800;letter-spacing:-.02em">Reports</h2>
    <p style="color:rgba(255,255,255,.36);font-size:.87rem;margin-top:4px" id="hCount">0 scans this session</p>
  </div>
  <div id="hEmpty" class="card" style="padding:70px;display:flex;flex-direction:column;align-items:center">
    <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="rgba(0,212,255,.26)" stroke-width="1.5" style="margin-bottom:14px"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
    <p style="color:rgba(255,255,255,.34)">No scans yet</p>
  </div>
  <div id="hTable" style="display:none">
    <div class="card" style="overflow:hidden">
      <div class="hh"><span>#</span><span>FILE</span><span>CLASS</span><span>CONF</span><span>TIME</span><span>STATUS</span></div>
      <div id="hRows"></div>
    </div>
  </div>
</div>

</main>
<footer>
  <div class="fi">
    <div style="font-size:1rem;font-weight:800">Wafer<span style="color:var(--blue)">AI</span>
      <span style="font-weight:400;font-size:.78rem;color:rgba(255,255,255,.26);margin-left:8px">FP16 Dual-Model + Smart Grid Detection v4</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px">
      <span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green)" class="pulse-dot"></span>
      <span class="mono" style="font-size:.68rem;color:rgba(255,255,255,.26)">FP16 ONNX Classify &bull; FP32 GradCAM &bull; Auto Grid Split</span>
    </div>
  </div>
</footer>
</div>

<script>
const CC={Bridge:'#ff6b6b',Clean:'#00ff88','CMP Scratches':'#00d4ff',Crack:'#ffaa00',
  LER:'#a78bfa',Open:'#f472b6',Other:'#fb923c',Vias:'#34d399'};

const STATS=[
  {l:'Accuracy',v:'98.0%',c:'#00ff88'},{l:'Model',v:'MNV3',c:'#00d4ff'},
  {l:'FP16 ONNX',v:'✓',c:'#ffaa00'},{l:'Classes',v:'8',c:'#a78bfa'},
  {l:'Input',v:'224×224',c:'#00d4ff'},{l:'Grid Split',v:'Auto',c:'#00ff88'}
];

let currentFile=null,currentPreviewURL=null,running=false,histArr=[],lastResult=null;
let Q=[],BR=[],VI=0;

function S(id){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
  document.getElementById('sec-'+id).classList.add('active');
  document.querySelectorAll('.nb').forEach(b=>{
    b.classList.toggle('active',b.getAttribute('onclick')&&b.getAttribute('onclick').includes("'"+id+"'"));
  });
  window.scrollTo(0,0);
}

// ── Single upload ──────────────────────────────────────────────
function onDrop(e){e.preventDefault();document.getElementById('upZone').classList.remove('drag');
  if(e.dataTransfer.files[0])setFile(e.dataTransfer.files[0]);}
function onFileSel(e){if(e.target.files[0])setFile(e.target.files[0]);e.target.value='';}

function setFile(f){
  currentFile=f;
  if(currentPreviewURL)URL.revokeObjectURL(currentPreviewURL);
  currentPreviewURL=URL.createObjectURL(f);
  const prev=document.getElementById('upPreview');
  prev.src=currentPreviewURL;
  document.getElementById('upPreviewWrap').style.display='block';
  document.getElementById('upIcon').style.display='none';
  document.getElementById('scanLine').style.display='block';
  document.getElementById('upLabel').textContent=f.name;
  document.getElementById('upLabel').style.color='var(--green)';
  document.getElementById('upSub').textContent='Click Analyze — grid auto-detected automatically';
  document.getElementById('runBtn').disabled=false;
  document.getElementById('resetBtn').style.display='inline-flex';
  hideErr();
}

function resetAll(){
  currentFile=null;
  if(currentPreviewURL){URL.revokeObjectURL(currentPreviewURL);currentPreviewURL=null;}
  document.getElementById('upPreviewWrap').style.display='none';
  document.getElementById('upIcon').style.display='flex';
  document.getElementById('scanLine').style.display='none';
  document.getElementById('upLabel').textContent='Drop SEM Image or Click to Upload';
  document.getElementById('upLabel').style.color='';
  document.getElementById('upSub').textContent='Single die OR full grid sheet — auto-detected & split';
  document.getElementById('runBtn').disabled=true;
  document.getElementById('resetBtn').style.display='none';
  document.getElementById('fi').value='';
  hideErr();
}

function showErr(m){document.getElementById('errMsg').textContent=m;document.getElementById('errBar').classList.add('show');}
function hideErr(){document.getElementById('errBar').classList.remove('show');}

function showOv(mode,sub){
  const ov=document.getElementById('overlay');ov.classList.add('show');
  document.getElementById('ovMode').textContent=mode;
  document.getElementById('ovTitle').textContent='Processing...';
  document.getElementById('ovSub').textContent=sub||'';
  document.getElementById('ovProgress').style.display='none';
}
function hideOv(){document.getElementById('overlay').classList.remove('show');}

// ── Single analysis ────────────────────────────────────────────
async function runAnalysis(){
  if(!currentFile||running)return;
  running=true;document.getElementById('runBtn').disabled=true;hideErr();
  showOv('Detecting image type...','FP16 ONNX → FP32 GradCAM');
  try{
    const fd=new FormData();fd.append('image',currentFile);
    const res=await fetch('/predict',{method:'POST',body:fd});
    if(!res.ok){const e=await res.json();throw new Error(e.error||'Server error');}
    const data=await res.json();
    lastResult=data;hideOv();
    if(data.mode==='single'){
      renderSingle(data,currentFile.name);
      addHist(data,currentFile.name,'single');
      S('results');
    } else {
      renderGrid(data,currentFile.name);
      if(data.cells&&data.cells.length>0)renderSingleFromCell(data.cells[0],currentFile.name,data.cells[0].label);
      addHist(data,currentFile.name,'grid');
      S('grid');
    }
  }catch(err){
    hideOv();
    showErr(err.message==='Failed to fetch'?'Cannot reach backend. Run: python waferai_v4_combined.py':err.message);
  }
  running=false;document.getElementById('runBtn').disabled=false;
}

// ── Batch queue ────────────────────────────────────────────────
function addToQ(e){
  Array.from(e.target.files).forEach(f=>{Q.push({file:f,status:'pending'});});
  e.target.value='';
  renderQueue();
  document.getElementById('batchRunBtn').disabled=Q.length===0;
  document.getElementById('clearQBtn').style.display=Q.length>0?'inline-flex':'none';
  document.getElementById('queueEmpty').style.display=Q.length===0?'block':'none';
}
function clearQ(){Q=[];renderQueue();document.getElementById('batchRunBtn').disabled=true;document.getElementById('clearQBtn').style.display='none';document.getElementById('queueEmpty').style.display='block';}
function renderQueue(){
  const el=document.getElementById('queueList');el.innerHTML='';
  Q.forEach((q,i)=>{
    const d=document.createElement('div');d.className='qi';
    d.innerHTML=`<div class="qs ${q.status}" id="qs${i}"></div><span style="flex:1;font-size:.82rem">${q.file.name}</span><span class="mono" style="font-size:.72rem;color:rgba(255,255,255,.3)">${(q.file.size/1024).toFixed(0)}KB</span>`;
    el.appendChild(d);
  });
}
function setQS(i,s){Q[i].status=s;const el=document.getElementById('qs'+i);if(el){el.className='qs '+s;}}

async function runBatch(){
  if(Q.length===0||running)return;
  running=true;BR=[];
  document.getElementById('batchProgress').style.display='block';
  const pfill=document.getElementById('pfill');pfill.style.width='0%';
  const total=Q.length;
  let done=0;
  for(let i=0;i<Q.length;i++){
    setQS(i,'running');
    try{
      const fd=new FormData();fd.append('image',Q[i].file);
      const res=await fetch('/predict',{method:'POST',body:fd});
      if(!res.ok){const e=await res.json();throw new Error(e.error||'err');}
      const data=await res.json();
      // Normalise grid vs single for batch display
      if(data.mode==='grid'){
        BR.push({success:true,is_grid:true,predicted_class:'Grid ('+data.total_cells+' cells)',
          confidence:data.avg_confidence,scores:{},gradcam_image:data.cells[0]?.gradcam_image||'',
          metadata:{filename:Q[i].file.name,device:data.metadata?.device||'cpu',
            model_fp16:'ONNX FP16 ✓',model_fp32:'FP32 ✓',pipeline:'FP16→FP32'},
          _grid_data:data});
      } else {
        BR.push({...data,success:true});
      }
      setQS(i,'done');addHist(data,Q[i].file.name,data.mode);
    }catch(e){BR.push({success:false,metadata:{filename:Q[i].file.name},error:e.message});setQS(i,'error');}
    done++;pfill.style.width=(done/total*100)+'%';
  }
  const first=BR.find(r=>r.success&&!r.is_grid);
  if(first){VI=BR.filter(r=>r.success).indexOf(first);renderSingle(first,first.metadata.filename);}
  renderBatch(BR);S('batch');running=false;
}

// ── Render single result ───────────────────────────────────────
function renderSingle(d,filename){
  if(!d||!d.success)return;
  const sorted=Object.entries(d.scores||{}).sort((a,b)=>b[1]-a[1]);
  const tc=d.predicted_class,cf=d.confidence,sec=sorted[1]||['-',0];
  document.getElementById('rCls').textContent=tc;
  document.getElementById('rConf').textContent=cf.toFixed(1)+'%';
  document.getElementById('r2nd').textContent='2nd: '+sec[0]+' '+Number(sec[1]).toFixed(1)+'%';
  document.getElementById('rOrig').src='data:image/png;base64,'+(d.original_image||'');
  const gs='data:image/png;base64,'+(d.gradcam_image||'');
  document.getElementById('rGrad').src=gs;
  document.getElementById('mFile').textContent=(filename||d.metadata?.filename||'—');
  document.getElementById('mDev').textContent=d.metadata?.device||'cpu';
  document.getElementById('mFp16').textContent=d.metadata?.model_fp16||'ONNX FP16 ✓';
  document.getElementById('mFp32').textContent=d.metadata?.model_fp32||'FP32 ✓';
  document.getElementById('mPipeline').textContent=d.metadata?.pipeline||'FP16→FP32';
  document.getElementById('dlGC').href=gs;
  const ws='data:image/png;base64,'+(d.wafer_map||'');
  document.getElementById('dlW').href=ws;document.getElementById('dlW').style.display='inline-flex';
  document.getElementById('wImg').src=ws;
  document.getElementById('wCls').textContent=tc;
  document.getElementById('wStatCls').textContent=tc;
  document.getElementById('wStatConf').textContent=cf.toFixed(1)+'%';
  document.getElementById('wStatSt').textContent=tc==='Clean'?'Clean':'Defective';
  document.getElementById('wStatSt').style.color=tc==='Clean'?'var(--green)':'var(--red)';
  document.getElementById('wEmpty').style.display='none';document.getElementById('wContent').style.display='block';
  renderScoreBars(sorted,tc,cf);
  document.getElementById('statGrid').innerHTML=STATS.map(s=>`<div class="stat-card" style="box-shadow:0 0 18px ${s.c}11"><div class="stat-lbl">${s.l}</div><div class="stat-val" style="color:${s.c}">${s.v}</div></div>`).join('');
  document.getElementById('noRes').style.display='none';document.getElementById('resContent').style.display='block';
  document.getElementById('gridBackBtn').style.display='none';document.getElementById('cellLabel').textContent='';
  const sr=BR.filter(r=>r.success&&!r.is_grid),nav=document.getElementById('resNav');
  if(sr.length>1){nav.style.display='flex';document.getElementById('navLbl').textContent=(VI+1)+'/'+sr.length;document.getElementById('btnPrev').disabled=VI===0;document.getElementById('btnNext').disabled=VI===sr.length-1;}
  else nav.style.display='none';
}

function renderSingleFromCell(cell,filename,cellLabel){
  const fake={scores:cell.scores,predicted_class:cell.predicted_class,confidence:cell.confidence,
    gradcam_image:cell.gradcam_image,original_image:cell.original_image,wafer_map:cell.wafer_map,
    success:true,metadata:{device:'cpu',model_fp16:'ONNX FP16 ✓',model_fp32:'FP32 ✓',pipeline:'FP16→FP32'}};
  renderSingle(fake,filename);
  document.getElementById('gridBackBtn').style.display='block';
  document.getElementById('cellLabel').textContent=cellLabel;
}

function renderScoreBars(sorted,topClass,topConf){
  const bars=document.getElementById('scoreBars');bars.innerHTML='';
  sorted.forEach(([name,score],i)=>{
    const isTop=i===0,col=CC[name]||'#00d4ff',pct=isTop?100:(score/topConf*100);
    const d=document.createElement('div');d.className='score-row';
    d.innerHTML=`<span class="sn" style="color:${isTop?col:'rgba(255,255,255,.36)'}">${name}</span>
      <div class="st"><div class="sf" style="width:0%;--w:${Math.max(pct,score>0?3:0)}%;
        background:${isTop?`linear-gradient(90deg,${col}88,${col})`:'linear-gradient(90deg,rgba(0,212,255,.2),rgba(0,212,255,.45))'};
        box-shadow:${isTop?`0 0 10px ${col}55`:'none'}"></div></div>
      <span class="sp" style="color:${isTop?col:'rgba(255,255,255,.3)'}">${score.toFixed(1)}%</span>`;
    bars.appendChild(d);
  });
  setTimeout(()=>{document.querySelectorAll('.sf').forEach(el=>el.style.width=el.style.getPropertyValue('--w')||'0%');},120);
}

// ── Render grid ────────────────────────────────────────────────
function renderGrid(data,filename){
  document.getElementById('gridSubtitle').textContent=
    `${data.grid_rows} × ${data.grid_cols} grid — ${data.total_cells} cells analysed`;
  const topCls=Object.entries(data.class_summary||{}).sort((a,b)=>b[1]-a[1])[0];
  document.getElementById('gridStats').innerHTML=[
    {l:'TOTAL CELLS',v:data.total_cells,c:'var(--blue)'},
    {l:'DEFECTIVE',v:data.defective_count,c:'var(--red)'},
    {l:'CLEAN',v:data.clean_count,c:'var(--green)'},
    {l:'AVG CONF',v:data.avg_confidence+'%',c:'var(--amber)'},
    {l:'TOP CLASS',v:topCls?topCls[0]:'—',c:CC[topCls?topCls[0]:'']||'var(--blue)'},
    {l:'GRID SIZE',v:data.grid_rows+'×'+data.grid_cols,c:'rgba(255,255,255,.6)'},
  ].map(s=>`<div class="bst"><div class="bst-val" style="color:${s.c}">${s.v}</div><div class="bst-lbl">${s.l}</div></div>`).join('');
  const ovSrc='data:image/png;base64,'+data.grid_overview;
  document.getElementById('overviewImg').src=ovSrc;
  document.getElementById('dlOverview').href=ovSrc;document.getElementById('dlOverview').style.display='inline-flex';
  const grid=document.getElementById('cellsGrid');grid.innerHTML='';
  (data.cells||[]).forEach((cell,i)=>{
    const col=CC[cell.predicted_class]||'#00d4ff',clean=cell.predicted_class==='Clean';
    const c=document.createElement('div');c.className='cell-card';
    c.innerHTML=`<img src="data:image/png;base64,${cell.gradcam_image}" alt="${cell.label}"/>
      <div class="cell-card-body">
        <div class="cc-cls" style="color:${col}">${cell.predicted_class}</div>
        <div class="cc-conf" style="color:${cell.confidence>95?'var(--green)':'var(--amber)'}">
          ${cell.confidence.toFixed(1)}%
          <span class="sp-pill ${clean?'sp-ok':'sp-def'}" style="font-size:.6rem;margin-left:6px">${clean?'clean':'defect'}</span>
        </div>
        <div class="cc-lbl">${cell.label}</div>
      </div>`;
    c.onclick=()=>{
      document.querySelectorAll('.cell-card').forEach(x=>x.classList.remove('sel'));c.classList.add('sel');
      renderSingleFromCell(cell,filename,cell.label);S('results');
    };
    grid.appendChild(c);
  });
  document.getElementById('gridEmpty').style.display='none';document.getElementById('gridContent').style.display='block';
}

// ── Render batch ───────────────────────────────────────────────
function renderBatch(results){
  const ok=results.filter(r=>r.success);
  const def=ok.filter(r=>r.predicted_class!=='Clean'&&!r.is_grid);
  const avg=ok.length?(ok.reduce((s,r)=>s+(r.confidence||0),0)/ok.length).toFixed(1):0;
  const cm={};ok.forEach(r=>{cm[r.predicted_class]=(cm[r.predicted_class]||0)+1;});
  const top=Object.entries(cm).sort((a,b)=>b[1]-a[1])[0];
  document.getElementById('bSubtitle').textContent=ok.length+' image'+(ok.length!==1?'s':'')+' analyzed';
  document.getElementById('bStats').innerHTML=[
    {l:'TOTAL',v:ok.length,c:'var(--blue)'},{l:'DEFECTIVE',v:def.length,c:'var(--red)'},
    {l:'CLEAN',v:ok.filter(r=>r.predicted_class==='Clean').length,c:'var(--green)'},
    {l:'AVG CONF',v:avg+'%',c:'var(--amber)'},
    {l:'TOP CLASS',v:top?top[0]:'—',c:CC[top?top[0]:'']||'var(--blue)'},
  ].map(s=>`<div class="bst"><div class="bst-val" style="color:${s.c}">${s.v}</div><div class="bst-lbl">${s.l}</div></div>`).join('');
  const grid=document.getElementById('bGrid');grid.innerHTML='';
  results.filter(r=>r.success).forEach((r,i)=>{
    const col=CC[r.predicted_class]||'#00d4ff',clean=r.predicted_class==='Clean';
    const c=document.createElement('div');c.className='bc'+(i===VI?' sel':'');
    c.innerHTML=`<img src="data:image/png;base64,${r.gradcam_image||''}" alt=""/>
      <div class="bc-body">
        <div class="bc-cls" style="color:${col}">${r.predicted_class}</div>
        <div class="bc-conf" style="color:${(r.confidence||0)>97?'var(--green)':'var(--amber)'}">
          ${Number(r.confidence||0).toFixed(1)}%
          <span class="sp-pill ${clean?'sp-ok':r.is_grid?'sp-grid':'sp-def'}" style="font-size:.6rem">${clean?'clean':r.is_grid?'grid':'defect'}</span>
        </div>
        <div class="bc-file">${r.metadata?.filename||''}</div>
      </div>`;
    c.onclick=()=>{
      VI=i;document.querySelectorAll('.bc').forEach(x=>x.classList.remove('sel'));c.classList.add('sel');
      if(r.is_grid){renderGrid(r._grid_data,r.metadata.filename);S('grid');}
      else{renderSingle(r,r.metadata?.filename);S('results');}
    };
    grid.appendChild(c);
  });
  results.filter(r=>!r.success).forEach(r=>{
    const c=document.createElement('div');c.className='bc';c.style.cssText='border-color:rgba(255,107,107,.3);cursor:default';
    c.innerHTML=`<div style="height:125px;background:rgba(255,107,107,.08);display:flex;align-items:center;justify-content:center">
      <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><circle cx="12" cy="16" r="1" fill="var(--red)"/></svg>
      </div><div class="bc-body"><div class="bc-cls" style="color:var(--red)">Error</div>
      <div class="bc-conf" style="color:rgba(255,255,255,.38);font-size:.7rem">${r.error||'Failed'}</div>
      <div class="bc-file">${r.metadata?.filename||'unknown'}</div></div>`;
    grid.appendChild(c);
  });
  document.getElementById('bEmpty').style.display='none';document.getElementById('bContent').style.display='block';
}

function prevR(){const sr=BR.filter(r=>r.success&&!r.is_grid);if(VI>0){VI--;renderSingle(sr[VI],sr[VI].metadata?.filename);}}
function nextR(){const sr=BR.filter(r=>r.success&&!r.is_grid);if(VI<sr.length-1){VI++;renderSingle(sr[VI],sr[VI].metadata?.filename);}}

// ── History ────────────────────────────────────────────────────
function addHist(data,filename,type){
  let thumb,cls,conf,status;
  if(type==='grid'){
    const first=data.cells&&data.cells[0];
    thumb=first?'data:image/png;base64,'+first.gradcam_image:'';
    cls=data.class_summary?Object.entries(data.class_summary).sort((a,b)=>b[1]-a[1])[0]?.[0]:'Mixed';
    conf=data.avg_confidence;status='grid';
  } else {
    thumb='data:image/png;base64,'+(data.gradcam_image||'');
    cls=data.predicted_class;conf=data.confidence;
    status=data.predicted_class==='Clean'?'clean':'defective';
  }
  histArr.unshift({file:filename,cls,conf,time:new Date().toLocaleTimeString(),thumb,status,type});
  document.getElementById('hEmpty').style.display='none';document.getElementById('hTable').style.display='block';
  document.getElementById('hCount').textContent=histArr.length+' scan(s) this session';
  document.getElementById('hRows').innerHTML=histArr.map((h,i)=>`
    <div class="hr">
      <span class="mono" style="color:rgba(255,255,255,.2);font-size:.72rem">${String(i+1).padStart(2,'0')}</span>
      <div class="hf"><img src="${h.thumb}" class="ht" alt=""/><span class="hn">${h.file}</span></div>
      <div style="display:flex;align-items:center;gap:8px">
        <span style="width:8px;height:8px;border-radius:50%;background:${CC[h.cls]||'var(--blue)'};flex-shrink:0;display:inline-block"></span>
        <span style="font-size:.84rem">${h.cls}</span>
      </div>
      <span class="mono" style="font-weight:700;color:${h.conf>97?'var(--green)':'var(--amber)'}">${typeof h.conf==='number'?h.conf.toFixed(1):h.conf}%</span>
      <span class="mono" style="font-size:.75rem;color:rgba(255,255,255,.35)">${h.time}</span>
      <span class="sp-pill ${h.status==='clean'?'sp-ok':h.status==='grid'?'sp-grid':'sp-def'}">${h.status}</span>
    </div>`).join('');
}

function setMode(m,btn){document.querySelectorAll('.mt').forEach(t=>t.classList.remove('active'));btn.classList.add('active');document.getElementById('rGrad').style.opacity=m==='split'?'0.5':'1';}
function setOp(v){document.getElementById('rGrad').style.opacity=v/100;document.getElementById('opVal').textContent=v+'%';}

document.getElementById('statGrid').innerHTML=STATS.map(s=>`<div class="stat-card" style="box-shadow:0 0 18px ${s.c}11"><div class="stat-lbl">${s.l}</div><div class="stat-val" style="color:${s.c}">${s.v}</div></div>`).join('');
</script>
</body></html>"""


@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "="*64)
    print("  WaferAI v4 — FP16 ONNX + FP32 GradCAM + Smart Grid Split")
    print("="*64)
    print(f"  Open:        http://localhost:{port}")
    print(f"  FP32 model:  {'✅ found' if os.path.exists(MODEL_PATH_FP32) else '⚠️  not found (demo)'}")
    print(f"  ONNX model:  {'✅ found (REQUIRED)' if os.path.exists(MODEL_PATH_ONNX) else '❌  NOT FOUND — app will NOT start without it'}")
    print(f"  Pipeline:    ONNX FP16 classify → FP32 GradCAM")
    print(f"  Grid Split:  ✅ enabled (auto-detects N×M grid sheets)")
    print(f"  Device:      {DEVICE}")
    print("="*64 + "\n")
    app.run(host="0.0.0.0", port=port, debug=False)
