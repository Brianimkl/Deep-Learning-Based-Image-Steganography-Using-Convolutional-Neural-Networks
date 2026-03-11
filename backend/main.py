
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse, StreamingResponse
from pathlib import Path
import io, traceback, math, logging
from .utils import tv_resize, preprocess_secret
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode as IM
from PIL import Image
from PIL import ImageOps
from .model import StegoSystem
from .train import train as train_fn
from .utils import to_pil 
from fastapi import Query

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger("stego")

#這邊是全域狀態
CKPT_PATH = Path("backend/runs_256x64_colorfix_v1/best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model = None
_cover_size = 128     #這兩個預設的size，之後會被之後的大小覆蓋
_secret_size = 32      

app = FastAPI(title="Image-in-Image Steganography API")

#預設的ip
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8001",
        "http://localhost:8001",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# 內部調整的工具
def tv_resize(img: Image.Image, size_hw, interpolation=IM.BILINEAR):
 
    try:
        return T.Resize(size_hw, interpolation=interpolation, antialias=True)(img)
    except TypeError:
        return T.Resize(size_hw, interpolation=interpolation)(img)

def pil_to_tensor(img: Image.Image):
    return T.ToTensor()(img)

def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
    #計算ssim
    mu_x = x.mean(dim=(2,3), keepdim=True)
    mu_y = y.mean(dim=(2,3), keepdim=True)
    sigma_x = ((x - mu_x)**2).mean(dim=(2,3), keepdim=True)
    sigma_y = ((y - mu_y)**2).mean(dim=(2,3), keepdim=True)
    sigma_xy = ((x - mu_x)*(y - mu_y)).mean(dim=(2,3), keepdim=True)
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2)
    )
    return ssim_map.mean()

#模型的載入
def load_model():
    global _model, _cover_size, _secret_size
    if CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=device)
        args = ckpt.get("args", {})
        _secret_size = int(args.get("secret_size", _secret_size))
        _cover_size  = int(args.get("cover_size",  _cover_size))
        m = StegoSystem(secret_size=_secret_size).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        _model = m
        logger.info(f"[load_model] loaded {CKPT_PATH} (cover={_cover_size}, secret={_secret_size}) on {device}")
        app.state.ckpt_path = str(CKPT_PATH) 
        app.state.ckpt_args = args
    else:
        _model = StegoSystem(secret_size=_secret_size).to(device)
        _model.eval()
        logger.info(f"[load_model] no ckpt, using fresh model (cover={_cover_size}, secret={_secret_size}) on {device}")

    app.state.model = _model
    app.state.device = device
    app.state.cover_size = _cover_size
    app.state.secret_size = _secret_size

@app.on_event("startup")
def _startup():
    load_model()

@app.get("/health")
def health():
    return {
        "ok": True,
        "ckpt": CKPT_PATH.exists(),
        "ckpt_path": getattr(app.state, "ckpt_path", str(CKPT_PATH)),
        "ckpt_args": getattr(app.state, "ckpt_args", {}),
        "cover_size": app.state.cover_size,
        "secret_size": app.state.secret_size,
        "device": str(app.state.device),
    }


@app.post("/train")
def train_endpoint(epochs: int = 3):
    path = train_fn(epochs=epochs, outdir="backend/runs")
    load_model()
    return {"ok": True, "ckpt": str(path)}

#encode

@app.post("/encode")
async def encode(
    cover: UploadFile = File(...),
    secret: UploadFile = File(...),
    sec_bg: str = Query("white", description='"white" or "black"'),
    sec_keep_aspect: int = Query(1),
    sec_binarize: int = Query(0),
    sec_th: float = Query(0.55)
):
    try:
        cover_b = await cover.read(); secret_b = await secret.read()
        if not cover_b or not secret_b:
            raise HTTPException(status_code=400, detail="empty file")

        cover_img  = Image.open(io.BytesIO(cover_b)).convert("RGB")
        secret_img = Image.open(io.BytesIO(secret_b)) 

        Cs = app.state.cover_size
        Ss = app.state.secret_size

        # 做一致化處理
        cover_resized  = tv_resize(cover_img,  (Cs, Cs), IM.BILINEAR)
        # 先做secret圖片的前處理，再做NEAREST
        secret_proc = preprocess_secret(secret_img, Ss,
                                       bg=sec_bg,
                                       keep_aspect=bool(sec_keep_aspect),
                                       binarize=bool(sec_binarize),
                                       th=sec_th)

        t = T.ToTensor()
        c = t(cover_resized).unsqueeze(0).to(app.state.device)
        s = t(secret_proc).unsqueeze(0).to(app.state.device)

        with torch.no_grad():
            stego, _ = app.state.model(c, s)
            stego = stego.clamp(0, 1)

        mse_enc = torch.mean((stego - c) ** 2).item()
        psnr_enc = float("inf") if mse_enc == 0 else 10 * math.log10(1.0 / mse_enc)

        stego_pil = T.ToPILImage()(stego[0].cpu())
        stego_pil = stego_pil.resize((cover_img.width, cover_img.height), Image.BICUBIC)
        buf = io.BytesIO(); stego_pil.save(buf, format="PNG"); buf.seek(0)

        headers = {
            "X-CKPT-SIZES":      f"cover={Cs},secret={Ss}",
            "X-ENC-MSE-COVER":   f"{mse_enc:.6f}",
            "X-ENC-PSNR":        f"{psnr_enc:.3f}",
            "X-SEC-PROC":        f"bg={sec_bg},keep={sec_keep_aspect},bin={sec_binarize},th={sec_th:.2f}"
        }
        return Response(content=buf.getvalue(), media_type="image/png", headers=headers)
    except HTTPException as e:
        return JSONResponse({"ok": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        logger.exception("ENCODE ERROR"); return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# decode

from fastapi import Query

@app.post("/decode")
async def decode(
    stego: UploadFile = File(...),
    secret_gt: UploadFile = File(None), #再輸入是可以做是否要計算ssim的選擇，要的話要多上傳一個secret的原圖
    binarize: int = Query(0, description="1=閾值化"),
    th: float = Query(0.50, description="閾值(0~1)"),
    gray: int = Query(0, description="1=灰階輸出"),
    scale: int = Query(0, description="放大倍數(0=自動)"),
    download: int = Query(0, description="1=強制下載檔案")
):
    try:
        stego_b = await stego.read()
        if not stego_b:
            raise HTTPException(status_code=400, detail="empty file")

        img = Image.open(io.BytesIO(stego_b)).convert("RGB")
        Cs = app.state.cover_size
        Ss = app.state.secret_size

        # 訓練完後，在梭回去cover size
        img_r = tv_resize(img, (Cs, Cs), IM.BILINEAR)
        x = pil_to_tensor(img_r).unsqueeze(0).to(app.state.device)

        with torch.no_grad():
            rec = app.state.model.decoder(x).clamp(0, 1)

        #如果前面有說要一起計算ssim時，在做計算
        ssim_val = None
        if secret_gt is not None:
            secret_b = await secret_gt.read()
            if secret_b:
                sec_img = Image.open(io.BytesIO(secret_b)).convert("RGB")
                # 讓 GT的尺寸保持跟訓練的依樣
                sec_r = tv_resize(sec_img, (Ss, Ss), IM.NEAREST)
                sec_t = pil_to_tensor(sec_r).unsqueeze(0).to(app.state.device)

                # ssim也用rgb三通到下去做計算
                ssim_val = float(ssim_simple(rec, sec_t).item())

        # 可以多選的，如果遇到logo類的可以開讓圖片更清晰
        if gray:
            rec = rec.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
        if binarize:
            if rec.shape[1] == 3:
                y = (0.299 * rec[:, 0:1] + 0.587 * rec[:, 1:2] + 0.114 * rec[:, 2:3])
            else:
                y = rec
            rec = (y > th).float().repeat(1, 3, 1, 1)

        
        arr = rec.detach().cpu().numpy()
        mn, mx, mm = float(arr.min()), float(arr.max()), float(arr.mean())

        # 在展示的時候讓圖片放大方便展示
        rec_pil = T.ToPILImage()(rec.squeeze(0).cpu())
        if scale <= 0:
            scale = max(1, 256 // Ss)
        rec_pil = rec_pil.resize((rec_pil.width * scale, rec_pil.height * scale), Image.NEAREST)

        buf = io.BytesIO(); rec_pil.save(buf, format="PNG"); buf.seek(0)

        headers = {
            "X-DEC-STATS": f"min={mn:.6f};max={mx:.6f};mean={mm:.6f}",
            "X-CKPT-SIZES": f"cover={Cs},secret={Ss}",
            "X-POSTPROC": f"gray={gray},binarize={binarize},th={th:.2f},scale={scale}"
        }

        # ssim加到header中
        if ssim_val is not None:
            headers["X-REC-SSIM-SECRET"] = f"{ssim_val:.4f}"

        if download:
            headers["Content-Disposition"] = 'attachment; filename="decoded.png"'
        return StreamingResponse(buf, media_type="image/png", headers=headers)

    except HTTPException as e:
        return JSONResponse({"ok": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        logger.exception("DECODE ERROR")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# 殘差的計算與展示(圖示化的那種)
@app.post("/residual")
async def residual(cover: UploadFile = File(...), stego: UploadFile = File(...), amp: int = Form(15)):
    try:
        cov = Image.open(io.BytesIO(await cover.read())).convert("RGB")
        stg = Image.open(io.BytesIO(await stego.read())).convert("RGB")
        H = min(cov.height, stg.height); W = min(cov.width, stg.width)
        cov = cov.resize((W, H), Image.BICUBIC)
        stg = stg.resize((W, H), Image.BICUBIC)

        t = T.ToTensor()
        cov_t = t(cov); stg_t = t(stg)
        diff = (stg_t - cov_t).abs().clamp(0, 1) * amp
        d = diff.mean(0, keepdim=True).expand(3, -1, -1)
        img = to_pil(d)
        buf = io.BytesIO(); img.save(buf, format="PNG"); buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# 分析
@app.post("/analyze")
async def analyze(cover: UploadFile = File(...), stego: UploadFile = File(...)):
    try:
        cov = Image.open(io.BytesIO(await cover.read())).convert("RGB")
        stg = Image.open(io.BytesIO(await stego.read())).convert("RGB")
        H = min(cov.height, stg.height); W = min(cov.width, stg.width)
        cov = cov.resize((W, H), Image.BICUBIC)
        stg = stg.resize((W, H), Image.BICUBIC)

        a = T.ToTensor()(cov); b = T.ToTensor()(stg)
        mse = torch.mean((a - b) ** 2).item()
        psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
        mu_x, mu_y = a.mean(), b.mean()
        sigma_x, sigma_y = a.var(), b.var()
        sigma_xy = ((a - mu_x) * (b - mu_y)).mean()
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))/((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
        return {"ok": True, "psnr_db": float(psnr), "ssim": float(ssim)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# 自我檢測在這裡會輸出所有圖片相關的指標與數據，但通常不用那麼詳細所以前面才有個別勾選

@app.post("/self_test")
async def self_test(
    cover: UploadFile = File(...),
    secret: UploadFile = File(...),
    binarize: int = Form(0),   
    th: float = Form(0.5)      #二質化的坎
):
    
    try:
        cov_b = await cover.read()
        sec_b = await secret.read()
        if not cov_b or not sec_b:
            raise HTTPException(status_code=400, detail="empty file")

        cov = Image.open(io.BytesIO(cov_b)).convert("RGB")
        sec = Image.open(io.BytesIO(sec_b)).convert("RGB")

        Cs = app.state.cover_size
        Ss = app.state.secret_size

        # 控制縮放的一致
        cov_r = tv_resize(cov, (Cs, Cs), IM.BILINEAR)
        sec_r = tv_resize(sec, (Ss, Ss), IM.NEAREST)

        to_tensor = T.ToTensor()
        c = to_tensor(cov_r).unsqueeze(0).to(app.state.device)   #(1,3,Cs,Cs)
        s = to_tensor(sec_r).unsqueeze(0).to(app.state.device)   #(1,3,Ss,Ss)

       
        with torch.no_grad():
            stego, _ = app.state.model(c, s)                     # stego:(1,3,Cs,Cs)
            rec = app.state.model.decoder(stego).clamp(0, 1)     # rec:(1,3,Ss,Ss)

        
        c_down = F.interpolate(c, size=(Ss, Ss), mode="bilinear", align_corners=False)
        mse_to_cover  = F.mse_loss(rec, c_down).item()
        mse_to_secret = F.mse_loss(rec, s).item()

        
        def _ssim(x: torch.Tensor, y: torch.Tensor,
                  C1: float = 0.01**2, C2: float = 0.03**2) -> float:
            
           
            mu_x = x.mean(dim=(2, 3), keepdim=True)
            mu_y = y.mean(dim=(2, 3), keepdim=True)
            sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
            sigma_y = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
            sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)
            ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
                (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
            )
            return ssim_map.mean().item()

        # cover圖的指標
        mse_cover = F.mse_loss(stego, c).item()
        if mse_cover == 0:
            psnr_cover = float("inf")
        else:
            psnr_cover = 10 * math.log10(1.0 / mse_cover)
        ssim_cover = _ssim(stego, c)

        # secret圖的指標
        ssim_secret = _ssim(rec, s)

        # TAF/IoU 
        
        def _to_gray(x: torch.Tensor) -> torch.Tensor:
            # x:(N,3,H,W) in [0,1]數值預設
            return x.mean(dim=1, keepdim=True)

        def _bin(x: torch.Tensor, thr: float) -> torch.Tensor:
            return (x >= thr).float()

        rec_gray = _to_gray(rec)       
        gt_gray  = _to_gray(s)

        b_rec = _bin(rec_gray, th)
        b_gt  = _bin(gt_gray,  th)

        taf_acc = (b_rec == b_gt).float().mean().item()

        inter = (b_rec * b_gt).sum()
        union = ((b_rec + b_gt).clamp(max=1)).sum()
        iou = (inter / (union + 1e-8)).item()

        #搶先觀看
       
        if binarize:
            rec_disp = b_rec.repeat(1, 3, 1, 1)   # (1,3,H,W) 0/1
        else:
            rec_disp = rec

        rec_pil = T.ToPILImage()(rec_disp.squeeze(0).cpu())
        #放大
        scale = max(1, 256 // Ss)
        rec_pil = rec_pil.resize((rec_pil.width * scale, rec_pil.height * scale), Image.NEAREST)

        buf = io.BytesIO()
        rec_pil.save(buf, format="PNG")
        buf.seek(0)

        headers = {
            "X-CKPT-SIZES":       f"cover={Cs},secret={Ss}",
            "X-REC-MSE-COVER":    f"{mse_to_cover:.6f}",
            "X-REC-MSE-SECRET":   f"{mse_to_secret:.6f}",
            "X-REC-TAF":          f"{taf_acc:.4f}",
            "X-REC-IOU":          f"{iou:.4f}",
            "X-COVER-PSNR":       f"{psnr_cover:.3f}",
            "X-COVER-SSIM":       f"{ssim_cover:.4f}",
            "X-SECRET-SSIM":      f"{ssim_secret:.4f}",
            "X-SECRET-TAF":       f"{taf_acc:.4f}",
        }
        return StreamingResponse(buf, media_type="image/png", headers=headers)

    except HTTPException as e:
        return JSONResponse({"ok": False, "error": e.detail}, status_code=e.status_code)
    except Exception as e:
        logger.exception("SELF_TEST ERROR")
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)
