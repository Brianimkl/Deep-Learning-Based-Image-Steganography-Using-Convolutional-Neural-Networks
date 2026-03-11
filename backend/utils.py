
import io, math, random
from typing import Tuple
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T



def from_pil(img: Image.Image) -> torch.Tensor:
    """
    PIL.Image -> tensor(C,H,W) in [0,1]
    """
    arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)

def to_pil(t: torch.Tensor) -> Image.Image:
    """
    tensor(C,H,W) in [0,1] 或 tensor(H,W) -> PIL.Image (RGB)
    """
    t = t.detach().cpu().clamp(0, 1)
    if t.dim() == 3:
        c, h, w = t.shape
        if c == 1:
            arr = (t.squeeze(0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGB")
        else:
            arr = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
    elif t.dim() == 2:
        arr = (t.numpy() * 255).astype(np.uint8)
        return Image.fromarray(arr, mode="L").convert("RGB")
    else:
        raise ValueError("to_pil expects 2D/3D tensor")



def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Peak Signal-to-Noise Ratio
    """
    mse = F.mse_loss(a, b, reduction='mean').item()
    if mse == 0:
        return 99.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))

def jpeg_compress_tensor(img_t: torch.Tensor, quality: int) -> torch.Tensor:
    
    pil = to_pil(img_t)
    buf = io.BytesIO()
    pil.save(buf, format='JPEG', quality=quality, subsampling=0)
    buf.seek(0)
    out = Image.open(buf).convert('RGB')
    return from_pil(out)

#做resize

def tv_resize(img: Image.Image, size: Tuple[int, int], interp=Image.BILINEAR) -> Image.Image:
    
    return img.resize(size, interp)

# secret圖做前處理

def preprocess_secret(
    img: Image.Image,
    size: int,
    bg: str = "white",          
    keep_aspect: bool = True,   #要不要保留原本比例
    binarize: bool = False,     #要不要二值化
    th: float = 0.55,           #二值化檻
) -> Image.Image:
    """
    把任意大小 / 可能帶 alpha 的 secret 圖，轉成 size×size 的 PIL.Image：
      1. RGBA -> RGB，依照 bg 填白 / 黑底
      2. 若 keep_aspect=True：長邊縮放到 size，短邊等比例，居中貼到背景
         若 keep_aspect=False：直接拉成 size×size
      3. 若 binarize=True：轉灰階、依 th 做 threshold，再轉回 RGB
    最後回傳 PIL.Image (RGB)，後面會再交給 T.ToTensor() 做 tensor 化。
    """
    # 在座透明度的處理
    if img.mode == "RGBA":
        if bg == "black":
            bg_color = (0, 0, 0)
        else:
            bg_color = (255, 255, 255)
        base = Image.new("RGB", img.size, bg_color)
        base.paste(img, mask=img.split()[-1])  #用alpha做遮罩
        img = base
    else:
        img = img.convert("RGB")

    #調尺寸
    if keep_aspect:
        
        w, h = img.size
        if w >= h:
            new_w = size
            new_h = int(h * size / max(w, 1))
        else:
            new_h = size
            new_w = int(w * size / max(h, 1))
        img_resized = img.resize((new_w, new_h), Image.NEAREST)

       
        if bg == "black":
            bg_color = (0, 0, 0)
        else:
            bg_color = (255, 255, 255)
        canvas = Image.new("RGB", (size, size), bg_color)
        offset = ((size - new_w) // 2, (size - new_h) // 2)
        canvas.paste(img_resized, offset)
        img_out = canvas
    else:
      
        img_out = img.resize((size, size), Image.NEAREST)

    #二質化(可以選擇不要都行)
    if binarize:
        g = img_out.convert("L")
        # th界在00~1之間所以之後成上255就是原本的
        g = g.point(lambda v: 255 if v >= int(th * 255) else 0)
        img_out = g.convert("RGB")

    return img_out
