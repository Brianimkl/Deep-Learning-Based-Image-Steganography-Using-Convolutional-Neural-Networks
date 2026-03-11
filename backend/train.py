
import argparse, random, io
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from tqdm import tqdm

from .model import StegoSystem
from .utils import to_pil  

torch.backends.cudnn.benchmark = True

#資料及
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class MyImageFolder(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.paths = sorted(
            [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
        )
        if not self.paths:
            raise RuntimeError(f"[MyImageFolder] No images found under {root}")
        print(f"[MyImageFolder] {root} -> {len(self.paths)} images")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return img, 0  


# ===================== Dataset =====================
class PairDataset(Dataset):
    def __init__(self, ds_cover, ds_secret, cover_size=128, secret_size=32):
        self.ds_cover = ds_cover
        self.ds_secret = ds_secret
        self.t_cover = T.Compose([
            T.Resize((cover_size, cover_size)),
            T.ToTensor()
        ])
        self.t_secret = T.Compose([
            T.Resize((secret_size, secret_size), interpolation=Image.NEAREST),
            T.ToTensor()
        ])

    def __len__(self):
        return max(len(self.ds_cover), len(self.ds_secret))

    def __getitem__(self, i):
        c = self.ds_cover[i % len(self.ds_cover)][0]
        s = self.ds_secret[random.randrange(0, len(self.ds_secret))][0]
        return self.t_cover(c), self.t_secret(s)


def make_loaders(cover_size=128, secret_size=32, batch=64):
    #先用兩基礎資料及訓練STL10 / CIFAR10
    ds_cover_base  = datasets.STL10('data', split='train', download=True, transform=None)
    ds_secret_base = datasets.CIFAR10('data', train=True,    download=True, transform=None)

    ds_cover  = ds_cover_base
    ds_secret = ds_secret_base

    #後面加上自然圖片與額外資料及
    cover_extra_root  = Path("data/my_covers")
    secret_extra_root = Path("data/my_secrets")

    if cover_extra_root.exists():
        try:
            ds_cover_extra = MyImageFolder(cover_extra_root)
            ds_cover = ConcatDataset([ds_cover_base, ds_cover_extra])
            print(f"[make_loaders] cover: STL10 + my_covers, total={len(ds_cover)}")
        except RuntimeError as e:
            print(f"[make_loaders] my_covers skipped: {e}")

    if secret_extra_root.exists():
        try:
            ds_secret_extra = MyImageFolder(secret_extra_root)
            ds_secret = ConcatDataset([ds_secret_base, ds_secret_extra])
            print(f"[make_loaders] secret: CIFAR10 + my_secrets, total={len(ds_secret)}")
        except RuntimeError as e:
            print(f"[make_loaders] my_secrets skipped: {e}")

    full = PairDataset(ds_cover, ds_secret, cover_size, secret_size)

    n = len(full)
    n_tr = int(n * 0.9)
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(n, generator=g).tolist()
    train = Subset(full, idx[:n_tr])
    valid = Subset(full, idx[n_tr:])

    train_loader = DataLoader(
        train, batch_size=batch, shuffle=True,
        num_workers=0, pin_memory=True
    )
    valid_loader = DataLoader(
        valid, batch_size=batch, shuffle=False,
        num_workers=0, pin_memory=True
    )
    return train_loader, valid_loader

#損術設計
class VGGPerceptual(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_FEATURES).features
        self.slice = nn.Sequential(*[vgg[i] for i in range(16)])  # 到 relu3_3
        for p in self.slice.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        fx = self.slice(x); fy = self.slice(y)
        return F.l1_loss(fx, fy)

def sobel_xy(x: torch.Tensor):
    ky = torch.tensor([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    kx = torch.tensor([[1, 0, -1],
                       [2, 0, -2],
                       [1, 0, -1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    weight_x = kx.repeat(x.size(1), 1, 1, 1)  # (C,1,3,3)
    weight_y = ky.repeat(x.size(1), 1, 1, 1)
    grad_x = F.conv2d(x, weight_x, padding=1, groups=x.size(1))
    grad_y = F.conv2d(x, weight_y, padding=1, groups=x.size(1))
    mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6).mean(dim=1, keepdim=True)
    return mag

def ssim_simple(x, y, C1=0.01**2, C2=0.03**2):
    mu_x = x.mean(dim=(2,3), keepdim=True)
    mu_y = y.mean(dim=(2,3), keepdim=True)
    sigma_x = ((x - mu_x)**2).mean(dim=(2,3), keepdim=True)
    sigma_y = ((y - mu_y)**2).mean(dim=(2,3), keepdim=True)
    sigma_xy = ((x - mu_x)*(y - mu_y)).mean(dim=(2,3), keepdim=True)
    ssim_map = ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))
    return ssim_map.mean()

def tv_loss(img):
   #雜點
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return (dx.abs().mean() + dy.abs().mean())

# Robustness 
def jpeg_compress_tensor(batch_img: torch.Tensor, quality: int) -> torch.Tensor:
    
    outs = []
    for t in batch_img:
        pil = to_pil(t)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        pil2 = Image.open(buf).convert("RGB")
        outs.append(T.ToTensor()(pil2))
    return torch.stack(outs, dim=0)

def add_noise(x: torch.Tensor, std=0.0):
    if std <= 0:
        return x
    return (x + torch.randn_like(x)*std).clamp(0,1)

#參數
def train(
    epochs=20,
    cover_size=128,
    secret_size=64,
    batch_size=32,
    lr=1e-3,
    lambda_cover=5.0,
    lambda_secret=6.0,
    lambda_perc=0.2,
    lambda_secret_ssim=0.30,
    lambda_secret_edge=0.15,
    jpeg_prob=0.2,
    jpeg_min=40,
    jpeg_max=95,
    noise_std=0.02,
    outdir='backend/runs'
):
    """
    前半段先讓模型學會怎麼去藏而後半段呢開始加入魯棒式訓練(英文字太多我打中文，後面也把secret的指數慢慢拉高，不要魯棒跟拉高同時一起會崩很快，最後收斂，secret的指數不是比重所以合起來不是
    1沒關係，因為我是用所有的損失函數乘上他們各自的指數下去做最後的數值，報告書裡面應該有寫
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    train_loader, valid_loader = make_loaders(cover_size, secret_size, batch_size)
    model = StegoSystem(secret_size=secret_size).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    perc  = VGGPerceptual().to(device)

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best.pt"
    best_val  = float('inf')

    warm_ratio = 0.30     #前半段不做jpg訓練
    lam_ssim_cov = 0.10   #指數權重
    lam_tv       = 1e-5

    for ep in range(1, epochs+1):
        progress = (ep - 1) / max(1, epochs)
        use_jpeg = (progress >= warm_ratio and jpeg_prob > 0)

        #把secret的指數拉高來訓練
        sec_scale   = 0.6 + 0.4 * max(0.0, (progress - warm_ratio) / (1 - warm_ratio))
        lam_cov     = float(lambda_cover)
        lam_sec     = float(lambda_secret) * sec_scale
        lam_per     = float(lambda_perc)
        lam_sec_jpg = lam_sec * 0.5  #這裡jpg的處理只針對decoder

        # 訓練
        model.train()
        tr_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        for cover, secret in pbar:
            cover, secret = cover.to(device), secret.to(device)

            #encoder加上微調去確保梯度
            stego, _ = model(cover, secret)
            stego_for_dec = add_noise(stego, std=noise_std)
            rec = model.decoder(stego_for_dec)

            #把secret重建的地方
            loss_secret_mse  = F.mse_loss(rec, secret)
            loss_secret_ssim = 1.0 - ssim_simple(rec, secret)
            loss_secret_edge = F.l1_loss(sobel_xy(rec), sobel_xy(secret))

            #cover端的
            loss_cover_mse  = F.mse_loss(stego, cover)
            loss_cover_ssim = 1.0 - ssim_simple(stego, cover)
            loss_tv_val     = tv_loss(stego)
            loss_perc_val   = perc(stego, cover)

            #訓練後期加入魯棒式訓練
            loss_secret_jpg = torch.tensor(0.0, device=device)
            if use_jpeg and random.random() < jpeg_prob:
                with torch.no_grad():
                    st_det = stego.detach()
                    q = random.randint(jpeg_min, jpeg_max)
                    st_jpeg = jpeg_compress_tensor(st_det, q).to(device)
                st_jpeg = add_noise(st_jpeg, std=noise_std)
                rec_jpeg = model.decoder(st_jpeg)
                loss_secret_jpg = F.mse_loss(rec_jpeg, secret)

            loss = (
                lam_cov     * loss_cover_mse   +
                lam_per     * loss_perc_val    +
                lam_ssim_cov* loss_cover_ssim  +
                lam_tv      * loss_tv_val      +
                lam_sec     * loss_secret_mse  +
                lambda_secret_ssim * loss_secret_ssim +
                lambda_secret_edge * loss_secret_edge +
                lam_sec_jpg * loss_secret_jpg
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            tr_loss += loss.item() * cover.size(0)
            pbar.set_postfix({
                "Lcov":  float(loss_cover_mse),
                "Lsec":  float(loss_secret_mse),
                "LsecJ": float(loss_secret_jpg),
                "Lperc": float(loss_perc_val),
                "LssimC":float(loss_cover_ssim),
                "Ltv":   float(loss_tv_val)
            })

        tr_loss /= len(train_loader.dataset)

        #vaild
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for cover, secret in valid_loader:
                cover, secret = cover.to(device), secret.to(device)
                stego, _ = model(cover, secret)

                rec = model.decoder(stego)
                loss_secret_mse  = F.mse_loss(rec, secret)
                loss_secret_ssim = 1.0 - ssim_simple(rec, secret)
                loss_secret_edge = F.l1_loss(sobel_xy(rec), sobel_xy(secret))

                loss_cover_mse  = F.mse_loss(stego, cover)
                loss_cover_ssim = 1.0 - ssim_simple(stego, cover)
                loss_tv_val     = tv_loss(stego)
                loss_perc_val   = perc(stego, cover)

                loss_secret_jpg = torch.tensor(0.0, device=device)
                if use_jpeg and jpeg_prob > 0:
                    q = random.randint(jpeg_min, jpeg_max)
                    st_jpeg = jpeg_compress_tensor(stego, q).to(device)
                    rec_jpeg = model.decoder(st_jpeg)
                    loss_secret_jpg = F.mse_loss(rec_jpeg, secret)

                loss = (
                    lam_cov     * loss_cover_mse   +
                    lam_per     * loss_perc_val    +
                    lam_ssim_cov* loss_cover_ssim  +
                    lam_tv      * loss_tv_val      +
                    lam_sec     * loss_secret_mse  +
                    lambda_secret_ssim * loss_secret_ssim +
                    lambda_secret_edge * loss_secret_edge +
                    lam_sec_jpg * loss_secret_jpg
                )
                va_loss += loss.item() * cover.size(0)

        va_loss /= len(valid_loader.dataset)
        print(f"[Epoch {ep}] train={tr_loss:.4f}  valid={va_loss:.4f}  "
              f"(use_jpeg={use_jpeg}, sec_scale={sec_scale:.2f})")

        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "args": {"secret_size": secret_size, "cover_size": cover_size}
            }, best_path)
            print(f"  -> Saved best to {best_path} (val={best_val:.4f})")

    return str(best_path)

#clt
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",       type=int,   default=20)
    ap.add_argument("--cover-size",   type=int,   default=128)
    ap.add_argument("--secret-size",  type=int,   default=64)
    ap.add_argument("--batch-size",   type=int,   default=32)
    ap.add_argument("--lr",           type=float, default=1e-3)

    ap.add_argument("--lambda-cover", type=float, default=5.0)
    ap.add_argument("--lambda-secret",type=float, default=6.0)
    ap.add_argument("--lambda-perc",  type=float, default=0.2)
    ap.add_argument("--lambda-secret-ssim", type=float, default=0.30)
    ap.add_argument("--lambda-secret-edge", type=float, default=0.15)

    ap.add_argument("--jpeg-prob",    type=float, default=0.2)
    ap.add_argument("--jpeg-min",     type=int,   default=40)
    ap.add_argument("--jpeg-max",     type=int,   default=95)
    ap.add_argument("--noise-std",    type=float, default=0.02)

    ap.add_argument("--outdir",       type=str,   default="backend/runs")
    args = ap.parse_args()

    path = train(
        epochs=args.epochs,
        cover_size=args.cover_size,
        secret_size=args.secret_size,
        batch_size=args.batch_size,
        lr=args.lr,
        lambda_cover=args.lambda_cover,
        lambda_secret=args.lambda_secret,
        lambda_perc=args.lambda_perc,
        lambda_secret_ssim=args.lambda_secret_ssim,
        lambda_secret_edge=args.lambda_secret_edge,
        jpeg_prob=args.jpeg_prob,
        jpeg_min=args.jpeg_min,
        jpeg_max=args.jpeg_max,
        noise_std=args.noise_std,
        outdir=args.outdir
    )
    print("Best saved to:", path)
