const API = "http://127.0.0.1:8000";

const $ = (id) => document.getElementById(id);
const blobToDataURL = (b) =>
  new Promise((r) => {
    const fr = new FileReader();
    fr.onload = () => r(fr.result);
    fr.readAsDataURL(b);
  });

$("cover").addEventListener("change", () => {
  const f = $("cover").files[0];
  if (!f) return;
  $("preview-cover").src = URL.createObjectURL(f);
});

$("secret").addEventListener("change", () => {
  const f = $("secret").files[0];
  if (!f) return;
  $("preview-secret").src = URL.createObjectURL(f);
});

$("stego").addEventListener("change", () => {
  const f = $("stego").files[0];
  if (!f) return;
  $("preview-stego2").src = URL.createObjectURL(f);
});

$("btn-encode").onclick = async () => {
  const cover = $("cover").files[0];
  const secret = $("secret").files[0];
  if (!cover || !secret) {
    alert("請先選擇 Cover 與 Secret");
    return;
  }

  const fd = new FormData();
  fd.append("cover", cover);
  fd.append("secret", secret);

  const res = await fetch(`${API}/encode`, { method: "POST", body: fd });
  const blob = await res.blob();
  $("preview-stego").src = await blobToDataURL(blob);

  
  const stegoFile = new File([blob], "stego.png", { type: "image/png" });
  const dt = new DataTransfer();
  dt.items.add(stegoFile);
  $("stego").files = dt.files;
  $("preview-stego2").src = $("preview-stego").src;
};

$("btn-decode").onclick = async () => {
  const stego = $("stego").files[0];
  if (!stego) {
    alert("請先上傳 stego 圖片");
    return;
  }

  const fd = new FormData();
  fd.append("stego", stego);

  const res = await fetch(`${API}/decode`, { method: "POST", body: fd });
  const blob = await res.blob();
  const recoveredUrl = await blobToDataURL(blob);

  const imgRecovered = $("preview-recovered");
  imgRecovered.src = recoveredUrl;
  imgRecovered.classList.remove("hidden");

 
  const imgSecret = $("preview-secret");
  if (imgSecret && imgSecret.complete && imgSecret.naturalWidth > 0) {
 
    const rect = imgSecret.getBoundingClientRect();
    imgRecovered.style.width = rect.width + "px";
    imgRecovered.style.height = rect.height + "px";
  } else {
   
    imgRecovered.style.width = "256px";
    imgRecovered.style.height = "256px";
  }
};

$("btn-train").onclick = async () => {
  $("train-log").textContent = "訓練中…（終端機也會看到訓練列印）";
  const res = await fetch(`${API}/train?epochs=3`, { method: "POST" });
  const json = await res.json();
  $("train-log").textContent = `完成：${JSON.stringify(
    json,
    null,
    2
  )}\n重新整理頁面後即可使用最新模型。`;
};
