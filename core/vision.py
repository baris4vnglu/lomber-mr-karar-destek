"""JPG/PNG MR analysis with Qwen2.5-VL, streamed.

Two short calls on the SAME model (no GPU model swap):
  1. image -> findings
  2. findings + rule-selected guidelines -> advice (text only)
If step 1 states no positive finding, step 2 is skipped and a fixed referral
line is used, so the small model can never suggest surgery for a normal scan.
No probabilities here: the trained image model needs full DICOM series.
"""

from __future__ import annotations

import base64
import io
import json
import os
from collections.abc import AsyncIterator

import httpx
from PIL import Image, UnidentifiedImageError

from .advisor import OLLAMA_URL, positive_finding_tags, select_guidelines

VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "qwen2.5vl:3b")
MAX_IMAGES = 4
MAX_SIDE = 512  # enough for lumbar MR detail; fewer image tokens = faster
KEEP_ALIVE = "60m"
# Ollama's own estimate leaves most layers on CPU on a 4 GB card; the 3B model
# fits fully (~2.7 GB at ctx 2048), which is ~25% faster than the default.
OPTIONS = {"temperature": 0.1, "num_ctx": 2048, "num_gpu": int(os.getenv("OLLAMA_NUM_GPU", "99"))}

ANALYSIS_PROMPT = """Görüntü(ler) bir hastanın lomber (bel) MR kesitleridir. Bir HEKİM için sadece görüntü bulgularını yaz.
- Görüntü bel MR'ı değilse veya okunamıyorsa sadece bunu söyle.
- Sekans ipucu: BOS (beyin omurilik sıvısı) parlak/beyazsa T2, koyuysa T1.
- Her disk seviyesi için tek kısa madde: disk yüksekliği/sinyali, herniasyon, kanal veya foramen darlığı. Normalse "normal" yaz.
- Emin değilsen "net değerlendirilemiyor" yaz. Yüzde, tanı veya tedavi yazma.
Türkçe, sadece şu şablon:
**Görüntü analizi**
- Kesit/sekans: <sagittal/aksiyel>, <T1/T2>
- <seviye>: <bulgu>"""

ADVICE_PROMPT = """Aşağıdaki MR görüntü analizine göre HEKİME en fazla 4 kısa öneri yaz. Hastaya hitap etme.
- SADECE verilen kılavuz maddelerini kullan, her önerinin sonuna kimliğini köşeli parantezle yaz, ör. [NICE-EXERCISE].
- Kılavuz metnini kopyalama; her öneri tek kısa cümle. İlaç dozu verme. Kesin tanı koyma.
- Sadece maddeler yaz, başlık veya açıklama ekleme.

GÖRÜNTÜ ANALİZİ:
{analysis}

KILAVUZ MADDELERİ:
{guidelines}"""

NO_FINDING_ADVICE = (
    "- Görüntü analizinde belirgin patolojik bulgu tanımlanmadı veya bulgular net değerlendirilemedi; "
    "radyolog değerlendirmesi ve klinik muayene ile korelasyon önerilir [IMAGING-CORRELATION]\n"
    "- Tedavi kararı görüntüye değil, klinik tablo ile görüntülemenin uyumuna göre verilmelidir [IMAGING-CORRELATION]"
)


class ImageError(ValueError):
    pass


def prepare_image(data: bytes) -> str:
    """Validate and downscale an upload; returns base64 PNG."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise ImageError("Yüklenen dosyalardan biri geçerli bir JPG/PNG değil.")
    img = img.convert("L").convert("RGB")  # MR is grayscale; drops alpha/odd modes
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _stream_chat(client: httpx.AsyncClient, message: dict, num_predict: int) -> AsyncIterator[str]:
    body = {
        "model": VISION_MODEL,
        "messages": [message],
        "stream": True,
        "keep_alive": KEEP_ALIVE,
        "options": {**OPTIONS, "num_predict": num_predict},
    }
    async with client.stream("POST", OLLAMA_URL, json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            text = chunk.get("message", {}).get("content", "")
            if text:
                yield text
            if chunk.get("done"):
                break


async def stream_analysis(images_b64: list[str]) -> AsyncIterator[str]:
    """Yield the analysis, then the advice, as the model writes them."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0)) as client:
        analysis = ""
        async for piece in _stream_chat(client, {"role": "user", "content": ANALYSIS_PROMPT, "images": images_b64}, 300):
            analysis += piece
            yield piece

        yield "\n\n**Hekime öneriler**\n"
        if not positive_finding_tags(analysis):
            yield NO_FINDING_ADVICE
            return
        guidelines = select_guidelines(None, [], False, analysis)
        prompt = ADVICE_PROMPT.format(
            analysis=analysis.strip(), guidelines="\n".join(f"[{g['id']}] {g['text']}" for g in guidelines)
        )
        async for piece in _stream_chat(client, {"role": "user", "content": prompt}, 400):
            yield piece


async def warm_up() -> None:
    """Load the model into GPU memory at startup so the first analysis does not pay the load time."""
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            await client.post(
                OLLAMA_URL.replace("/api/chat", "/api/generate"),
                json={"model": VISION_MODEL, "keep_alive": KEEP_ALIVE, "options": OPTIONS},
            )
    except httpx.HTTPError:
        pass
