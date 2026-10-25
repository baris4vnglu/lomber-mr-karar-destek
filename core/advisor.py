"""Doctor-facing summary: pick guideline snippets by rules, then let Qwen write the text.

The LLM never produces probabilities itself: it only rewrites the numbers from
the image model and cites the guideline snippets we hand it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import httpx

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
GUIDELINES_FILE = Path(__file__).resolve().parents[1] / "knowledge" / "guidelines.json"

RED_FLAGS_TR = {
    "bladder_bowel": "İdrar/gaita kontrol bozukluğu",
    "saddle_anesthesia": "Eyer anestezisi",
    "progressive_weakness": "İlerleyici motor kayıp",
    "systemic": "Ateş / kilo kaybı / kanser öyküsü",
}

SYSTEM_PROMPT = """Sen bir radyoloji/ortopedi karar destek asistanısın. Yalnızca HEKİME yazıyorsun, asla hastaya hitap etme.
Kurallar:
- Sadece sana verilen görüntü yorumunu, model olasılıklarını, rapor metnini ve kılavuz maddelerini kullan. Yeni olasılık veya sayı uydurma.
- Görüntü yorumu bulgu bildirmiyorsa veya görüntünün okunamadığını söylüyorsa, öneri yerine radyolog değerlendirmesi öner.
- Her öneriyi köşeli parantez içinde kılavuz kimliğiyle kaynakla, ör. [NICE-EXERCISE]. Listede olmayan kaynak verme.
- İlaç önerirken sadece kılavuz maddelerinde geçen ilaç gruplarını kullan, doz verme.
- Kırmızı bayrak varsa ilk cümlede acil değerlendirme gerektiğini belirt.
- Kesin tanı koyma; "düşündürür", "ile uyumlu" gibi ifadeler kullan. Klinik muayene ile doğrulanması gerektiğini hatırlat.
- Kılavuz metinlerini kopyalama; bu hastaya uyanları seç ve tek cümleyle yaz. En fazla 6 öneri.
- Görüntü modeli sayılarını tekrar yazma; onlar ayrıca gösteriliyor. Sadece yorumla.
- Türkçe, kısa ve maddeli yaz. Tam olarak şu şablonu kullan:

**Değerlendirme**
- <bulguların kısa klinik yorumu>

**Öneriler**
- <öneri> [KILAVUZ-ID]

Bu iki başlıktan sonra başka bölüm yazma."""


def load_guidelines() -> list[dict]:
    return json.loads(GUIDELINES_FILE.read_text(encoding="utf-8"))


VISION_KEYWORDS = {
    "radiculopathy": ("herni", "fıtık", "ekstrüzyon", "protrüzyon", "kök", "foramen", "foraminal"),
    "stenosis": ("darlık", "stenoz", "daralma"),
}
NEGATIONS = ("yok", "değil", "değerlendirilem", "saptanmadı", "izlenmedi", "görülmedi", "normal")


def positive_finding_tags(text: str) -> set[str]:
    """Tags for findings stated positively; "herniasyon yok / değerlendirilemiyor" does not count."""
    tags = set()
    for sentence in re.split(r"[.\n;]", text.lower()):
        if any(n in sentence for n in NEGATIONS):
            continue
        for tag, words in VISION_KEYWORDS.items():
            if any(w in sentence for w in words):
                tags.add(tag)
    return tags


def select_guidelines(
    findings: list[dict] | None, red_flags: list[str], radicular_pain: bool, vision_text: str = ""
) -> list[dict]:
    # Red flags override everything: the small LLM must not be handed conservative
    # options it could recommend instead of urgent referral.
    if red_flags:
        return [g for g in load_guidelines() if {"red_flag", "general"} & set(g["tags"])]
    tags = {"general", "conservative"} | positive_finding_tags(vision_text)
    if radicular_pain:
        tags.add("radiculopathy")
    for f in findings or []:
        if f["p_moderate_or_severe"] < 0.5:
            continue
        if f["condition"] in ("spinal_canal_stenosis", "left_subarticular_stenosis", "right_subarticular_stenosis"):
            tags.add("stenosis")
        if "foraminal" in f["condition"] or "subarticular" in f["condition"]:
            tags.add("radiculopathy")
    return [g for g in load_guidelines() if tags & set(g["tags"])]


def notable_findings(findings: list[dict], threshold: float = 0.3, minimum: int = 3, maximum: int = 5) -> list[dict]:
    ranked = sorted(findings, key=lambda f: f["p_moderate_or_severe"], reverse=True)
    picked = [f for f in ranked if f["p_moderate_or_severe"] >= threshold]
    return (picked if len(picked) >= minimum else ranked[:minimum])[:maximum]


def build_user_prompt(
    findings: list[dict] | None,
    model_info: dict,
    report_text: str,
    patient: dict,
    red_flags: list[str],
    guidelines: list[dict],
    vision_text: str = "",
) -> str:
    parts = []
    p = patient
    parts.append(
        "HASTA: "
        f"yaş={p.get('age') or 'belirtilmedi'}, cinsiyet={p.get('sex') or 'belirtilmedi'}, "
        f"şikâyet süresi={p.get('duration') or 'belirtilmedi'}, "
        f"bacağa yayılan ağrı={'evet' if p.get('radicular_pain') else 'hayır/belirtilmedi'}, "
        f"not={p.get('notes') or '-'}"
    )
    parts.append("KIRMIZI BAYRAKLAR: " + (", ".join(RED_FLAGS_TR[r] for r in red_flags) if red_flags else "yok"))

    if findings:
        lines = [
            f"- {f['level_label']} {f['condition_tr']}: orta/ağır olasılığı %{f['p_moderate_or_severe'] * 100:.0f} "
            f"(orta %{f['p_moderate'] * 100:.0f}, ağır %{f['p_severe'] * 100:.0f})"
            for f in notable_findings(findings)
        ]
        parts.append(
            "GÖRÜNTÜ MODELİ SONUÇLARI (RSNA 2024 ile eğitilmiş deneysel model, kalibre olasılıklar; "
            f"doğrulama ağırlıklı log-loss={model_info.get('val_weighted_logloss')}):\n" + "\n".join(lines)
        )
    else:
        parts.append("GÖRÜNTÜ MODELİ SONUÇLARI: yok (model yüklü değil veya MR yüklenmedi).")

    if vision_text.strip():
        parts.append(
            "GÖRSEL DİL MODELİ YORUMU (yüklenen JPG/PNG kesitlerden, DÜŞÜK GÜVENİLİRLİK; "
            "raporla çelişirse rapora öncelik ver):\n" + vision_text.strip()
        )
    parts.append("RADYOLOJİ RAPORU:\n" + (report_text.strip() or "(rapor girilmedi)"))
    parts.append("KILAVUZ MADDELERİ:\n" + "\n".join(f"[{g['id']}] ({g['source']}) {g['text']}" for g in guidelines))
    if red_flags:
        parts.append(
            "ÖNEMLİ: Kırmızı bayrak mevcut. Öneriler bölümünde yalnızca acil değerlendirme/sevk öner; "
            "egzersiz, ilaç veya bekleme önerme."
        )
    parts.append("Bu bilgilere göre hekime yönelik karar destek özetini yaz.")
    return "\n\n".join(parts)


async def write_summary(user_prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}],
        "stream": False,
        "keep_alive": "30m",
        # Default context (2k) silently truncates the prompt start, dropping the system rules.
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
        resp = await client.post(OLLAMA_URL, json=payload)
        resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def findings_markdown(findings: list[dict] | None) -> str:
    """Model numbers are written by code, never by the LLM, so they cannot be mislabelled."""
    if not findings:
        return "**Görüntü bulguları (model)**\n- Model sonucu yok (görüntü modeli yüklü değil veya MR yüklenmedi)."
    lines = [
        f"- {f['level_label']} {f['condition_tr']}: orta/ağır olasılığı %{f['p_moderate_or_severe'] * 100:.0f}"
        for f in notable_findings(findings)
    ]
    return "**Görüntü bulguları (model)**\n" + "\n".join(lines)
