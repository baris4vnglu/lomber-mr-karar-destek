"""Lomber MR Karar Destek – doktor web arayüzü.

    cd mr_asistan
    ..\\venv\\Scripts\\python app.py      ->  http://localhost:8001
"""

import os
import shutil
import tempfile
import zipfile
from pathlib import Path

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from core import advisor, vision
from core.constants import SERIES_ROLE_LABELS, SERIES_ROLES
from core.dicom_io import describe_series, group_by_series, load_study_stacks
from core.inference import Predictor

BASE_DIR = Path(__file__).resolve().parent
predictor = Predictor(Path(os.getenv("MR_MODELS_DIR", BASE_DIR / "models")))

app = FastAPI(title="Lomber MR Karar Destek")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.on_event("startup")
async def warm_up_vision_model():
    await vision.warm_up()


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/status")
def status():
    return {"model": predictor.model_info(), "llm": advisor.OLLAMA_MODEL, "vision_llm": vision.VISION_MODEL}


def _safe_extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest.resolve()):  # zip-slip guard
                raise HTTPException(400, "ZIP içinde geçersiz dosya yolu var.")
        zf.extractall(dest)


def _analyze_images(study_dir: Path) -> dict:
    groups = group_by_series(study_dir)
    if not groups:
        raise HTTPException(400, "ZIP içinde okunabilir DICOM dosyası bulunamadı.")
    series = describe_series(groups)

    # Auto-map: for each role take the matching series with the most slices.
    mapping: dict[str, str] = {}
    for s in sorted(series, key=lambda s: -s["n_files"]):
        if s["guessed_role"] and s["guessed_role"] not in mapping:
            mapping[s["guessed_role"]] = s["series_uid"]
    missing = [SERIES_ROLE_LABELS[r] for r in SERIES_ROLES if r not in mapping]

    stacks = load_study_stacks({role: groups[uid]["files"] for role, uid in mapping.items()})
    result = {
        "series": series,
        "missing_series": missing,
        "previews": predictor.series_previews(stacks),
        "findings": None,
        "heatmaps": [],
    }
    if predictor.ready:
        findings = predictor.predict(stacks)
        result["findings"] = findings
        top = sorted(findings, key=lambda f: f["p_moderate_or_severe"], reverse=True)[:3]
        result["heatmaps"] = [predictor.gradcam(stacks, f["target"]) for f in top]
    return result


@app.post("/api/analyze")
async def analyze(
    file: UploadFile | None = File(None),
    report_text: str = Form(""),
    age: str = Form(""),
    sex: str = Form(""),
    duration: str = Form(""),
    radicular_pain: bool = Form(False),
    notes: str = Form(""),
    red_flags: list[str] = Form([]),
):
    red_flags = [r for r in red_flags if r in advisor.RED_FLAGS_TR]
    if (file is None or not file.filename) and not report_text.strip():
        raise HTTPException(400, "En azından bir MR (ZIP) veya rapor metni girin.")

    image_result = {"series": [], "missing_series": [], "previews": [], "findings": None, "heatmaps": []}
    if file is not None and file.filename:
        tmp = Path(tempfile.mkdtemp(prefix="mr_"))
        try:
            zip_path = tmp / "study.zip"
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            if not zipfile.is_zipfile(zip_path):
                raise HTTPException(400, "Yüklenen dosya ZIP değil. DICOM klasörünü ZIP'leyip yükleyin.")
            study_dir = tmp / "study"
            study_dir.mkdir()
            _safe_extract(zip_path, study_dir)
            image_result = await run_in_threadpool(_analyze_images, study_dir)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    patient = {"age": age, "sex": sex, "duration": duration, "radicular_pain": radicular_pain, "notes": notes}
    guidelines = advisor.select_guidelines(image_result["findings"], red_flags, radicular_pain)
    prompt = advisor.build_user_prompt(
        image_result["findings"], predictor.model_info(), report_text, patient, red_flags, guidelines
    )
    try:
        summary, llm_error = await advisor.write_summary(prompt), None
    except httpx.HTTPError as e:
        summary, llm_error = "", f"Ollama'ya ulaşılamadı veya hata döndü ({e.__class__.__name__}). 'ollama serve' çalışıyor mu?"

    return {
        **image_result,
        "model": predictor.model_info(),
        "red_flags": [advisor.RED_FLAGS_TR[r] for r in red_flags],
        "guidelines": guidelines,
        "summary": (
            advisor.findings_markdown(image_result["findings"]) + "\n\n" + summary if image_result["findings"] else summary
        ),
        "llm_error": llm_error,
    }


@app.get("/api/guidelines")
def guidelines():
    return advisor.load_guidelines()


@app.post("/api/analyze-images")
async def analyze_images(images: list[UploadFile] = File(...)):
    """JPG/PNG -> one Qwen2.5-VL call (analysis + advice), streamed as plain text."""
    images = [im for im in images if im.filename]
    if not images:
        raise HTTPException(400, "Lütfen en az bir resim seçin.")
    if len(images) > vision.MAX_IMAGES:
        raise HTTPException(400, f"En fazla {vision.MAX_IMAGES} resim yükleyebilirsiniz.")
    try:
        prepared = [vision.prepare_image(await im.read()) for im in images]
    except vision.ImageError as e:
        raise HTTPException(400, str(e))

    async def body():
        try:
            async for chunk in vision.stream_analysis(prepared):
                yield chunk
        except httpx.HTTPError as e:
            yield (
                f"\n\n[HATA] Görsel model ({vision.VISION_MODEL}) çalıştırılamadı ({e.__class__.__name__}). "
                f"'ollama serve' açık mı, 'ollama pull {vision.VISION_MODEL}' yapıldı mı?"
            )

    return StreamingResponse(body(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8001, reload=False)
