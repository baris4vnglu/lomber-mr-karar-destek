"""Build a self-contained Kaggle notebook (rsna_lumbar_train.ipynb) from train_rsna.py.

The shared core code (constants, DICOM preprocessing, model) is inlined so the
notebook needs no extra uploads and stays identical to what inference uses.

    python training/build_notebook.py
"""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORE = HERE.parent / "core"
START, END = "# --- CORE IMPORTS", "# --- END CORE IMPORTS ---"


def strip_header(src: str) -> str:
    src = re.sub(r"^from __future__ import annotations\n", "", src, flags=re.M)
    # relative imports of sibling core modules (single- or multi-line)
    src = re.sub(r"^from \.\w+ import \([^)]*\)\n", "", src, flags=re.M)
    src = re.sub(r"^from \.\w+ import [^\n]*\n", "", src, flags=re.M)
    return src


def build() -> Path:
    train_src = (HERE / "train_rsna.py").read_text(encoding="utf-8")
    start = train_src.index(START)
    end = train_src.index(END) + len(END)
    core_src = "\n\n".join(
        f"# ===== core/{name} =====\n" + strip_header((CORE / name).read_text(encoding="utf-8"))
        for name in ("constants.py", "dicom_io.py", "model.py")
    )
    script = train_src[:start] + core_src + train_src[end:]

    intro = """# RSNA 2024 Lumbar Spine – LumbarNet eğitimi

**Kaggle ayarları (sağ panel):**
1. *Add Input* → **RSNA 2024 Lumbar Spine Degenerative Classification** yarışma verisini ekleyin (önce yarışma sayfasında kuralları kabul etmelisiniz).
2. *Accelerator* → **GPU T4 x2** veya **P100**.
3. *Internet* → **On** (ImageNet ön-eğitimli ağırlıklar için).

Çalıştırınca önce DICOM'lar `/kaggle/working/cache` içine önbelleğe alınır (~20–30 dk), sonra eğitim başlar (~15 epoch).
Bitince **Output** sekmesinden `rsna_lumbar.pth` ve `model_config.json` dosyalarını indirip `mr_asistan/models/` klasörüne koyun.

Hızlı deneme için son hücrede `sys.argv` satırına `--max-studies 100 --epochs 2` ekleyebilirsiniz."""

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)},
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": script.replace('if __name__ == "__main__":\n    main()\n', "").splitlines(keepends=True),
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": ["import sys\n", "sys.argv = ['train']  # ör: ['train', '--epochs', '15']\n", "main()\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = HERE / "rsna_lumbar_train.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(f"written: {build()}")
