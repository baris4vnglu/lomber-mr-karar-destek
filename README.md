# Lomber MR Karar Destek (MVP)

Hekime yönelik, **araştırma amaçlı** bel MR karar destek aracı. Klinik kararın yerine geçmez.

```
MR (DICOM ZIP) ─► Görüntü modeli (LumbarNet, RSNA 2024) ─► seviye bazında darlık olasılıkları + Grad-CAM
Rapor + hasta bilgisi ─────────────────────────────────┐
                                                       ├─► Qwen (Ollama) + kılavuz maddeleri ─► hekim özeti
Kırmızı bayraklar ─► kural: yalnız acil sevk maddeleri ─┘
```

## Çalıştırma

Site sadece JPG/PNG MR resmi alır (en fazla 4). Qwen2.5-VL önce görüntü bulgularını yazar; belirgin bulgu varsa aynı model kılavuz maddelerine dayanarak hekime öneri yazar, yoksa sabit "radyolog değerlendirmesi" önerisi gösterilir. Model ekran kartına tam yüklenir (`num_gpu=99`); mr.jpeg analizi ~7 sn, öneri adımıyla ~20 sn. Yüzde/olasılık vermez.

```bash
ollama pull qwen2.5vl:3b
cd mr_asistan
..\venv\Scripts\python app.py
```
Tarayıcıda: http://localhost:8001

Eğitilmiş model yoksa site **yalnız rapor modunda** çalışır (sadece rapor metni + kılavuzlar).

## Görüntü modelini eğitme (Kaggle, ücretsiz GPU)

1. https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification → *Join / I Understand and Accept* (kuralları kabul edin).
2. Kaggle → *Create* → *Notebook* → *File* → *Import notebook* → `training/rsna_lumbar_train.ipynb`.
3. Sağ panel: *Add Input* → yarışma verisi, *Accelerator* → GPU T4, *Internet* → On.
4. *Run All* (önbellek ~20-30 dk + eğitim ~15 epoch).
5. *Output*'tan `rsna_lumbar.pth` ve `model_config.json` dosyalarını `mr_asistan/models/` içine koyun, sunucuyu yeniden başlatın.

Eğitim kodunu değiştirirseniz notebook'u yeniden üretin: `python training/build_notebook.py`.

## Dosyalar

| Dosya | Görev |
|---|---|
| `core/constants.py` | Seviyeler, bulgular, seri tipleri |
| `core/dicom_io.py` | DICOM okuma, kesit seçimi, seri tanıma (eğitim ve çıkarımda ortak) |
| `core/model.py` | 3 dallı EfficientNet-B0 modeli |
| `core/inference.py` | Kalibre olasılıklar, Grad-CAM |
| `core/advisor.py` | Kılavuz seçimi + Qwen istemi |
| `knowledge/guidelines.json` | NICE NG59 / NASS kılavuz özetleri (kendi cümlelerimizle) |
| `training/train_rsna.py` | Eğitim scripti; `build_notebook.py` bunu Kaggle notebook'una çevirir |

## Sınırlılıklar

- RSNA etiketleri **darlık derecesidir** (spinal kanal, foraminal, subartiküler); "fıtık var/yok" etiketi yoktur. Model herniasyonu doğrudan değil, ona bağlı darlıkları derecelendirir.
- Olasılıklar doğrulama setinde sıcaklık ölçekleme ile kalibre edilir; yine de başka cihaz/protokollerde performans düşebilir.
- Qwen 3B küçük bir modeldir; kılavuz metnini kopyalama eğilimi ve hata yapabilir. Kırmızı bayrak durumunda konservatif seçenekler modele hiç verilmez.
- Kılavuz özetleri hekim tarafından gözden geçirilmelidir. Gerçek hasta verisi için KVKK (anonimleştirme) ve tıbbi cihaz mevzuatı geçerlidir.
