![Sentiment Analysis](background.jpeg)
# Analisis Sentimen Ulasan (ID)

_Pembaruan dari proyek **Machine-Learning-For-Indonesian-Text-Sentiment-Classification**._

Aplikasi ini mengklasifikasikan ulasan berbahasa Indonesia menjadi **negative / neutral / positive**. Fokusnya membantu tim **Quality Assurance / Support** menangkap keluhan lebih cepat, dengan antarmuka yang sederhana untuk uji teks tunggal maupun unggah file.

---

## Fitur Singkat
- **Split berbasis grup (GroupSplit)** per `product_name` untuk menghindari kebocoran data.
- **Fitur robust**: TF-IDF word (1–2) + char (3–5); **LinearSVC** + kalibrasi probabilitas.
- **Ambang khusus “negative”**: profil **Seimbang**, **Tanggap Keluhan**, atau **Kustom**.
- **Normalisasi percakapan** (mis. `kurang bagus → kurang_bagus`, `slow respon → slow_respon`) dan penanganan negasi.
- **Streamlit app**: uji teks tunggal, unggah **CSV/Excel**, tampilkan probabilitas per kelas & token kosakata yang terdeteksi.

---

## Format Data (Minimal)
- `product_name` – identitas produk (untuk GroupSplit)  
- `review_text` – teks ulasan  
- `sentiment` – label: **0=negative**, **1=neutral**, **2=positive**

---

## Cara Menjalankan
```bash
pip install -r requirements.txt
# pastikan artifact model tersedia:
# sentiment_pipeline_group.joblib  (di root /models /artifacts)
streamlit run app.py
```

**Developed by Akmal Falah Darmawan**  
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sentiment-analysis-on-social-reviews-eyfpvdqq5dycv9qt4iroed.streamlit.app/)

