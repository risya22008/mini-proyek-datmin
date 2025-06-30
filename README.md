# 🖼️ Image Segmentation App - KMeans Clustering

Aplikasi berbasis Streamlit untuk melakukan segmentasi gambar menggunakan algoritma **K-Means Clustering** dan mengevaluasi hasilnya dengan **Silhouette Score**. Cocok untuk eksplorasi visual dalam pemrosesan citra, clustering, dan computer vision.

---

## 🚀 Fitur Utama

-   ✅ Upload banyak gambar (`.jpg`, `.png`, `.jpeg`).
-   ✅ Proses segmentasi otomatis dengan **K-Means**.
-   ✅ Evaluasi kualitas cluster menggunakan **Silhouette Score**.
-   ✅ Tampilkan segmentasi terbaik beserta semua variasi jumlah klaster.
-   ✅ Antarmuka pengguna yang elegan dan ramah mata (mendukung mode gelap).

---

## 📦 Instalasi

### 1. Kloning Repositori
```bash
git clone [https://github.com/username/image-segmentation-app.git](https://github.com/username/image-segmentation-app.git)
cd image-segmentation-app
```

### 2. Buat dan Aktifkan Environment (Opsional tapi disarankan)
```bash
# Untuk macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Untuk Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instal Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
streamlit run app.py
```

---

## 🖼️ Tampilan

| Gambar Asli | Gambar Tersegmentasi |
| :---------: | :------------------: |
|  *[Contoh Gambar Asli]* | *[Contoh Gambar Hasil Segmentasi]* |

---

## 🔧 Struktur Folder

```
.
├── app.py                   # File utama aplikasi Streamlit
├── requirements.txt         # Daftar dependensi Python
├── assets/                  # Folder untuk contoh gambar (opsional)
└── README.md
```

---

## ⚙️ Dependensi Utama

File `requirements.txt` akan berisi:
```
streamlit
numpy
opencv-python-headless
scikit-learn
matplotlib
Pillow
scipy
```

---

## 📌 Catatan

-   **Silhouette Score** digunakan untuk secara objektif menentukan jumlah cluster (K) yang paling optimal untuk segmentasi.
-   Semua proses *clustering* dilakukan pada ruang warna **RGB** dan tidak memerlukan data berlabel.
-   Untuk efisiensi, gambar secara otomatis diubah ukurannya menjadi **128x128 piksel** sebelum diproses.
