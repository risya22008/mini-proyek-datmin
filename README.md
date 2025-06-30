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
## 📱 Link Aplikasi
https://mini-proyek-datmin-6m3rbnvbrcut8prqmtgqmg.streamlit.app/

---

## 🖼️ Tampilan

| Gambar Asli | Gambar Tersegmentasi |
| :---------: | :------------------: |
|  ![image](https://github.com/user-attachments/assets/6992ae8e-57c9-4eee-b886-0f4ef47685e9)| ![image](https://github.com/user-attachments/assets/3c75cfef-6f0c-4f3e-a332-4b950c27cec3)|

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
