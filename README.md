# 🌴 Deteksi dan Klasifikasi Kesehatan Pohon Sawit Menggunakan Deep Learning pada Citra Udara

## 📌 Deskripsi Proyek
Proyek ini bertujuan untuk **mendeteksi** pohon sawit dari citra udara (drone) sekaligus **mengklasifikasi kondisi kesehatannya** (sehat atau kurang sehat).  
Model deep learning yang digunakan adalah **YOLOv8s** untuk deteksi objek, kemudian dilakukan klasifikasi kesehatan menggunakan fitur tambahan dari **NDVI masking**.

---

## 🧠 Metodologi

### 1. **Deteksi Pohon Sawit**
- Model **YOLOv8s** digunakan untuk mendeteksi bounding box dari pohon sawit pada citra lapangan.  
- Hasil deteksi berupa koordinat bounding box setiap pohon dalam satu foto.

### 2. **Masking NDVI**
- Setiap bounding box diekstraksi kemudian diubah menjadi **masking NDVI (Normalized Difference Vegetation Index)**.  
- NDVI dipakai sebagai **augmentasi tambahan** agar tingkat kehijauan daun lebih jelas, sehingga mempermudah klasifikasi kesehatan pohon.

### 3. **Klasifikasi Kesehatan**
- Setelah bounding box diperoleh, setiap pohon diklasifikasikan menjadi:
  - 🌱 **Sehat**  
  - 🍂 **Kurang Sehat**  
- Klasifikasi dilakukan berdasarkan fitur visual dari **citra RGB + NDVI masking**.

---

## 🚀 Deployment
Model ini dideploy menggunakan **Hugging Face Spaces** dengan framework **Gradio**, sehingga pengguna dapat:
1. Mengunggah citra udara kebun sawit.
2. Melihat hasil deteksi pohon beserta bounding box.
3. Mendapatkan label kesehatan masing-masing pohon (sehat / kurang sehat).

👉 [Hugging Face Demo](#) *(https://huggingface.co/spaces/jeremymboe/PalmTree_Detection_HealthClassification_BIGTOR)*
Untuk mencoba hasil model dapat menggunakan dataset di folder **test/image** pada **dataset_prepocessedYolo**

##  📝 Paper
Project ini dilombakan pada Gemastik Divisi Penambangan Data 2025 yang bisa diakses di folder **paper**.

---
