
# Pengenalan Wajah dengan Local Binary Pattern (LBP) dan Random Forest

Aplikasi ini menggunakan Local Binary Pattern (LBP) untuk ekstraksi fitur wajah dan Random Forest sebagai algoritma klasifikasi untuk mengenali wajah dari gambar. Aplikasi dibangun dengan Python dan Streamlit sebagai antarmuka pengguna untuk mempermudah penggunaan.

## Fitur Utama
- **Ekstraksi Fitur LBP**: Ekstrak pola tekstur lokal dari gambar wajah untuk menghasilkan fitur yang dapat digunakan untuk klasifikasi.
- **Klasifikasi dengan Random Forest**: Gunakan Random Forest untuk membangun model pengenalan wajah berdasarkan dataset.
- **Antarmuka Streamlit**: Antarmuka sederhana untuk mengunggah gambar dan melihat hasil pengenalan wajah.

---

## Struktur Proyek
```
.
├── dataset/
│   ├── deni1.jpg
│   ├── deni2.jpg
│   ├── deni3.jpg
│   ├── prabowo1.jpg
│   ├── prabowo2.jpg
│   ├── prabowo3.jpg
│   ├── yoona1.jpg
│   ├── yoona2.jpg
│   ├── yoona3.jpg
├── app.py
├── requirements.txt
├── README.md
```

- **`dataset/`**: Folder berisi gambar dataset wajah yang digunakan untuk melatih dan menguji model.
- **`app.py`**: Kode utama aplikasi yang menjalankan proses pelatihan dan pengenalan wajah.
- **`requirements.txt`**: File dependensi untuk aplikasi.
- **`README.md`**: Dokumentasi proyek.

---

## Instalasi

1. Clone repositori ini:
   ```bash
   git clone https://github.com/username/face-recognition-lbp-randomforest.git
   cd face-recognition-lbp-randomforest
   ```

2. Buat dan aktifkan virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate     # Untuk Windows
   ```

3. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```

---

## Cara Menjalankan Aplikasi

1. Pastikan dataset gambar telah ditempatkan di folder `dataset/`. Contoh struktur dataset:
   ```
   dataset/
   ├── deni1.jpg
   ├── deni2.jpg
   ├── prabowo1.jpg
   ├── yoona1.jpg
   ```

2. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```

3. Unggah gambar melalui antarmuka Streamlit dan lihat hasil pengenalan wajah.

---

## Penjelasan Kode

### Ekstraksi Fitur dengan LBP
Fitur LBP dihitung dengan membandingkan piksel pusat dengan tetangganya untuk mendapatkan pola biner, kemudian diubah menjadi histogram.

### Klasifikasi dengan Random Forest
Model `RandomForestClassifier` dilatih menggunakan fitur LBP dari dataset wajah. Model ini digunakan untuk memprediksi identitas wajah dari gambar uji.

### Evaluasi Akurasi
Dataset dibagi menjadi data latih (80%) dan data uji (20%) menggunakan `StratifiedShuffleSplit`. Akurasi dihitung pada data uji.

---

## Contoh Output

1. **Antarmuka Streamlit**
   - Unggah gambar wajah melalui antarmuka pengguna.
   - Aplikasi akan menampilkan hasil prediksi wajah yang dikenali.

2. **Akurasi Model**
   - Akurasi model ditampilkan setelah pelatihan, misalnya:
     ```
     Model berhasil dilatih! Akurasi: 90.00%
     ```

---

## Kebutuhan Sistem
- Python >= 3.8
- Streamlit
- OpenCV
- scikit-learn
- NumPy

---

## Masalah yang Sering Dihadapi

1. **Akurasi Rendah**:
   - Tambahkan lebih banyak data untuk setiap kelas.
   - Gunakan augmentasi data untuk memperbanyak variasi.

2. **Gambar Tidak Dimuat**:
   - Pastikan gambar dalam format `.jpg` atau `.png` dan folder `dataset/` memiliki gambar yang benar.

3. **Kesalahan Path Dataset**:
   - Periksa bahwa path gambar di dalam kode cocok dengan struktur folder Anda.

---

## Kontributor
- **Nama Anda**  
  Pengembang Aplikasi Pengenalan Wajah.

---

## Lisensi
Proyek ini menggunakan lisensi MIT. Silakan lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.

---

## Catatan
Jika Anda memiliki pertanyaan atau masalah, jangan ragu untuk membuka isu atau hubungi saya melalui email. Selamat mencoba! 🎉
