
# Deteksi dan Pengenalan Wajah Menggunakan LBP dan Random Forest

Proyek ini dibuat untuk memenuhi **Tugas UTS Klasifikasi Pola**. Implementasi ini menggunakan metode **Local Binary Patterns (LBP)** dan **Random Forest** untuk melakukan **Deteksi Wajah** dan **Pengenalan Wajah**.

## Deskripsi Singkat

Tujuan utama dari proyek ini adalah menciptakan aplikasi yang dapat mendeteksi dan mengenali wajah. Metode yang digunakan adalah:

1. **Local Binary Patterns (LBP):**
   - Deskriptor berbasis tekstur yang digunakan untuk menganalisis gambar pixel demi pixel.
   - Mengubah gambar ke dalam format yang mempermudah model dalam mempelajari dan mengenali fitur wajah.

2. **Random Forest:**
   - Algoritma pembelajaran mesin yang kuat untuk melatih model dengan data wajah.
   - Memungkinkan aplikasi untuk mengklasifikasi dan mengenali wajah secara akurat.

## Fitur

- **Deteksi Wajah:** Mendeteksi wajah dalam gambar atau video.
- **Pengenalan Wajah:** Mengenali wajah yang terdeteksi dengan membandingkan dengan model yang telah dilatih.
- **Dataset Kustom:** Pengguna dapat melatih model dengan dataset kustom untuk pengenalan yang lebih personal.
- **Pemrosesan Real-Time:** Mendukung deteksi dan pengenalan wajah dari kamera secara langsung.

## Metodologi

1. **Local Binary Patterns (LBP):**
   - Menangkap struktur lokal gambar.
   - Mengubah setiap pixel menjadi nilai biner dengan membandingkan pixel dengan tetangganya.

2. **Random Forest:**
   - Metode pembelajaran terawasi untuk mengklasifikasi wajah.
   - Menggabungkan beberapa decision tree untuk meningkatkan akurasi dan generalisasi.

## Alur Kerja

1. **Preprocessing:**
   - Gambar input dikonversi menjadi grayscale.
   - Fitur LBP diekstraksi dari gambar yang telah diproses.

2. **Pelatihan:**
   - Fitur LBP yang diekstraksi digunakan untuk melatih model Random Forest.

3. **Pengenalan:**
   - Gambar input dianalisis, dan wajah yang terdeteksi dibandingkan dengan model yang telah dilatih.

## Cara Instalasi

### Prasyarat
Pastikan Anda memiliki:
- Python 3.x
- Library: `opencv-python`, `numpy`, `scikit-learn`, `matplotlib`

### Langkah Instalasi
1. Clone repositori:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```

3. Siapkan dataset:
   - Letakkan gambar pelatihan di folder `data/train`.
   - Letakkan gambar pengujian di folder `data/test`.

4. Jalankan aplikasi:
   ```bash
   python main.py
   ```

## Struktur Direktori

```
📁 project-directory/
├── 📁 data/
│   ├── 📁 train/        # Dataset pelatihan
│   ├── 📁 test/         # Dataset pengujian
├── 📁 models/           # Model yang sudah dilatih
├── 📄 main.py           # Script utama
├── 📄 lbp.py            # Ekstraksi fitur LBP
├── 📄 train_model.py    # Script pelatihan model
├── 📄 detect_faces.py   # Deteksi wajah
├── 📄 requirements.txt  # Daftar dependensi
```

## Panduan Penggunaan

1. **Melatih Model:**
   - Letakkan gambar wajah di folder `data/train` dengan label yang sesuai.
   - Jalankan script pelatihan:
     ```bash
     python train_model.py
     ```

2. **Menguji Aplikasi:**
   - Letakkan gambar pengujian di folder `data/test`.
   - Jalankan script deteksi dan pengenalan wajah:
     ```bash
     python detect_faces.py
     ```

3. **Deteksi Real-Time:**
   - Eksekusi file `main.py` untuk mendeteksi wajah dari kamera secara langsung.

## Potensi Aplikasi

- Sistem absensi.
- Keamanan dan pengawasan.
- Pengalaman pelanggan personal di ritel.
- Sistem kontrol akses.

## Pengembangan ke Depan

- Menambahkan dukungan model pembelajaran mendalam untuk meningkatkan akurasi.
- Ekstensi untuk pengenalan berbasis video.
- Menyediakan GUI agar aplikasi lebih mudah digunakan.

## Kontributor

- **Nama Anda**  
  - *Tugas UTS Klasifikasi Pola*

## Lisensi

Proyek ini dilisensikan di bawah lisensi MIT. Lihat file `LICENSE` untuk detail lebih lanjut.
