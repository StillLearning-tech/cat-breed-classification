# 🐱 Cat Breed Classification

## Klasifikasi 5 Ras Kucing Menggunakan CNN dengan MobileNetV2

### 📌 Deskripsi Proyek

Cat Breed Classification merupakan proyek machine learning berbasis Deep Learning yang bertujuan untuk mengklasifikasikan ras kucing berdasarkan gambar menggunakan metode Convolutional Neural Network (CNN).

Pada proyek ini, digunakan arsitektur **MobileNetV2** dengan pendekatan Transfer Learning untuk membantu model mengenali karakteristik visual dari lima ras kucing. Model yang telah dilatih kemudian diintegrasikan ke dalam aplikasi berbasis Streamlit sehingga dapat digunakan untuk melakukan prediksi gambar kucing.

### 🎯 Tujuan Proyek

- Mengembangkan model CNN untuk klasifikasi ras kucing.
- Menerapkan Transfer Learning menggunakan MobileNetV2.
- Melakukan preprocessing dan augmentasi data gambar.
- Mengevaluasi performa model menggunakan data testing.
- Mengimplementasikan model ke dalam aplikasi web menggunakan Streamlit.

### 🐈 Kelas Klasifikasi

Model dapat mengenali lima jenis ras kucing:

1. Bengal
2. Maine Coon
3. Russian Blue
4. Siamese
5. Snowshoe

### 📊 Dataset

Dataset yang digunakan terdiri dari **6.911 gambar** yang terbagi ke dalam lima kelas ras kucing.

Pembagian dataset dilakukan secara stratified dengan komposisi:

| Dataset | Jumlah Gambar | Persentase |
|---|---:|---:|
| Training | 4.837 | 70% |
| Validation | 1.037 | 15% |
| Testing | 1.037 | 15% |
| **Total** | **6.911** | **100%** |

Sumber DataSet : https://www.kaggle.com/datasets/imbikramsaha/cat-breeds?resource=download

### ⚙️ Metode dan Arsitektur Model

Model klasifikasi dikembangkan menggunakan Convolutional Neural Network (CNN) dengan arsitektur MobileNetV2.

Tahapan pengembangan model:

1. **Data Preprocessing**
   - Mengubah gambar ke format RGB.
   - Mengubah ukuran gambar menjadi 150 × 150 piksel.
   - Melakukan normalisasi nilai piksel menjadi 0–1.
   - Membagi dataset menjadi training, validation, dan testing.

2. **Data Augmentation**
   - Random Horizontal Flip.
   - Random Rotation.
   - Random Zoom.

3. **Transfer Learning**
   - Menggunakan MobileNetV2 pretrained ImageNet.
   - Melakukan feature extraction dengan membekukan layer pretrained.
   - Melakukan fine-tuning pada 30 layer terakhir MobileNetV2, dengan layer Batch Normalization tetap dibekukan.

4. **Klasifikasi**
   - Global Average Pooling.
   - Dense Layer dengan 128 neuron dan aktivasi ReLU.
   - Dropout untuk membantu mengurangi overfitting.
   - Output layer dengan 5 neuron dan aktivasi Softmax.

5. **Evaluasi Model**
   - Accuracy dan Loss.
   - Confusion Matrix.
   - Precision, Recall, dan F1-Score.

### 📈 Hasil Evaluasi Model

Berdasarkan pengujian menggunakan testing dataset, diperoleh hasil:

| Parameter | Hasil |
|---|---:|
| Model | MobileNetV2 |
| Testing Accuracy | **78,78%** |
| Testing Loss | 0,5083 |
| Jumlah Data Testing | 1.037 |

Hasil evaluasi menunjukkan bahwa model mampu melakukan klasifikasi terhadap lima ras kucing dengan testing accuracy sebesar 78,78%.

### 🛠️ Teknologi yang Digunakan

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Google Colab
- GitHub

### 📁 Struktur Repository

```text
cat-breed-classification/
│
├── app.py
│
├── cat_breed_mobilenetv2_final.keras
│
├── requirements.txt
│
└── README.md
```

Keterangan:

- `app.py` : Source code aplikasi klasifikasi ras kucing menggunakan Streamlit.
- `cat_breed_mobilenetv2_final.keras` : File model MobileNetV2 yang telah dilatih.
- `requirements.txt` : Daftar library Python yang dibutuhkan.
- `README.md` : Dokumentasi proyek.

### 🚀 Cara Menjalankan Aplikasi Secara Lokal

1. Clone repository:

```bash
git clone https://github.com/StillLearning-tech/cat-breed-classification.git
```

2. Masuk ke direktori proyek:

```bash
cd cat-breed-classification
```

3. Install seluruh dependency:

```bash
pip install -r requirements.txt
```

4. Jalankan aplikasi Streamlit:

```bash
streamlit run app.py
```

5. Buka alamat lokal Streamlit yang muncul pada terminal untuk mengakses aplikasi.

### 🌐 Deployment Aplikasi

Aplikasi telah diimplementasikan menggunakan Streamlit.

**Link aplikasi:** https://cat-breed-classification-mwoty5yrv4q4iszskvuykk.streamlit.app/

### 👨‍💻 Pengembang

Proyek ini dikembangkan sebagai bagian dari tugas perkuliahan.

**GitHub Repository:**  
https://github.com/StillLearning-tech/cat-breed-classification
