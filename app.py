
# ==================================================
# CAT BREED CLASSIFICATION
# IMPLEMENTASI CNN MENGGUNAKAN MOBILENETV2
# ==================================================

import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image, UnidentifiedImageError
from pathlib import Path
import io


# ==================================================
# 1. KONFIGURASI APLIKASI
# ==================================================

st.set_page_config(
    page_title="Cat Breed Classification",
    page_icon="🐱",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR / "cat_breed_mobilenetv2_final.keras"
)

CLASS_NAMES = [
    "Bengal",
    "Maine Coon",
    "Russian Blue",
    "Siamese",
    "Snowshoe"
]

IMG_SIZE = (150, 150)


# ==================================================
# 2. MEMUAT MODEL
# ==================================================

@st.cache_resource(show_spinner="Memuat model MobileNetV2...")
def load_model():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"File model tidak ditemukan: {MODEL_PATH}"
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


# ==================================================
# 3. PREPROCESSING GAMBAR
# ==================================================

def preprocess_image(image):

    # Mengubah gambar menjadi RGB
    image = image.convert("RGB")

    # Resize menjadi 150 x 150
    # Menggunakan nearest agar sesuai dengan
    # preprocessing load_img pada eksperimen Colab
    image = image.resize(
        IMG_SIZE,
        resample=Image.Resampling.NEAREST
    )

    # Mengubah gambar menjadi array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Normalisasi pixel 0-255 menjadi 0-1
    image_array = image_array / 255.0

    # Menambahkan batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ==================================================
# 4. TAMPILAN UTAMA APLIKASI
# ==================================================

st.title("🐱 Cat Breed Classification")

st.markdown(
    """
    ### Klasifikasi Ras Kucing Menggunakan CNN

    Aplikasi ini menggunakan model **MobileNetV2
    Transfer Learning** untuk mengklasifikasikan
    citra kucing ke dalam lima kategori ras.
    """
)

st.divider()

st.subheader("Upload Gambar Kucing")

uploaded_file = st.file_uploader(
    "Pilih gambar kucing yang ingin diklasifikasikan",
    type=["jpg", "jpeg", "png", "webp"]
)


# ==================================================
# 5. PROSES PREDIKSI
# ==================================================

if uploaded_file is not None:

    try:

        # Membaca gambar
        original_image = Image.open(
            io.BytesIO(uploaded_file.getvalue())
        ).convert("RGB")

        st.image(
            original_image,
            caption="Gambar yang diunggah"
        )

        # Preprocessing
        input_image = preprocess_image(
            original_image.copy()
        )

        # Memuat model
        model = load_model()

        # Prediksi
        with st.spinner("Sedang mengenali ras kucing..."):

            predictions = model.predict(
                input_image,
                verbose=0
            )[0]

        # Mengambil prediksi tertinggi
        predicted_index = int(
            np.argmax(predictions)
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        predicted_score = float(
            predictions[predicted_index]
        ) * 100


        # ==================================================
        # 6. MENAMPILKAN HASIL
        # ==================================================

        st.divider()

        st.subheader("Hasil Klasifikasi")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediksi Ras Kucing",
                predicted_class
            )

        with col2:

            st.metric(
                "Skor Softmax",
                f"{predicted_score:.2f}%"
            )

        st.success(
            f"Model memprediksi gambar sebagai: {predicted_class}"
        )


        # ==================================================
        # 7. PROBABILITAS SETIAP KELAS
        # ==================================================

        st.subheader("Skor Setiap Kelas")

        for class_name, score in zip(
            CLASS_NAMES,
            predictions
        ):

            score_percent = float(score) * 100

            st.write(
                f"**{class_name} — {score_percent:.2f}%**"
            )

            st.progress(
                min(max(float(score), 0.0), 1.0)
            )


        # ==================================================
        # 8. INFORMASI MODEL
        # ==================================================

        st.divider()

        st.caption(
            "Model: MobileNetV2 Transfer Learning + Fine-Tuning"
        )

        st.caption(
            "Testing accuracy pada dataset eksperimen: 78.78%"
        )

        st.info(
            "Aplikasi ini dirancang untuk lima ras kucing: "
            "Bengal, Maine Coon, Russian Blue, Siamese, "
            "dan Snowshoe. Model selalu memilih satu dari "
            "lima kelas tersebut. Skor Softmax bukan "
            "jaminan kebenaran prediksi."
        )


    except UnidentifiedImageError:

        st.error(
            "File tidak dapat dibaca sebagai gambar. "
            "Silakan unggah file JPG, JPEG, PNG, atau WEBP yang valid."
        )

    except Exception as error:

        st.error("Terjadi kesalahan saat memproses gambar.")

        st.exception(error)


else:

    st.info(
        "Silakan unggah gambar kucing untuk memulai klasifikasi."
    )

