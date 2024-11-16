import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Fungsi untuk menghitung LBP
def lbp_calculated_pixel(img, x, y):
    center = img[x, y]
    neighbors = [
        (x - 1, y - 1), (x - 1, y), (x - 1, y + 1),
        (x, y + 1), (x + 1, y + 1), (x + 1, y),
        (x + 1, y - 1), (x, y - 1)
    ]
    binary_vals = [1 if img[nx, ny] >= center else 0 for nx, ny in neighbors if 0 <= nx < img.shape[0] and 0 <= ny < img.shape[1]]
    power_val = [2**i for i in range(len(binary_vals))]
    return sum(b * p for b, p in zip(binary_vals, power_val))

# Ekstrak fitur LBP
def get_lbp_features(image):
    height, width = image.shape
    lbp_img = np.zeros((height, width), np.uint8)
    for x in range(1, height - 1):
        for y in range(1, width - 1):
            lbp_img[x, y] = lbp_calculated_pixel(image, x, y)
    hist_lbp = cv2.calcHist([lbp_img], [0], None, [256], [0, 256])
    return hist_lbp.flatten()

# Muat dataset
def load_dataset(image_paths, labels):
    features, valid_labels = [], []
    for path, label in zip(image_paths, labels):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            image = cv2.resize(image, (100, 100))
            features.append(get_lbp_features(image))
            valid_labels.append(label)
    return np.array(features), np.array(valid_labels)

# Latih model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluasi akurasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Pengenalan wajah dengan bounding box, nama, dan keterangan teks
def recognize_face_with_output(image, model, label_names):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Deteksi wajah
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return image, "Tidak ada wajah yang terdeteksi.", []

    recognized_names = []  # Simpan nama yang dikenali
    for (x, y, w, h) in faces:
        # Ekstrak area wajah
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (100, 100))
        features = get_lbp_features(face)
        
        # Prediksi nama
        prediction = model.predict([features])[0]
        name = label_names[prediction]
        recognized_names.append(name)
        
        # Gambar kotak di sekitar wajah
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Tambahkan nama pada kotak
        cv2.putText(image, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    return image, "Wajah dikenali!", recognized_names

# Streamlit Interface
def main():
    st.title("Pengenalan Wajah")
    st.write("Aplikasi ini berjalan menggunakan metode LBP + Random Forest untuk deteksi dan pengenalan wajah.")
    
    # Path dataset sederhana
    image_paths = [
        "dataset/image1.png",
        "dataset/image2.png",
        "dataset/image3.png",
        "dataset/image4.png"
    ]
    labels = [0, 1, 2, 3]
    label_names = ["Deni", "Prabowo", "Yoona", "Tzuyu"]
    
    try:
        # Load dataset
        st.write("Memuat dataset...")
        X, y = load_dataset(image_paths, labels)
        
        # Bagi dataset menjadi latih dan uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Latih model
        st.write("Melatih model...")
        model = train_model(X_train, y_train)
        
        # Evaluasi model
        st.write("Evaluasi model...")
        accuracy = evaluate_model(model, X_test, y_test)
        st.write(f"Model berhasil dilatih! Akurasi: {accuracy * 100:.2f}%")
        
        # Pengenalan wajah
        uploaded_file = st.file_uploader("Unggah gambar untuk pengenalan", type=["jpg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            if image is not None:
                result_image, message, recognized_names = recognize_face_with_output(image, model, label_names)
                st.write(message)
                
                # Tampilkan nama-nama yang dikenali dalam teks
                if recognized_names:
                    st.write(f"Wajah yang dikenali: {', '.join(recognized_names)}")
                
                # Konversi gambar ke format RGB untuk ditampilkan
                result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_GRAY2RGB)
                st.image(result_image_rgb, caption="Hasil Pengenalan Wajah", use_container_width=True)
            else:
                st.write("Gagal memuat gambar.")
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

if __name__ == "__main__":
    main()

st.markdown("---") 
st.markdown("<footer style='text-align: center;'>Dibuat dengan &hearts; Oleh Kelompok 6 - Klepon</footer>", unsafe_allow_html=True)