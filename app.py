import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
import streamlit as st

# Fungsi untuk mendapatkan nilai piksel pada koordinat (x, y) atau mengembalikan 0 jika di luar gambar
def get_pixel(img, center, x, y):
    """Get pixel value at coordinate (x,y) or return 0 if it's outside the image"""
    new_value = 0
    try:
        if img[x, y] >= center:
            new_value = 1
    except:
        pass
    return new_value

# Fungsi untuk menghitung LBP pada piksel tertentu
def lbp_calculated_pixel(img, x, y):
    """Calculate LBP value for a given pixel"""
    center = img[x, y]
    val_ar = []
    # top_left
    val_ar.append(get_pixel(img, center, x - 1, y - 1))
    # top
    val_ar.append(get_pixel(img, center, x - 1, y))
    # top_right
    val_ar.append(get_pixel(img, center, x - 1, y + 1))
    # right
    val_ar.append(get_pixel(img, center, x, y + 1))
    # bottom_right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))
    # bottom
    val_ar.append(get_pixel(img, center, x + 1, y))
    # bottom_left
    val_ar.append(get_pixel(img, center, x + 1, y - 1))
    # left
    val_ar.append(get_pixel(img, center, x, y - 1))
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

# Fungsi untuk mengekstrak fitur LBP dari gambar
def get_lbp_features(image):
    """Extract LBP features from an image"""
    height, width = image.shape
    img_lbp = np.zeros((height, width), np.uint8)
    
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            img_lbp[i, j] = lbp_calculated_pixel(image, i, j)
    
    # Menghitung histogram fitur LBP
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    return hist_lbp.flatten()

# Fungsi untuk memuat dataset
def load_dataset(dataset_path):
    """Load images and labels from dataset directory"""
    images = []
    labels = []
    label_dict = {}
    current_label = 0
    
    # Iterasi melalui setiap direktori orang
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            label_dict[person_name] = current_label
            
            # Proses setiap gambar untuk orang tersebut
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    # Ukuran gambar untuk konsistensi
                    image = cv2.resize(image, (100, 100))
                    # Ekstrak fitur LBP
                    features = get_lbp_features(image)
                    images.append(features)
                    labels.append(current_label)
            
            current_label += 1
    
    return np.array(images), np.array(labels), label_dict

# Fungsi untuk melatih model KNN
def train_model(X, y):
    """Train KNN classifier"""
    # Membagi dataset menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Inisialisasi dan latih KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Evaluasi model
    accuracy = knn.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    return knn

# Fungsi untuk mendeteksi wajah pada gambar
def detect_face(image):
    """Detect face in the image using Haar Cascade Classifier"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Deteksi wajah pada gambar
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, image  # Tidak ada wajah yang terdeteksi
    
    # Memotong gambar untuk hanya menyertakan wajah pertama yang terdeteksi
    (x, y, w, h) = faces[0]
    face = image[y:y + h, x:x + w]
    return face, faces

# Fungsi untuk mengenali wajah dalam gambar
def recognize_face(image, model, label_dict):
    """Recognize face in a given image"""
    # Deteksi wajah pada gambar
    face, faces = detect_face(image)
    if face is None:
        return "Tidak ada wajah yang terdeteksi."
    
    # Ukuran gambar wajah untuk konsistensi
    face = cv2.resize(face, (100, 100))
    
    # Ekstrak fitur LBP
    features = get_lbp_features(face)
    
    # Prediksi
    prediction = model.predict([features])[0]
    
    # Mendapatkan nama orang dari label
    for name, label in label_dict.items():
        if label == prediction:
            return name
    
    return "Tidak dikenal"

# Streamlit Interface
def main():
    st.title("Pengenalan Wajah dengan LBP dan KNN")
    st.write("Unggah folder dataset yang berisi subfolder untuk setiap orang dengan gambar-gambar wajah.")
    
    dataset_path = st.text_input("Masukkan path ke folder dataset:")
    if dataset_path:
        # Memuat dan menyiapkan dataset
        st.write("Memuat dataset...")
        X, y, label_dict = load_dataset(dataset_path)
        
        # Melatih model
        st.write("Melatih model...")
        model = train_model(X, y)
        
        st.write("Model berhasil dilatih!")
        
        # Unggah gambar untuk pengenalan wajah
        uploaded_file = st.file_uploader("Unggah gambar untuk pengenalan", type=["jpg", "png"])
        if uploaded_file is not None:
            # Membaca gambar untuk pengenalan wajah
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                st.write("Error: Gambar tidak dapat dimuat.")
            else:
                # Mengenali wajah
                result = recognize_face(image, model, label_dict)
                st.write(f"Wajah yang dikenali: {result}")

if __name__ == "__main__":
    main()
