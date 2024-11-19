import os
import cv2
import numpy as np
import streamlit as st
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def extract_lbp_features(image, P=16, R=2):
    """Extract LBP features from image"""
    lbp = local_binary_pattern(image, P, R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype('float32')
    hist /= (hist.sum() + 1e-7)
    return hist

def load_dataset(dataset_path):
    """Load and process dataset"""
    features = []
    labels = []
    label_names = []
    
    all_folders = [f for f in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, f))]
    
    for label_idx, folder_name in enumerate(all_folders):
        folder_path = os.path.join(dataset_path, folder_name)
        label_names.append(folder_name)
        
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, image_file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    image = cv2.resize(image, (200, 200))
                    image = cv2.equalizeHist(image)
                    lbp_features = extract_lbp_features(image)
                    features.append(lbp_features)
                    labels.append(label_idx)
    
    return np.array(features), np.array(labels), label_names

def predict_face(image, model, label_names):
    """Predict face from image"""
    if image is None:
        return None, "Failed to load image"
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(70, 70))

    if len(faces) == 0:
        return image, []

    result_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    predictions = []

    for (x, y, w, h) in faces:
        face_roi = gray[y:y + h, x:x + w]
        face_roi = cv2.resize(face_roi, (250, 250))
        face_roi = cv2.equalizeHist(face_roi)
        features = extract_lbp_features(face_roi)

        probs = model.predict_proba([features])[0]
        pred_idx = np.argmax(probs)
        accuracy = probs[pred_idx] * 100
        name = label_names[pred_idx]

        predictions.append((name, accuracy))
        label = f"{name} ({accuracy:.1f}%)"

        color = (0, 255, 0) if accuracy > 80 else (0, 255, 255) if accuracy > 60 else (0, 0, 255)
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(result_image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return result_image, predictions

def main():
    st.title("Face Recognition App")
    
    # Sidebar for dataset path
    dataset_path = st.sidebar.text_input("Dataset Path", "dataset")
    
    if not os.path.exists(dataset_path):
        st.error(f"Dataset path '{dataset_path}' does not exist!")
        return
        
    # Load and train model
    with st.spinner("Loading dataset and training model..."):
        features, labels, label_names = load_dataset(dataset_path)
        
        if len(features) == 0:
            st.error("Error: No data loaded!")
            return
            
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels)
        
        model = RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        train_accuracy = model.score(X_train, y_train)
        test_accuracy = model.score(X_test, y_test)
        
        st.sidebar.metric("Training Accuracy", f"{train_accuracy*100:.2f}%")
        st.sidebar.metric("Testing Accuracy", f"{test_accuracy*100:.2f}%")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        result_image, predictions = predict_face(image, model, label_names)
        
        if predictions:
            st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), caption='Detected Face(s)')
            
            st.subheader("Predictions:")
            for name, accuracy in predictions:
                st.write(f"- {name}: {accuracy:.1f}%")
        else:
            st.warning("No faces detected in the image")

if __name__ == "__main__":
    main()