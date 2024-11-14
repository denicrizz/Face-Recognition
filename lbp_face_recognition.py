import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os

def get_pixel(img, center, x, y):
    """Get pixel value at coordinate (x,y) or return 0 if it's outside the image"""
    new_value = 0
    try:
        if img[x,y] >= center:
            new_value = 1
    except:
        pass
    return new_value

def lbp_calculated_pixel(img, x, y):
    """Calculate LBP value for a given pixel"""
    center = img[x,y]
    val_ar = []
    # top_left
    val_ar.append(get_pixel(img, center, x-1, y-1))
    # top
    val_ar.append(get_pixel(img, center, x-1, y))
    # top_right
    val_ar.append(get_pixel(img, center, x-1, y+1))
    # right
    val_ar.append(get_pixel(img, center, x, y+1))
    # bottom_right
    val_ar.append(get_pixel(img, center, x+1, y+1))
    # bottom
    val_ar.append(get_pixel(img, center, x+1, y))
    # bottom_left
    val_ar.append(get_pixel(img, center, x+1, y-1))
    # left
    val_ar.append(get_pixel(img, center, x, y-1))
    
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = 0
    for i in range(len(val_ar)):
        val += val_ar[i] * power_val[i]
    return val

def get_lbp_features(image):
    """Extract LBP features from an image"""
    height, width = image.shape
    img_lbp = np.zeros((height, width), np.uint8)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            img_lbp[i, j] = lbp_calculated_pixel(image, i, j)
    
    # Calculate histogram of LBP features
    hist_lbp = cv2.calcHist([img_lbp], [0], None, [256], [0, 256])
    return hist_lbp.flatten()

def load_dataset(dataset_path):
    """Load images and labels from dataset directory"""
    images = []
    labels = []
    label_dict = {}
    current_label = 0
    
    # Iterate through each person's directory
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_path):
            label_dict[person_name] = current_label
            
            # Process each image for the current person
            for image_name in os.listdir(person_path):
                image_path = os.path.join(person_path, image_name)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is not None:
                    # Resize image for consistency
                    image = cv2.resize(image, (100, 100))
                    # Extract LBP features
                    features = get_lbp_features(image)
                    images.append(features)
                    labels.append(current_label)
            
            current_label += 1
    
    return np.array(images), np.array(labels), label_dict

def train_model(X, y):
    """Train KNN classifier"""
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    
    # Evaluate model
    accuracy = knn.score(X_test, y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    
    return knn

def recognize_face(image_path, model, label_dict):
    """Recognize face in a given image"""
    # Read and preprocess image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return "Error: Could not load image"
    
    # Resize image
    image = cv2.resize(image, (100, 100))
    
    # Extract LBP features
    features = get_lbp_features(image)
    
    # Predict
    prediction = model.predict([features])[0]
    
    # Get person name from label
    for name, label in label_dict.items():
        if label == prediction:
            return name
    
    return "Unknown"

# Example usage
if __name__ == "__main__":
    dataset_path = "dataset_folder"  # Replace with your dataset path
    
    # Load and prepare dataset
    print("Loading dataset...")
    X, y, label_dict = load_dataset(dataset_path)
    
    # Train model
    print("Training model...")
    model = train_model(X, y)
    
    # Example recognition
    test_image_path = "test_image.jpg"  # Replace with test image path
    result = recognize_face(test_image_path, model, label_dict)
    print(f"Recognized person: {result}")