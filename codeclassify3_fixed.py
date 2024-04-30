import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import warnings

from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

class ImageClassifier:
    def __init__(self, dataset_dir, model_dir, feature_dir):
        self.dataset_dir = dataset_dir
        self.model_dir = model_dir
        self.feature_dir = feature_dir
        self.data = []
        self.labels = []
        # Membuat folder model dan fitur jika belum ada
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.feature_dir, exist_ok=True)

    def extract_histogram(self, image):
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 255, 0, 255, 0, 255])
        cv2.normalize(hist, hist)
        hist = hist.flatten()
        return hist
    
    def equalize_color_image(self, image):
        if image is None:
            print("Error: Cannot read image.")
            return
        image_equalized = image.copy()
        for i in range(3):
            image_equalized[:,:,i] = cv2.equalizeHist(image[:,:,i])
        return image_equalized

    def process_image(self, image, filename):
        image = self.equalize_color_image(image)
        return image
    
    def load_data(self):
        for folder in os.listdir(self.dataset_dir):
            folder_path = os.path.join(self.dataset_dir, folder)
            
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    
                    if file.endswith('.jpg'):
                        image = cv2.imread(file_path)                     
                        image = self.process_image(image, file)

                        features = self.extract_histogram(image)

                        self.data.append(features)
                        self.labels.append(folder)

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)

    def train_decision_tree(self):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(self.data, self.labels)
        return decision_tree
    
    def train_knn(self):
        knn = KNeighborsClassifier(n_neighbors=253)
        knn.fit(self.data, self.labels)
        return knn
        
    def train_classifier(self):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.labels, test_size=0.5, random_state=42)

        classifier = self.train_knn()
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        print(classification_report(y_test, y_pred))

        self.classifier = classifier

    def save_classifier(self):
        np.save(os.path.join(self.feature_dir, 'data.npy'), self.data)
        np.save(os.path.join(self.feature_dir, 'labels.npy'), self.labels)

        classifier = self.classifier

        with open(os.path.join(self.model_dir, 'knn_model.pkl'), 'wb') as f:
            pickle.dump(classifier, f)
    
    def load_classifier(self):
        with open(os.path.join(self.model_dir, 'knn_model.pkl'), 'rb') as f:
            self.classifier = pickle.load(f)

    def test_classifier(self, test_data):
        test_features = np.array([self.extract_histogram(image) for image in test_data])
        test_predictions = self.classifier.predict(test_features)
        return test_predictions


if __name__ == "__main__":
    folder_current = r'C:\Code\Python\Machine Learning'
    DATASET_DIR = os.path.join(folder_current,  'Sample', 'Mata')
    MODEL_DIR = os.path.join(folder_current, 'Model')
    FEATURE_DIR = os.path.join(folder_current, 'Fitur')

    # Create an instance of ImageClassifier and train the KNN classifier
    classifier = ImageClassifier(DATASET_DIR, MODEL_DIR, FEATURE_DIR)
    classifier.load_data()
    classifier.train_classifier()
    classifier.save_classifier()

    # Load trained classifier
    classifier.load_classifier()

    # Example of testing the classifier with new data
    test_image = 'Iris.jpg'
    test_image_path = os.path.join(folder_current,'Sample', test_image)  # Masukkan path gambar uji di sini
    test_data = [cv2.imread(test_image_path)]
    predictions = classifier.test_classifier(test_data)
    print("Predictions:", predictions)
