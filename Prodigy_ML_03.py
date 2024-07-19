import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray


def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir("C:/Users/Pramod/PycharmProjects/Prodigy_t/ML/data"):
        img_path = os.path.join('data', 'cat_b.jpg')
        img = imread("C:/Users/Pramod/PycharmProjects/Prodigy_t/ML/data/cat_b.jpg")
        img = rgb2gray(img)
        img = resize(img, (50, 50), anti_aliasing=True)
        images.append(img.flatten())
        labels.append(label)
    return images, labels

cat_folder = "C:/Users/Pramod/PycharmProjects/Prodigy_t/ML/cats"
dog_folder = "C:/Users/Pramod/PycharmProjects/Prodigy_t/ML/dogs"

print("Loading images...")
cat_images, cat_labels = load_images(cat_folder, 0)
dog_images, dog_labels = load_images(dog_folder, 1)

X = np.array(cat_images + dog_images)
y = np.array(cat_labels + dog_labels)

print(f"Total images loaded: {len(X)}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training SVM model...")
svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_train, y_train)

print("Evaluating model...")
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


def predict_image(image_path, model):
    img = imread(image_path)
    img = rgb2gray(img)
    img = resize(img, (50, 50), anti_aliasing=True).flatten()
    prediction = model.predict([img])
    return "Cat" if prediction[0] == 0 else "Dog"

test_image_path = 'C:/Users/Pramod/PycharmProjects/Prodigy_t/ML/data/cat_b.jpg'
result = predict_image(test_image_path, svm)
print(f"The test image is classified as: {result}")