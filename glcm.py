import os
import cv2
import imutils
import numpy
from skimage.feature import greycomatrix, greycoprops
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

def get_features(path):
    img = cv2.imread(path)
    img = imutils.resize(img, width=500)
    image = 255 * img
    image = numpy.where((image > 80) & (image < 160), 200, image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(gray, [6], [0, numpy.pi / 4, numpy.pi / 2, 3 * numpy.pi / 4], 256, True, True)
    con = greycoprops(glcm, 'contrast')[0, 0]
    dis = greycoprops(glcm, 'dissimilarity')[0, 0]
    hom = greycoprops(glcm, 'homogeneity')[0, 0]
    cor = greycoprops(glcm, 'correlation')[0, 0]

    return con, dis, hom, cor


def main():
    train_feature = []
    labels = []
    train_dir = 'textures-train'
    test_dir = 'textures-test'

    for class_dir in os.listdir(train_dir):
        for file in os.listdir(f'{train_dir}/{class_dir}'):
            path = f'{train_dir}/{class_dir}/{file}'
            feature = get_features(path)

            train_feature.append(feature)
            labels.append(class_dir)

    scaler = MinMaxScaler()
    scaler.fit(train_feature)
    train_feature = scaler.transform(train_feature)

    model = SVC(gamma='auto')
    model.fit(train_feature, labels)

    for file in os.listdir(test_dir):
        path = f'{test_dir}/{file}'
        feature = get_features(path)
        feature = scaler.transform([feature])
        prediction = model.predict(feature)

        img = cv2.imread(path)
        cv2.putText(img, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow(path, img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()