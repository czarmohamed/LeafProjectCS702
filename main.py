import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack import dct, idct
from skimage import io, color, restoration, img_as_float, filters
import pandas as pd
import math
from scipy.signal import convolve2d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import greycomatrix, greycoprops
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")  # "error", "ignore", "always", "default", "module" or "once"
# ------------------------------------------------------------------
# Read Image Acer Campestre
leaf = cv2.imread(r"ImageData\ny1079-01-1.jpg")
cv2.imshow("Original Image",leaf)

cv2.waitKey(0)
# ------------------------------------------------------------------
# Convert to Greyscale
img = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)
cv2.imshow('Greyscale', img)
cv2.waitKey(0)


# ------------------------------------------------------------------
# DCT Transform Functions
def dcttransform(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')


def idcttransform(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')


dctimg = (dcttransform(img))
idctimg = (idcttransform(dctimg))

np.allclose(img, idctimg)

plt.gray()
plt.subplot(121), plt.imshow(img), plt.axis('off'), plt.title('Original', size=20)
plt.subplot(122), plt.imshow(idctimg), plt.axis('off'), plt.title('Reconstructed', size=20)
plt.show()
# ------------------------------------------------------------------
# Morph
kernel = np.ones((2,2),np.uint8)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
kernel = np.ones((6,6),np.uint8)
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

cv2.imshow("Opening", opening)
cv2.waitKey(0)
cv2.imshow("BlackHat", blackhat)
cv2.waitKey(0)
# ------------------------------------------------------------------
# Blur Filters

median = cv2.medianBlur(img,ksize=3)
gaussian = cv2.GaussianBlur(img,(5,5),0)
imgFloat = img_as_float(img)
psf = np.ones((5, 5)) / 25
imgWiener = convolve2d(imgFloat, psf, 'same')
imgWiener += 0.1 * imgWiener.std() * np.random.standard_normal(imgWiener.shape)
decon = restoration.wiener(imgWiener, psf, 1100)
cv2.imshow("Median", median)
cv2.waitKey(0)
cv2.imshow("Gaussian", gaussian)
cv2.waitKey(0)
cv2.imshow("Wiener", decon)
cv2.waitKey(0)
# ------------------------------------------------------------------
# Thresholding
thresh = cv2.medianBlur(img,ksize=3)# smooth the image
thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)

cv2.imshow("Threshold Image", thresh)


cv2.waitKey(0)

# ------------------------------------------------------------------

# Sobel Edge Detection

edgesobel = filters.sobel(thresh)
cv2.imshow("Sobel", edgesobel)
cv2.waitKey(0)
# ------------------------------------------------------------------
# Input images from dataset and process
imgarray = []
files = glob.glob(r"ImageData\*.jpg")

for myFile in files:
    img = cv2.imread(myFile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.medianBlur(img, 3)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    imgarray.append(thresh)

# Display all images in the dataset
"""
for i in range(len(imgarray)):
    cv2.imshow("image", imgarray[i])
    cv2.waitKey(0)
"""

# ------------------------------------------------------------------
# Calculate the feature values for each image
featureData = []

for i in range(len(imgarray)):
    distances = [1]
    angles = [0]
    properties = ['energy', 'homogeneity', 'contrast', 'correlation', 'dissimilarity']

    glcm = greycomatrix(imgarray[i],
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)

    featureData.append(np.hstack([greycoprops(glcm, prop).ravel() for prop in properties]))
# print(featureData)
logFeatureData = []

for i in range(len(featureData)):
    features = featureData[i]
    for j in range(len(features)):
        if features[j] > 0:
            features[j] = (math.log10(features[j]))
        else:
            features[j] = math.log10(-1 * features[j])

    logFeatureData.append(features)
# ------------------------------------------------------------------
# calculate the Hu moment invariant for each image

logHuMomentData = []
for i in range(len(imgarray)):
    # Calculate Moments
    moments = cv2.moments(imgarray[i])
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    logTransform = []
    for i in range(len(huMoments)):
        if huMoments[i] > 0:
            logTransform.append(math.log10(huMoments[i]))
        else:
            logTransform.append(math.log10(-1 * huMoments[i]))
    for p in range(len(logTransform)):
        logTransform[p] = round(logTransform[p], 3)
    logHuMomentData.append(logTransform)

# Print every feature vector in the dataset
print("List of feature vectors for every image in the Dataset: ")
for i in range(len(logHuMomentData)):
    print("Feature Vector for Image:",i+1)
    print(logFeatureData[i])
    print(",")
    print(logHuMomentData[i])
    print("\n")
# ------------------------------------------------------------------
# k-NN algorithm
featureVector = ['energy', 'homogeneity', 'contrast', 'correlation', 'dissimilarity', 'Hu1',
         'Hu2', 'Hu3', 'Hu4', 'Hu5', 'Hu6', 'Hu7', 'Class']

dataset = pd.read_csv(r"leafdataset.txt", names=featureVector)

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 12].values

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1)


# normalize the dataset
scaler = StandardScaler()
scaler.fit(xtrain)

xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(xtrain, ytrain)

ypred = classifier.predict(xtest)

print(confusion_matrix(ytest, ypred))
print(classification_report(ytest, ypred))
# ------------------------------------------------------------------
# k error calculation
error = []

for i in range(1, 45):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain, ytrain)
    predict = knn.predict(xtest)
    error.append(np.mean(predict != ytest))
plt.figure(figsize=(12, 6))
plt.plot(range(1, 45), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')
plt.show()

