import math

import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops

leaf = cv2.imread(r"C:\Users\user-pc\Downloads\ImageData\New folder\wb1188-10-2.jpg")
cv2.imshow("Original Image",leaf)

img = cv2.cvtColor(leaf, cv2.COLOR_BGR2GRAY)
thresh = cv2.medianBlur(img, 3)
thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
cv2.waitKey(0)
cv2.imshow("Original Image",thresh)
cv2.waitKey(0)
featureData = []
distances = [1]
angles = [0]
properties = ['energy', 'homogeneity', 'contrast', 'correlation', 'dissimilarity']

glcm = greycomatrix(thresh,
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

logHuMomentData = []

# Calculate Moments
moments = cv2.moments(thresh)
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
print("List of feature vectors for the Dataset: ")
File_object = open(r"newdata.txt",mode="w")
for i in range(len(logHuMomentData)):
    File_object.write(str(logFeatureData[i]))
    File_object.write(",")
    File_object.write(str(logHuMomentData[i]))
    print(logFeatureData[i])
    print(",")
    print(logHuMomentData[i])
    print("\n")
