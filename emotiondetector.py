from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread("img1.png")
plt.imshow(img1[:, :, ::-1]) 
plt.show()

result = DeepFace.analyze(img1, actions = ['emotion'])

print("emotion:" , result[0]['dominant_emotion'])