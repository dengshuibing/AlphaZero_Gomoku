import cv2


image1 = cv2.imread('image1.jpg')
image2 = cv2.imread('image2.jpg')


image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
merged_image = cv2.hconcat([image1, image2])


cv2.imshow('Merged Image', merged_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
