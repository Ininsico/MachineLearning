import cv2
print("Testing backgrounds:")
for i in [0,1,2]:
    cap = cv2.VideoCapture(i)
    print(f'Index {i} : {cap.isOpened()}')
    cap.release()
    