from ultralytics import YOLO

if __name__ =='__main__':
    model=YOLO(r'G:\py\pythonProjectdemo\runs\detect\train3\weights\best.pt')
    model.predict(r'C:\Users\Lenovo\Desktop\OIP.jpg')