from ultralytics import YOLO

if __name__ =='__main__':

     model =YOLO("yolov8n.yaml").load("yolov8n.pt")
     model.train(data=r"G:\py\pythonProjectdemo\ultralytics-main\datasets\apple.yaml",imgsz=640,epochs=20,batch=16)