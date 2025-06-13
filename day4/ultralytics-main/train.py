from ultralytics import YOLO

if __name__ =='__main__':

     model =YOLO("yolov8n.yaml").load("yolov8n.pt")
     model.train(data=r"G:\py\pythonProjectdemo\day4\ultralytics-main\datasets_cups\cups.yaml",imgsz=640,epochs=20,batch=16)