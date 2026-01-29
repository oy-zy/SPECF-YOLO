
from ultralytics import YOLO

def main():
    # 加载模型
    model = YOLO(".../best.pt")  #/images/test/39.jpg',  MHDL_2025_5  - fsas_2025_ceshi fsas_2025  从头开始构建新模型\yolov8n-C2f_DCNv4.yaml\yolov8n-SNHL.yaml

    model.predict(source='D:/videos/csj-500/datasets/cs-xinceshiji/5.jpg' ,
                  save=True,
                  iou = 0.3)

if __name__ == '__main__':
    main()

