
from ultralytics import YOLO

def main():
    # 加载模型
    model = YOLO("./ultralytics/models/v8/yolov8n.yaml")  #  MHDL_2025_5  - fsas_2025_ceshi fsas_2025  从头开始构建新模型\yolov8n-C2f_DCNv4.yaml\yolov8n-SNHL.yaml

    #model = YOLO("./ablation/train_2025_100_N_sgj_1/weights/last.pt")
    # 使用模型
    model.train(data="./ultralytics/datasets/YOLO-suogoujian0.yaml",
                epochs=150,
                device='cuda:0',
                imgsz=640,
                batch=8,
                verbose=True,
                amp=False,
                optimizer='SGD',
                project="RUOD",
                name="train_2025_500_N_sgjxin_3",
                #resume=True, # 如过想续训就设置last.pt的地址
                profile=True,)  # 训练模型 device='cuda:0',
    metrics = model.val(name="val_2025_500_N_sgjxin_3")  # 在验证集上评估模型性能

if __name__ == '__main__':
    main()
##############################################scsa

