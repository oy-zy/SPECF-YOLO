import warnings

warnings.filterwarnings('ignore')

result_name = 'val_2025_sgj_100_N_fsas_12'
if __name__ == '__main__':
    from ultralytics import YOLO
    model = YOLO('F:/SPECF-YOLO/Ablation/train_2025_sgj_100_N_fsas12/weights/best.pt')
    model.val(data='./ultralytics/datasets/YOLO-suogoujian0.yaml',
              split='val',
              imgsz=640,
              batch=16,
              iou=0.6,
              conf=0.001,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='ablation',
              name=result_name,
              workers=0,
              )
