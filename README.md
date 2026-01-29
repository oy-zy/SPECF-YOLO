# SPECF-YOLO
<p align="center">
  <img src="https://github.com/oy-zy/image/blob/main/1230858173bf3ed33b4796f51b2c0728.png" alt="SPECF-YOLO">
</p>
(a)SCWM & FFIEM.
<p align="center">
  <img src="https://github.com/oy-zy/image/blob/main/e04db8b97bb3c72db5609ea4ffab5fe9.png" alt="SPECF-YOLO">
</p>
(b)FLGRomer.
<p align="center">
  <img src="https://github.com/oy-zy/image/blob/main/fa7594bf10666044d3ea39eb6f8c5f80.png" alt="SPECF-YOLO">
</p>
(c) PBBNet

## Prerequisites
We strongly recommend Anaconda as the environment.

Python: 3.8

PyTorch: 1.11

CUDA: 11.3

## Evaluation
1. We are providing our pretrained model, and the evaluation code can be used without the training. Download pretrained model from Mega: [link](https://mega.nz/file/fB8x0BgZ#X2hnO_fnn3fAUrLxHwn5zWICjpxYQyYU0hwxAGlL26E).

2. Modify the model directory in the test.py file.

3. Evaluate the model.
```
python main_val.py
```

## Training
1. Configure the files required to run, and modify the root path in the "train.py" based on your dataset location.  

2. Run train.py.
```
python main_train.py
```  
