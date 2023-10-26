"""
@Project ：ClassicModelRebuild 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/23 21:52 
"""
from ultralytics import YOLO

model = YOLO("yolov8m.pt")


# results = model.predict(
#    source=['images/human1.jpg','images/human2.jpg'],
#    conf=0.25,
#    save=True,
#    imgsz=320
# )

results = model.train(data='/home/zbl/datasets/road_damage/road_damage.yaml',
                      epochs=100, imgsz=640)

