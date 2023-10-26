"""
@Project ：ClassicModelRebuild 
@File    ：val.py
@IDE     ：PyCharm 
@Author  ：paul623
@Date    ：2023/10/24 22:02 
"""
from ultralytics import YOLO

model = YOLO("/home/zbl/codeLab/remotePython/ClassicModelRebuild/YOLO/runs/detect/train3/weights/best.pt")

# metrics = model.val()
#
# metrics.box.map50

imgs = ['/home/zbl/datasets/road_damage_kaggle/China_Drone_000205_jpg.rf.1ab3c3aa2df7f52d88d0922a6c522544.jpg',
        '/home/zbl/datasets/road_damage_kaggle/China_Drone_000061_jpg.rf.b5f27ce4ae8c61695855c0a83b1bdd7a.jpg',
        '/home/zbl/datasets/road_damage_kaggle/China_Drone_000048_jpg.rf.19d49b225c53f19fabb5ecda9a7281f1.jpg',
        '/home/zbl/datasets/road_damage_kaggle/example1.jpg',
        '/home/zbl/datasets/road_damage_kaggle/79-out_ori.jpg',
        '/home/zbl/datasets/road_damage_kaggle/61-out_ori.jpg',
        '/home/zbl/datasets/road_damage_kaggle/50-out_ori.jpg',
        '/home/zbl/datasets/road_damage_kaggle/48-out_ori.jpg',
        '/home/zbl/datasets/road_damage_kaggle/24629-out_ori.jpg',
        '/home/zbl/datasets/road_damage_kaggle/24628-out_ori.jpg',
        '/home/zbl/datasets/road_damage_kaggle/IMG_20231025_221134.jpg',
        '/home/zbl/datasets/road_damage_kaggle/IMG_20231025_221124.jpg']

results = model.predict(
   source=imgs,
   conf=0.25,
   save=True,
   imgsz=640
)
