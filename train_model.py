#######导入包#######
from ultralytics import YOLO
#######加载预训练模型#######
model = YOLO('yolo_sources/yolov8n.pt')
#######训练模型#######
model.train(
    data = 'data.yaml',
    epochs = 200,
    imgsz = 640,
    batch = 4,
    device = 0,
    workers = 0
)
