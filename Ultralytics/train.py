from ultralytics import YOLO

model = YOLO("yolov11x-seg.pt")

model.train(data="data.yaml", epochs=100)