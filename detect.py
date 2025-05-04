from ultralytics import YOLO

model = YOLO("model/weapon-detector.pt")
results = model("test-image.jpg", conf=0.3)
for result in results:
    print(result.boxes.xyxy, result.boxes.conf)
    result.show()