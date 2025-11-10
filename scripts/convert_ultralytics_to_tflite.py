# Sur votre PC, avec ultralytics installé
from ultralytics import YOLO
model = YOLO("models/model.pt")
model.export(format="tflite")  # Cela créera un model_float32.tflite