from ultralytics import YOLO

# Charger le modèle YOLOv8
model = YOLO('model.pt')

# Exporter le modèle au format ONNX
# (Pas de tflite, pas de int8 pour l'instant)
model.export(format='onnx', imgsz=640) 

print("Conversion en ONNX (model.onnx) terminée avec succès !")