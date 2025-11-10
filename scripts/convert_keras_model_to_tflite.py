# Sur votre PC, avec tensorflow complet installé
import tensorflow as tf
model = tf.keras.models.load_model("models/modelCNN.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Ajoutez des optimisations si nécessaire
tflite_model = converter.convert()
with open("models/modelCNN.tflite", "wb") as f:
    f.write(tflite_model)