import tensorflow as tf

# Load model
model = tf.keras.models.load_model("model/brain_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save file
with open("model/brain_model.tflite", "wb") as f:
    f.write(tflite_model)

print(" Converted to TFLite!")