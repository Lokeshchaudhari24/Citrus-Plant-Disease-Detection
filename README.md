# 🍊 Citrus Plant Disease Detection using CNN

This is a mini-project developed by **Lokesh Chaudhari** that detects various citrus leaf diseases using Convolutional Neural Networks (CNN) with TensorFlow/Keras.

## 🧪 Objective

To automate the detection of citrus plant diseases using image classification techniques to help farmers and agricultural experts take quick and accurate actions.

## 📂 Dataset

- Source: Local dataset (`Leaves from kaggle/` directory)
- Categories:
  - Healthy
  - Canker
  - Greening
  - Sooty Mold
  - Scab

Images were resized to **255x255** pixels and batch loaded for training.

```python
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "Leaves",  
    shuffle=True,
    image_size=(255, 255),
    batch_size=32
)
