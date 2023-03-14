import numpy as np
import tensorflow as tf
import cv2
import base64

def load_model(model_path):
    # Load TensorFlow Lite model.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def get_input_output_tensors(interpreter):
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return input_details, output_details

def preprocess_image(img_b64, input_shape):
    # Decode the base64 image.
    img_np = np.frombuffer(base64.b64decode(img_b64), dtype=np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)

    # Convert the image to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize the image to match the input tensor shape.
    resized = cv2.resize(gray, (input_shape[1], input_shape[2]))

    # Convert the grayscale image to RGB with 3 channels.
    input_data = cv2.cvtColor(np.array(resized, dtype=np.float32), cv2.COLOR_GRAY2RGB)

    # Normalize the image data.
    input_data = input_data / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

def classify_image(interpreter, input_details, input_data):
    # Set the input tensor.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference with the TensorFlow Lite model.
    interpreter.invoke()

    # Get the output tensor.
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Get the most probable classification result.
    class_id = np.argmax(output_data)

    # Get the probabilities for each class.
    probabilities = output_data[0]

    return class_id, probabilities


def predictionByBase64Img(img_b64):
    model_path = "model.tflite"
    class_names = ['Tree', 'Butterfly', 'Leaf', 'Sun', 'Platypus', 'Clothes', 'Flower', 'IceCream']

    interpreter = load_model(model_path)
    input_details, output_details = get_input_output_tensors(interpreter)
    input_shape = input_details[0]['shape']
    input_data = preprocess_image(img_b64, input_shape)
    class_id, probabilities = classify_image(interpreter, input_details, input_data)

    # Convert probabilities to percentages.
    percentages = probabilities * 100

    # Get the percentage for the predicted class.
    class_percentage = percentages[class_id]

    return class_names[class_id], class_percentage

# if __name__ == '__main__':
#     model_path = "/Users/jarawin/Desktop/combi/model.tflite"
#     img_b64 = "your_image_base64_string"
#     class_names = ['Tree', 'Butterfly', 'Leaf', 'Sun', 'Platypus', 'Clothes', 'Flower', 'IceCream']

#     interpreter = load_model(model_path)
#     input_details, output_details = get_input_output_tensors(interpreter)
#     input_shape = input_details[0]['shape']
#     input_data = preprocess_image(img_b64, input_shape)
#     class_id = classify_image(interpreter, input_details, input_data)
#     print("The image is classified as:", class_names[class_id])