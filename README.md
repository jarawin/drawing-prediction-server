# Image Classification Project

This project is an image classification model using Keras and TensorFlow. It contains the required Python libraries listed in the `requirements.txt` file.

## Installation

Follow these steps to set up the project:

1. Make sure you have Python 3.6 or newer installed on your system. You can download it from the [official Python website](https://www.python.org/downloads/).

2. (Optional) It is recommended to create a virtual environment for this project to avoid conflicts with other installed packages. You can create a virtual environment using the following command:
   python -m venv venv

3. Activate the virtual environment:

   - For Windows:
     venv\Scripts\activate

   - For macOS/Linux:
     source venv/bin/activate

4. Install the required packages using the `requirements.txt` file:

   pip install -r requirements.txt

5. Run the Python script that contains the image classification model.

   - For Create a new model:
     python createModel.py

   - For Load a model and predict by image:
     python predicWithImg.py

   - For route to predict by image in flask server:
     python app.py

## Usage

After installing the required packages, you can run your Python script that contains the image classification model.

Make sure your virtual environment is active when running the script.

## License

This project is licensed under the terms of the MIT license.
