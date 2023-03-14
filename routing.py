from flask import Flask, request, jsonify, send_from_directory,render_template_string
from flask_cors import CORS
import markdown
import os

from predictWithBase64Img import predictionByBase64Img

app = Flask(__name__)
CORS(app)

@app.route('/')
def display_readme():
    with open("README.md", "r") as readme_file:
        content = readme_file.read()
        html_content = markdown.markdown(content)
        return render_template_string(html_content)


@app.route('/classify')
def home():
    print("Classify route.")
    return "<h1>This is classify route.</h1>"



@app.route('/classify', methods=['POST'])
def classify_image():
    data = request.get_json()
    img_b64 = data['image_base64']
    
    if not img_b64:
        print("No base64 image data provided.")
        return jsonify({"error": "No base64 image data provided"}), 400
    
    try:
        print("Classifying image...")
        result, class_percentage = predictionByBase64Img(img_b64)
        print("Classification result: ", result)
        print("Classification percentage: ", class_percentage)
        return jsonify({"classification": result, "percentage": str(class_percentage)}), 200
    except Exception as e:
        print("Error during classification: ", str(e))
        return jsonify({"error": f"Error during classification: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7654)