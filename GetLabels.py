from flask import Flask, request, jsonify
from Labels import Labels
from GetOutputFromMLModels import return_labels
import json
app = Flask(__name__)

@app.route("/labels")
def get_labels():
    # take the input image and text
    #img = request.files['image']
    #description = request.description['desc']
    # call the function written by Mingwei and Luis
    l= Labels()
    l = return_labels()
    jsonStr = json.dumps(l.__dict__)
    # Package the response and send it back
    return jsonStr
