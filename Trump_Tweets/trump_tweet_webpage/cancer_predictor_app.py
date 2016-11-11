from flask import Flask, jsonify, request, render_template

from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

#---------- MODEL IN MEMORY ----------------#

# Read the scientific data on breast cancer survival,
# Build a LogisticRegression predictor on it
patients = pd.read_csv("haberman.data", header=None)
patients.columns=['age','year','nodes','survived']
patients=patients.replace(2,0)  # The value 2 means death in 5 years

X = patients[['age','year','nodes']]
Y = patients['survived']
PREDICTOR = LogisticRegression().fit(X,Y)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = Flask(__name__)

# Homepage
@app.route("/")
def index():
    """
    Homepage: serve our visualization page, index.html
    """
    return render_template("index.html")

# Get an example and return it's score from the predictor model
@app.route("/score/", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = request.json
    x = np.matrix(data["example"])
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
