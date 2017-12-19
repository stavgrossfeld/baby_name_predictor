
# export FLASK_APP=serving_model.py
# run flask


from flask import Flask, request, render_template
import pandas as pd
import nltk

from sklearn.tree import DecisionTreeClassifier
#from sklearn.cross_validation import train_test_split

import pickle
f = open('feature_names.pickle', 'rb')
FEATURE_NAMES = pickle.load(f)
f.close()
f = open('name_classifier.pickle', 'rb')
model = pickle.load(f)
f.close()

app = Flask(__name__)

# predict on any name code
def predict_name(name, clf):
    """predict name and pass classifier """
    name_grams = []
    for gram in nltk.ngrams(name, 2):
      name_grams.append("".join(gram))

    feature_names = [col for col in FEATURE_NAMES if col not in ["is_girl", "names"]]

    pred_features = pd.DataFrame(pd.Series({feature: 1 if feature in name_grams else 0 for feature in feature_names})).transpose()

    #print clf.predict(pred_features)[0]
    return clf.predict(pred_features)[0]


app = Flask(__name__)

@app.route('/', methods = ["GET","POST"])
def serve_template():
  if request.method == 'GET':
    return render_template('form.html')
  else:
    clf = model
    baby_name = request.get_data()
    baby_names = str(baby_name).replace("'","").replace("%2C"," ").split("=")[1].rstrip().split("+")
    predicted_list = [(name, predict_name(name.lower(), clf)) for name in baby_names]
    #print predicted_list
    pred_df = pd.DataFrame([pred for pred in predicted_list])
    pred_df.columns = ["name","is_girl"]
    return pred_df.to_html()

@app.route("/gender/<message>")
def gender(message):
    clf = model
    baby_name = str(message)
    baby_names = baby_name.replace("+"," ").replace("'","").replace("%2C"," ").split(" ")
    predicted_list = [(name, predict_name(name.lower(), clf)) for name in baby_names]
    #print predicted_list
    pred_df = pd.DataFrame([pred for pred in predicted_list])
    pred_df.columns = ["name","is_girl"]
    return pred_df.to_html()


if __name__ == "__main__":
  app.run()
