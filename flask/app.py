import json
import nltk
import random
import pandas as pd
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification

# # Package sentence tokenizer
# nltk.download('punkt')
# # Package lemmatization
# nltk.download('wordnet')
# # Package multilingual wordnet data
# nltk.download('omw-1.4')


# Importing the dataset
with open('../kampus_merdeka_cmplt.json') as content:
  data1 = json.load(content)

# Mendapatkan semua data ke dalam list
tags = [] # data tag
inputs = [] # data input atau pattern
responses = {} # data respon
words = [] # Data kata
classes = [] # Data Kelas atau Tag
documents = [] # Data Kalimat Dokumen
ignore_words = ['?', '!'] # Mengabaikan tanda spesial karakter

for intent in data1['intents']:
  responses[intent['tag']]=intent['responses']
  for lines in intent['patterns']:
    inputs.append(lines)
    tags.append(intent['tag'])
    for pattern in intent['patterns']:
      w = nltk.word_tokenize(pattern)
      words.extend(w)
      documents.append((w, intent['tag']))
      # add to our classes list
      if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Konversi data json ke dalam dataframe
data = pd.DataFrame({"patterns":inputs, "tags":tags})

label = data['tags'].values.flatten().tolist()
label = list(dict.fromkeys(label))

intents = json.loads(open('../kampus_merdeka_cmplt.json').read())

def fetch_data():
    saved_directory = '../model'
    tokenizer = AutoTokenizer.from_pretrained(saved_directory)
    model = TFAutoModelForSequenceClassification.from_pretrained(saved_directory)

    return tokenizer, model

def preprocess(text):
    # define stopword for bahasa
    file_path = '../stopwordbahasa.csv'
    df = pd.read_csv(file_path)

    # Ubah DataFrame menjadi list array menggunakan fungsi values
    stop_words = df.values.flatten().tolist()

    #Lowercase the text
    text = text.lower()

    #Split the text into words
    words = text.split()

    # Perform stemming
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

def predict(input_text, model):
    labels = label
    input_text = preprocess(input_text)
    new_X = dict(tokenizer(input_text, padding=True, truncation=True, max_length=50, return_tensors='tf'))
    predictions = model.predict(new_X)
    predictions = labels[tf.argmax(predictions['logits'][0].tolist())]
    return predictions

def getResponse(predictions, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == predictions):
            result = random.choice(i['responses'])
            print(result)
            break
        else:
            result = "Maaf chatbot tidak mengerti, coba tanyakan lagi"
    return result

def chatbot_response(msg):
    ints = predict(msg, model)
    print(ints)
    res = getResponse(ints, intents)
    print(res)
    return res

tokenizer, model = fetch_data()

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    print(userText)
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run()