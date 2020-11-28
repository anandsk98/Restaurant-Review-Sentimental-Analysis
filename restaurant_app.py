from flask import Flask, render_template, request
import pickle
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer

filename='restaurant-review-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

bow='bow-vectorizer.pkl'
bow = pickle.load(open(bow, 'rb'))

app = Flask(__name__)



@app.route('/')
def home():
    return render_template('home.html')


@app.route('/review', methods=['POST'])
def man():
   if request.method == 'POST':
       
        review =request.form['review']
        ps = PorterStemmer()
        corpus = []
    
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
        
        bow1=bow.transform(corpus).toarray()
        pred = classifier.predict(bow1)
        
        return render_template('after.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)

