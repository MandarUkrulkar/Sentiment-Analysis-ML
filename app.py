
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    sample_review = request.form.get('iptext')
    import pandas as pd
    df3=pd.read_csv('updated_reviews_with_scraped_edited.csv')
    df3=df3.dropna()
    df3=df3.reset_index(drop=True) 
    #creating target variable
    X=df3['Cleaned reviews']
    y=df3.Liked


    #Using  vectorizer which convets words into vectors.

    from sklearn.feature_extraction.text import CountVectorizer
    Cvect = CountVectorizer()

    X= Cvect.fit_transform(df3['Cleaned reviews']).toarray()

    #Creating a train test split from data
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.1,random_state=100)

    import pickle
    model = pickle.load(open('Model.pkl','rb'))

    import re   # Regular Expression package
    def predict_sentiment(sample_review):
        sample_review = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_review)
        sample_review = sample_review.lower()
        temp = Cvect.transform([sample_review]).toarray()
        return model.predict(temp)

    
    if predict_sentiment(sample_review):
        output= 'possitive'
    else:
        output= 'negative'
    
    return render_template('index.html', prediction_text=f'This review : "{sample_review} " is {output}')



if __name__ == "__main__":
    app.run(debug=True)