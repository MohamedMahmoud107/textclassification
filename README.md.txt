After I finish training the models, I save one of the best models and the vectorizer using joblib. I then use them later to construct a Streamlit app.

The app first checks if the text is clean and provides the user with three options:

1.Upload a file

2.Use a URL

3.Write the document manually

After the user provides the input 
if the text is not in English
the app will translate it to English using deep_translator (ensure your network connection is active before using it).

The output will display:

The predicted class of the text

The confidence score (the probability that the text belongs to a specific class)