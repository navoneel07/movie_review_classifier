import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import keras
import tensorflow
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

reviews = [""]

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

reviews = [clean_text(review) for review in reviews]

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(reviews)
list_tokenized_train = tokenizer.texts_to_sequences(reviews)

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

model = keras.models.load_model("ci.h5")

scores = model.predict(X_t)

decision_list = []
for i in range(len(reviews)):
    if(scores[i][0]>=0.5):
        decision_list.append("positive")
    else:
        decision_list.append("negative")
print(decision_list)
