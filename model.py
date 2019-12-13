import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.models import Model, Sequential
from keras import initializers, regularizers, constraints, optimizers, layers

df1 = pd.read_csv('labeledTrainData.tsv', delimiter="\t")
df1 = df1.drop(['id'], axis=1)

df2 = pd.read_csv('IMDB Dataset.csv',encoding="latin-1")
df2.columns = ["review","sentiment"]

df2 = df2[df2.sentiment != 'unsupplied']
df2['sentiment'] = df2['sentiment'].map({'positive': 1, 'negative': 0})
df = pd.concat([df1, df2]).reset_index(drop=True)

def clean_text(text):
    TAG_RE = re.compile(r'<[^>]+>')
    text = TAG_RE.sub('', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = " ".join(text)
    return text

df['Processed_Reviews'] = df.review.apply(lambda x: clean_text(x))

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['Processed_Reviews'])
list_tokenized_train = tokenizer.texts_to_sequences(df['Processed_Reviews'])

maxlen = 130
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
y = df['sentiment']

embed_size = 128
model = Sequential()
model.add(Embedding(max_features, embed_size))
model.add(GRU(units=32,  dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
epochs = 8
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.save("ci.h5")
