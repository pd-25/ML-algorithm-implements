import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

try:
  STOP_WORDS = set(stopwords.words('english'))
except LookupError:
  STOP_WORDS = set(ENGLISH_STOP_WORDS)

def transform_text(text: str):
  text = text.lower()
  text = nltk.tokenize.wordpunct_tokenize(text)
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    if i not in STOP_WORDS and i not in string.punctuation:
      y.append(i)

  text = y[:]
  y.clear()

  for i in text:
    y.append(ps.stem(i))
  return " ".join(y)

