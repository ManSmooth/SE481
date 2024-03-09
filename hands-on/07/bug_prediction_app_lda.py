from flask import Flask, request
from scipy.sparse import hstack
import pickle
import string
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def preprocess(text, stopword_set, stemmer):
    cleaned_text = text.translate(str.maketrans('', '', '!"#$%&\'()*+,.<=>?@[]^`{|}~' + u'\xa0'))
    cleaned_text = cleaned_text.lower()
    cleaned_text = cleaned_text.translate(str.maketrans(string.whitespace, ' ' * len(string.whitespace), ''))
    cleaned_text = ' '.join(['_variable_with_underscore' if '_' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_variable_with_dash' if '-' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_long_variable_name' if len(t) > 15 and t[0] != '#' else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_weburl' if t.startswith('http') and '/' in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_number' if re.sub('[\\/;:_-]', '', t).isdigit() else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_variable_    with_address' if re.match('.*0x[0-9a-f].*', t) else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_name_with_number' if re.match('.*[a-f]*:[0-9]*', t) else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_number_starts_with_one_character' if re.match('[a-f][0-9].*', t) else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_number_starts_with_three_characters' if re.match('[a-f]{3}[0-9].*', t) else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_version' if any(i.isdigit() for i in t) and t.startswith('v') else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_localpath' if ('\\' in t or '/' in t) and ':' not in t else t for t in cleaned_text.split()])
    cleaned_text = ' '.join(['_image_size' if t.endswith('px') else t for t in cleaned_text.split()])
    tokenized_text = word_tokenize(cleaned_text)

    sw_removed_text = [word for word in tokenized_text if word not in stopword_set]
    sw_removed_text = [word for word in sw_removed_text if len(word) > 2]
    stemmed_text = ' '.join([stemmer.stem(w) for w in sw_removed_text])

    return stemmed_text


app = Flask(__name__)
app.tfidf_vectorizer = pickle.load(open('resource/github_bug_prediction_tfidf_vectorizer.pkl', 'rb'))
app.basic_model = pickle.load(open('resource/github_bug_prediction_basic_model.pkl', 'rb'))
app.stopword_set = set(stopwords.words())
app.stemmer = PorterStemmer()

@app.route('/predict_basic', methods=['GET'])
def predict_basic():
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    title = argList['title'][0]
    body = argList['body'][0]
    predict = app.basic_model.predict_proba(hstack([app.tfidf_vectorizer.transform([preprocess(' '.join([title, body]), app.stopword_set, app.stemmer)])]))
    response_object['predict_as'] = 'bug' if predict[0][1] > 0.5 else 'not bug'
    response_object['bug_prob'] = predict[0][1]
    return response_object

if __name__ == '__main__':
    app.run(debug=False)
