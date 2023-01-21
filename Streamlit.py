import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import streamlit as st
from joblib import dump, load
import pickle
import lightgbm

stop_words = ['a',
 'about',
 'above',
 'after',
 'again',
 'against',
 'ain',
 'all',
 'am',
 'an',
 'and',
 'any',
 'are',
 'aren',
 "aren't",
 'as',
 'at',
 'be',
 'because',
 'been',
 'before',
 'being',
 'below',
 'between',
 'both',
 'but',
 'by',
 'can',
 'couldn',
 "couldn't",
 'd',
 'did',
 'didn',
 "didn't",
 'do',
 'does',
 'doesn',
 "doesn't",
 'doing',
 'don',
 "don't",
 'down',
 'during',
 'each',
 'few',
 'for',
 'from',
 'further',
 'had',
 'hadn',
 "hadn't",
 'has',
 'hasn',
 "hasn't",
 'have',
 'haven',
 "haven't",
 'having',
 'he',
 'her',
 'here',
 'hers',
 'herself',
 'him',
 'himself',
 'his',
 'how',
 'i',
 'if',
 'in',
 'into',
 'is',
 'isn',
 "isn't",
 'it',
 "it's",
 'its',
 'itself',
 'just',
 'll',
 'm',
 'ma',
 'me',
 'mightn',
 "mightn't",
 'more',
 'most',
 'mustn',
 "mustn't",
 'my',
 'myself',
 'needn',
 "needn't",
 'no',
 'nor',
 'not',
 'now',
 'o',
 'of',
 'off',
 'on',
 'once',
 'only',
 'or',
 'other',
 'our',
 'ours',
 'ourselves',
 'out',
 'over',
 'own',
 're',
 's',
 'same',
 'shan',
 "shan't",
 'she',
 "she's",
 'should',
 "should've",
 'shouldn',
 "shouldn't",
 'so',
 'some',
 'such',
 't',
 'than',
 'that',
 "that'll",
 'the',
 'their',
 'theirs',
 'them',
 'themselves',
 'then',
 'there',
 'these',
 'they',
 'this',
 'those',
 'through',
 'to',
 'too',
 'under',
 'until',
 'up',
 've',
 'very',
 'was',
 'wasn',
 "wasn't",
 'we',
 'were',
 'weren',
 "weren't",
 'what',
 'when',
 'where',
 'which',
 'while',
 'who',
 'whom',
 'why',
 'will',
 'with',
 'won',
 "won't",
 'wouldn',
 "wouldn't",
 'y',
 'you',
 "you'd",
 "you'll",
 "you're",
 "you've",
 'your',
 'yours',
 'yourself',
 'yourselves']

def cleanResume(resumeText):
  
    resumeText=resumeText.lower()
    # resumeText = re.sub('http\S+\s*', ' ', resumeText)  # remove URLs
    # resumeText = re.sub('#\S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'\d*',r'', resumeText)  
    resumeText = re.sub('\s+', ' ', resumeText)
    word_tokens = word_tokenize(resumeText)
    filtered_sentence = [w for w in word_tokens if not w in stop_words] #Contains words other than stop words
    # tagged_corpus=pos_tag(filtered_sentence)
    # tagged_corpus_list=[prat_lemmatize(token, tag) for token, tag in tagged_corpus]
    resumeText=' '.join(filtered_sentence)
    return resumeText

y= ['Corporate Communications', 'Securities Settlement',
       'Antitrust', 'Financial Crime', 'Commodities Trading', 'Examinations',
       'Insurance', 'Required Disclosures', 'Consumer protection',
       'Market Risk', 'Natural Disasters', 'Securities Management',
       'Information Filing', 'Quotation', 'Financial Accounting',
       'Securities Clearing', 'Listing', 'Records Maintenance', 'Delivery',
       'Monetary and Economic Policy', 'Banking', 'Regulatory Actions',
       'Securities Sales', 'Compliance Management', 'Fees and Charges',
       'Licensing', 'Legal Proceedings', 'Corporate Governance', 'Exemptions',
       'Legal', 'Contract Provisions', 'Payments and Settlements', 'IT Risk',
       'Trade Pricing', 'Licensure and certification', 'Trade Settlement',
       'Market Abuse', 'Regulatory Reporting', 'Powers and Duties',
       'Money-Laundering and Terrorist Financing', 'Accounting and Finance',
       'Fraud', 'Broker Dealer', 'Securities Issuing', 'Risk Management',
       'Forms', 'Definitions', 'Liquidity Risk', 'Money Services', "Research"]

def predFunc(name, text, model):
  concat=name+text
  cleaned=cleanResume(concat)
  vectorizer=load('vectorizer.pkl')

  
  transformed=vectorizer.transform([cleaned])
  print(transformed)
  pred=model.predict(transformed)
  ans = pd.DataFrame(pred, columns = y.columns)
  
  l=ans.columns
  empty_list=[]
  for i, rows in ans.iterrows():
    for j in l:
      if rows[j]==1:
        empty_list.append(j)

  
  return empty_list

# Title
st.header("Regulatory Compliance Prediction")

name = st.text_input("Enter Name of Regulation")
desc = st.text_input("Enter Description Regulation")

if st.button("Submit"):
    with open("model.pkl", 'rb') as file: #model
      clf =  pickle.load(file) #picckle load
    l=predFunc(name, desc, clf)

    for i in l:
        st.text(f"The output: {i}")