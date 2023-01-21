import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gradio as gr

nltk.download()

import pickle 
with open("Testing\model.pkl", 'rb') as file: #model
      clf =  pickle.load(file)

from joblib import dump, load
vectorizer = load('Testing\vectorizer.pkl')

l1 = ['Corporate Communications', 'Securities Settlement',
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
       'Forms', 'Definitions', 'Liquidity Risk', 'Money Services','Research']

def cleanResume(resumeText):
    stop_words = set(stopwords.words('english')) #creating a set of stop words
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

import pickle

def predFunc(name, text):
 
  concat=name+text
  cleaned=cleanResume(concat)
  transformed=vectorizer.transform([cleaned])
  pred=clf.predict(transformed)
  ans = pd.DataFrame(pred, columns = l1)
  
  l=ans.columns
  empty_list=[]
  for i, rows in ans.iterrows():
    for j in l:
      if rows[j]==1:
        empty_list.append(j)

  
  return (empty_list)

name="Renminbi RMB Haircut - February 4, 2020"
text="Pursuant to Section 2.6.2 of the Clearing House Procedures of HKFE Clearing Corporation Limited (HKCC) and Section 10.4.3.1 of the Operational Clearing Procedures of The SEHK Options Clearing House Limited (SEOCH), the Clearing Houses have determined to adjust the haircut on RMB deposited as margin collateral from 3.1% to 3.5% after the close of business on 7 February 2020. The haircut shall be applied on a daily basis to determine the value of any RMB allowed to be used as a cover for the margin requirements of HKCC and/or SEOCH participants for contracts with settlement currency prescribed in HKD or USD. Participants should make necessary funding arrangements to cover any shortfall to their margin requirements resulting from the adjustment of the RMB haircut."
print(predFunc(name,text))



demo = gr.Interface(predFunc,['text','text'],['text'])
demo.launch()