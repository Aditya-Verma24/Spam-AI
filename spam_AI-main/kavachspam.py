#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd


# In[14]:


df=pd.read_csv(r"D:Users\User\Downloads\spam.csv" ,encoding = "ISO-8859-1")


# In[18]:


df.head()


# In[19]:


df.shape


# In[20]:


#1 Data Cleaning #2 eda #3 text preprocessing #4 model building #6 improve #7 website #8 deployment


# In[21]:


#Data cleaning 
df.info()


# In[22]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[23]:


df.sample(5)


# In[24]:


#rename the columns 
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[25]:


df.sample(5)


# In[26]:


from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()


# In[27]:


df['target']=encoder.fit_transform(df['target'])


# In[28]:


df.head()


# In[29]:


#checking missing values
df.isnull().sum()


# In[30]:


#checking duplicate values
df.duplicated().sum()


# In[31]:


#remove duplicates
df=df.drop_duplicates(keep='first')


# In[32]:


df.duplicated().sum()


# In[33]:


#EDA-exploratory data analysis
df.head()


# In[34]:


df['target'].value_counts()


# In[35]:


import matplotlib.pyplot as plt
plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct="%0.2f")


# In[36]:


#data is imbalanced , lets balance it 
import nltk
get_ipython().system('pip install nltk')


# In[37]:


nltk.download('punkt')


# In[38]:


df['num_character']=df['text'].apply(len)


# In[39]:


df.head()


# In[40]:


#fetching the no.of words 
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[41]:


df.head()


# In[42]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[43]:


df.head()


# In[44]:


df[['num_character','num_words','num_sentences']].describe()


# In[45]:


#ham message
df[df['target']==0][['num_character','num_words','num_sentences']].describe()


# In[46]:


#spam message
df[df['target']==1][['num_character','num_words','num_sentences']].describe()


# In[47]:


import seaborn as sns


# In[48]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_character'])
sns.histplot(df[df['target']==1]['num_character'],color='purple')


# In[49]:


plt.figure(figsize=(12,6))
sns.histplot(df[df['target']==0]['num_words'])
sns.histplot(df[df['target']==1]['num_words'],color='purple')


# In[50]:


#pearson correlation coefficient to figure out the relation between these two 


# In[51]:


sns.pairplot(df,hue='target')


# In[52]:


sns.heatmap(df.corr(),annot=True)


# In[53]:


#3 text/data preprocessing
# lower case 
# tokenization
#removing special character
#removing stop words and punctuation
#stemming
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
           
            
    text=y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text=y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
            
    return " ".join(y)


# In[54]:


import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words('english')


# In[55]:


import string
string.punctuation


# In[67]:


transform_text('Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...')


# In[68]:


df['text'][0]


# In[69]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[70]:


df['transformed_text']=df['text'].apply(transform_text)


# In[71]:


df.head()


# In[61]:


get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
wc=WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[72]:


spam_wc=wc.generate(df[df['target']==1]['transformed_text'].str.cat(sep=" "))


# In[73]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[74]:


ham_wc=wc.generate(df[df['target']==0]['transformed_text'].str.cat(sep=" "))


# In[109]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[110]:


df.head()


# In[75]:


spam_corpus=[]
for msg in df[df['target']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)
        


# In[76]:


len(spam_corpus)


# In[120]:


get_ipython().system('pip install collections')
from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[77]:


ham_corpus=[]
for msg in df[df['target']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[122]:


len(ham_corpus)


# In[123]:


sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[78]:


#4 model building 
# NAIVE-BAYES ALGO 
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv=CountVectorizer()
tfidf=TfidfVectorizer()


# In[79]:


X=tfidf.fit_transform(df['transformed_text']).toarray()


# In[80]:


X.shape


# In[81]:


y=df['target'].values


# In[82]:


y


# In[83]:


from sklearn.model_selection import train_test_split


# In[84]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[85]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[86]:


gnb=GaussianNB()
mnb=MultinomialNB()
bnb=BernoulliNB()


# In[87]:


gnb.fit(X_train,y_train)
y_pred1=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[88]:


mnb.fit(X_train,y_train)
y_pred2=mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[90]:


bnb.fit(X_train,y_train)
y_pred3=gnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[91]:


#tfidf --> mnb 


# In[95]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# In[96]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier()
mnb = MultinomialNB()
dtc = DecisionTreeClassifier(max_depth=5)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbdt = GradientBoostingClassifier(n_estimators=50,random_state=2)


# In[97]:


clfs = {
    'SVC' : svc,
    'KN' : knc, 
    'NB': mnb, 
    'DT': dtc, 
    'LR': lrc, 
    'RF': rfc, 
    'AdaBoost': abc, 
    'BgC': bc, 
    'ETC': etc,
    'GBDT':gbdt,
}    


# In[98]:


def train_classifier(clf,X_train,y_train,X_test,y_test):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    
    return accuracy,precision


# In[99]:


train_classifier(svc,X_train,y_train,X_test,y_test)


# In[100]:


accuracy_scores = []
precision_scores = []

for name,clf in clfs.items():
    
    current_accuracy,current_precision = train_classifier(clf, X_train,y_train,X_test,y_test)
    
    print("For ",name)
    print("Accuracy - ",current_accuracy)
    print("Precision - ",current_precision)
    
    accuracy_scores.append(current_accuracy)
    precision_scores.append(current_precision)


# In[101]:


performance_df = pd.DataFrame({'Algorithm':clfs.keys(),'Accuracy':accuracy_scores,'Precision':precision_scores}).sort_values('Precision',ascending=False)


# In[102]:


performance_df


# In[103]:


performance_df1 = pd.melt(performance_df, id_vars = "Algorithm")


# In[104]:


performance_df1


# In[105]:


sns.catplot(x = 'Algorithm', y='value', 
               hue = 'variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation='vertical')
plt.show()


# In[106]:


# model improve
# 1. Change the max_features parameter of TfIdf


# In[110]:


# Voting Classifier
svc = SVC(kernel='sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[111]:


voting = VotingClassifier(estimators=[('svm', svc), ('nb', mnb), ('et', etc)],voting='soft')


# In[112]:


voting.fit(X_train,y_train)


# In[113]:


y_pred = voting.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[114]:


# Applying stacking
estimators=[('svm', svc), ('nb', mnb), ('et', etc)]
final_estimator=RandomForestClassifier()


# In[115]:


from sklearn.ensemble import StackingClassifier


# In[116]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[118]:


clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Accuracy",accuracy_score(y_test,y_pred))
print("Precision",precision_score(y_test,y_pred))


# In[119]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:




