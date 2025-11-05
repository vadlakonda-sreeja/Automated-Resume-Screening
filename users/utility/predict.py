import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from django.conf import settings
path = settings.MEDIA_ROOT + '\\' + 'UpdatedResumeDataSet.csv'
data = pd.read_csv(path)
def cleanResume(resumeText):
    resumeText = re.sub('httpS+s*', ' ', resumeText)  # remove URLs
    resumeText = re.sub('RT|cc', ' ', resumeText)  # remove RT and cc
    resumeText = re.sub('#S+', '', resumeText)  # remove hashtags
    resumeText = re.sub('@S+', '  ', resumeText)  # remove mentions
    resumeText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[]^_`{|}~"""), ' ', resumeText)  # remove punctuations
    resumeText = re.sub(r'[^x00-x7f]',r' ', resumeText) 
    resumeText = re.sub('s+', ' ', resumeText)  # remove extra whitespace
    return resumeText
data['cleaned_resume'] = data.Resume.apply(lambda x: cleanResume(x))
var_mod = ['Category']
le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
requiredText = data['cleaned_resume'].values
requiredTarget = data['Category'].values
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    stop_words='english',
    max_features=1500)
word_vectorizer.fit(requiredText)
WordFeatures = word_vectorizer.transform(requiredText)
X_train,X_test,y_train,y_test = train_test_split(WordFeatures,requiredTarget,random_state=0, test_size=0.2)


def training_knn():
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf = OneVsRestClassifier(KNeighborsClassifier())
    clf.fit(X_train, y_train)
    y_pred_knn = clf.predict(X_test)
    accuracy_score(y_test, y_pred_knn)
    print("Classification Accuracy:", accuracy_score(y_test, y_pred_knn))
    print("Classification Report\n")
    nb_cr = classification_report(y_test, y_pred_knn,output_dict=True)
    print(classification_report(y_test, y_pred_knn))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_knn))
    cm = confusion_matrix(y_test, y_pred_knn)
    nb_cr = pd.DataFrame(nb_cr).transpose()
    nb_cr = pd.DataFrame(nb_cr)
    return nb_cr.to_html

def training_NB():
    from sklearn.naive_bayes import MultinomialNB
    naive = MultinomialNB()
    naive.fit(X_train,y_train)
    y_pred_nb = naive.predict(X_test)
    naive.fit(X_train, y_train)
    y_pred_nb = naive.predict(X_test)
    accuracy_score(y_test, y_pred_nb)
    print("Classification Accuracy:", accuracy_score(y_test, y_pred_nb))
    print("Classification Report\n")
    nb_cr = classification_report(y_test, y_pred_nb,output_dict=True)
    print(classification_report(y_test, y_pred_nb))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, y_pred_nb))
    cm = confusion_matrix(y_test, y_pred_nb)
    nb_cr = pd.DataFrame(nb_cr).transpose()
    nb_cr = pd.DataFrame(nb_cr)
    return nb_cr.to_html

def training_SVM():
    from sklearn.svm import SVC
    svm = SVC()
    svm.fit(X_train,y_train)
    pred_svm = svm.predict(X_test)
    accuracy_score(y_test, pred_svm)
    print("Classification Accuracy:", accuracy_score(y_test, pred_svm))
    print("Classification Report\n")
    nb_cr = classification_report(y_test, pred_svm,output_dict=True)
    print(classification_report(y_test, pred_svm))
    print("Confusion Matrix\n")
    print(confusion_matrix(y_test, pred_svm))
    cm = confusion_matrix(y_test, pred_svm)
    nb_cr = pd.DataFrame(nb_cr).transpose()
    nb_cr = pd.DataFrame(nb_cr)
    return nb_cr.to_html

def prediction(details):
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    clf = OneVsRestClassifier(KNeighborsClassifier())
    input = [details]
    print(input)
    clf.fit(X_train, y_train)
    vec = word_vectorizer.transform(input)
   
    x = vec.toarray()
        
    prediction = clf.predict(x)[0]
    minta = le.classes_
    result = minta[prediction]
    return result

def pdf(filename):
    from PyPDF2 import PdfFileReader

    # open the PDF file
    pdfFile = open(filename, 'rb')

    # create PDFFileReader object to read the file
    pdfReader = PdfFileReader(pdfFile)

    print("PDF File name: " + str(pdfReader.getDocumentInfo().title))
    print("PDF File created by: " + str(pdfReader.getDocumentInfo().creator))
    print("- - - - - - - - - - - - - - - - - - - -")

    numOfPages = pdfReader.getNumPages()
    output = []
    for i in range(0, numOfPages):



        print("Page Number: " + str(i))

        print("- - - - - - - - - - - - - - - - - - - -")
        pageObj = pdfReader.getPage(i)
        #print(pageObj.extractText())
        print("- - - - - - - - - - - - - - - - - - - -")
        output.append(pageObj.extractText())
        print(output)

    # close the PDF file object
    pdfFile.close()
    vec = word_vectorizer.transform(requiredText)
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    clf = OneVsRestClassifier(KNeighborsClassifier())
    print(input)
    clf.fit(X_train, y_train)
    print(type(vec))
    x = vec.toarray()
    prediction = clf.predict(x)[0]
    minta = le.classes_
    result = minta[prediction]
    return result

   