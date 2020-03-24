import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def preprocessing(path):
    column_names = ['name','gender','probabilityBeMale']
    df = pd.read_csv(path, names=column_names)

    df_names = df
    df_names.gender.replace({'M':1, 'F':0}, inplace = True)
    df_names.gender.unique()
    return df_names


def mnb(df):
    Xfeatures = df['name']
    cv = CountVectorizer()
    X = cv.fit_transform(Xfeatures)
    y = df.gender
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)

    mnb_model = open("./models/mnb_model.pkl", "wb")
    cv_model = open("./models/cv_model.pkl", "wb")
    joblib.dump(clf, mnb_model)
    joblib.dump(cv, cv_model)
    mnb_model.close()
    cv_model.close()


# Accuracy of our Model
    print("Accuracy of Model",clf.score(X_test,y_test)*100,"%")

# Accuracy of our Model
    print("Accuracy of Model",clf.score(X_train,y_train)*100,"%")

    return cv,clf


# By Analogy most female names ends in 'A' or 'E' or has the sound of 'A'
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }


def vectorizer(df_names):

# Extract the features for the dataset
    df_X = features(df_names['name'])

    df_y = df_names['gender']

    # corpus = features(["Mike", "Julia"])
    # dv = DictVectorizer()
    # dv.fit(corpus)
    # transformed = dv.transform(corpus)


    dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.33, random_state=42)

    dv = DictVectorizer()
    dv.fit_transform(dfX_train)


# Model building Using DecisionTree
    dclf = DecisionTreeClassifier()
    my_xfeatures =dv.transform(dfX_train)
    dclf.fit(my_xfeatures, dfy_train)


    ## Accuracy of Models Decision Tree Classifier Works better than Naive Bayes
    # Accuracy on training set
    print(dclf.score(dv.transform(dfX_train), dfy_train))

    # Accuracy on test set
    print(dclf.score(dv.transform(dfX_test), dfy_test))

    decisiontreModel = open("./models/decisiontreemodel.pkl", "wb")
    dicModel = open("./models/dicmodel.pkl", "wb")
    joblib.dump(dclf, decisiontreModel)
    joblib.dump(dv, dicModel)
    decisiontreModel.close()
    dicModel.close()
    return dv, dclf

