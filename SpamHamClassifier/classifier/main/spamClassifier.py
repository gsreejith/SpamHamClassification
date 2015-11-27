from pandas import DataFrame
from numpy import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import f1_score

#Open the file and read the messages and labels into a DataFrame.
def readDatasetIntoDataFrame():

    #Open file
    f = open("SpamHamDataset.txt", "r");

    #New DataFrame with two columns
    df = DataFrame(columns=('label', 'text'))

    count = 0
    for line in f:
        tokens = line.split()
        flag = tokens[0] #The first word of each row is the label.
        text = ""

        #Concatenate all tokens, except the label, to get the content of the message itself.
        for x in range(1, tokens.__len__()):
            text = text + tokens[x]
            text = text + " "
            sig = 0
            if flag == 'spam':
                sig = 1
        #print label, "---", text
        df.loc[count] = [sig, text]
        count = count + 1

    #Housekeeping
    df = df.reindex(random.permutation(df.index))

    return df

def trainAndEvaluateModels(dataFrame):

    # We'll test-drive two vectorizers. HashingVectorizer is famed to be memory-efficient!
    vectorizers = {'CountVectorizer',
                   'HashingVectorizer',
                   'TfidfVectorizer'}

    # We'll also try out some classifiers. Should be fun!
    classifiers = {'MultinomialNB',
                   'BernoulliNB',
                   'SGDClassifier',
                   'PassiveAggressiveClassifier'}

    for vectorizer in vectorizers: #For each combination of vectorizer and classifier
        for classifier in classifiers:
            vect = None
            if (vectorizer == 'CountVectorizer'):
                vect = CountVectorizer()
            elif (vectorizer == 'HashingVectorizer'):
                vect = HashingVectorizer(stop_words='english', non_negative=True, norm=None, binary=False)
            elif (vectorizer == 'TfidfVectorizer'):
                vect = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)

            clf = None
            if (classifier == 'MultinomialNB'):
                clf = MultinomialNB()
            elif (classifier == 'BernoulliNB'):
                clf = BernoulliNB()
            elif (classifier == 'SGDClassifier'):
                clf = SGDClassifier()
            elif (classifier == 'PassiveAggressiveClassifier'):
                clf = PassiveAggressiveClassifier()

            #Some spirits don't mix!
            if (vectorizer == 'HashingVectorizer' and classifier == 'BernoulliNB'):
                continue

            if (vectorizer != 'TfIdfVectorizer'):
                #Setup a pipeline to vectorize and classify data.
                pipeline = Pipeline([('vectorizer', vect), ('classifier', clf)])

                #We will divide our dataset into 10 pieces, train on 9 of them and test on the remaining one.
                #We will do this until each piece has been the test piece atleast once.
                k_fold = KFold(n=len(dataFrame), n_folds=10)
                totalScore = 0

                for train_indices, test_indices in k_fold:
                    train_text = dataFrame.iloc[train_indices]['text'].values
                    train_y = dataFrame.iloc[train_indices]['label'].values

                    test_text = dataFrame.iloc[test_indices]['text'].values
                    test_y = dataFrame.iloc[test_indices]['label'].values

                    #Train the model on the training set
                    pipeline.fit(train_text, train_y)

                    #Test the model on the test set
                    predictions = pipeline.predict(test_text)

                    #print predictions
                    score = f1_score(test_y, predictions)
                    totalScore = totalScore + score


                print 'Vectorizer: ', vectorizer, ' Classifier: ', classifier, ' Average prediction score: ', (totalScore/10)

            #Some spirits don't mix!!
            if classifier == 'MultinomialNB':
                continue

            #Setup a pipeline to work with a transformer too.
            pipeline1 = Pipeline([('vector', vect),
                                 ('transform', TfidfTransformer()),
                                 ('classifier', clf)])

            #We will divide our dataset into 10 pieces, train on 9 of them and test on the remaining one.
            #We will do this until each piece has been the test piece atleast once.
            k_fold = KFold(n=len(dataFrame), n_folds=10)
            totalScore = 0

            for train_indices, test_indices in k_fold:
                train_content = dataFrame.iloc[train_indices]['text'].values
                train_labels = dataFrame.iloc[train_indices]['label'].values

                test_content = dataFrame.iloc[test_indices]['text'].values
                test_labels = dataFrame.iloc[test_indices]['label'].values

                #Train the model on the training set
                pipeline1.fit(train_content, train_labels)

                #Test the model on the test set
                predictions = pipeline1.predict(test_content)
                score = f1_score(test_labels, predictions)
                totalScore = totalScore + score

            print 'Vectorizer: ', vectorizer, ' TfIdfTransformer, Classifier: ', classifier, ' Average prediction score: ', (totalScore/10)


if __name__ == "__main__":
    print "Let's classify these short messages!!"

    # Read messages from the dataset into a DataFrame.
    df = readDatasetIntoDataFrame()
    print "Number of messages to classify: ", df.__len__() #Good to know!

    #Visually verified dataset contents; no need to clean. Wish every dataset were like this.

    #Train models with a variety of vectorizers and classifiers
    trainAndEvaluateModels(df)