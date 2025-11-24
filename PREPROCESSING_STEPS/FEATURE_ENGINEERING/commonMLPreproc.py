# '''
# A compilation of commonly-used ML techniques to address upstream steps in MLE pipelines. Techniques encompass
# (A) Data Engineering
# (B) Feature Engineering
# (C) Cleansing steps / operations of a possible data engineering-esque pipeline.

# '''


# '''
# Here
# (A) Given a mock dataset with multiple rows of data ( records ) and each col being a feature space, let us do the following
# - delete bad records or
# - imputation of the records ( defaults or summary states )
# - modularize function to execute.

# Example CSV dataset URL : https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv
# Wooooh an all numeric dataset sans a header line.

# '''

# Damn gotta learn python standard libraries.
# https://machinelearningmastery.com/load-machine-learning-data-python/#:~:text=The%20Python%20API%20provides%20the,directory%20(download%20from%20here).
import csv
import numpy as np
import pandas as pd
from sklearn.base import defaultdict
import math
from collections import defaultdict
    
# Get this functionality working later :-)
# filename = 'pima-indians-diabetes.data.csv'
# rawData = loadMLRawDataStdPyLib(filename)

# data =[[3,1,2],[3,4,5],[6,7,8]]
# rawData = pd.DataFrame(data)
# zerothCol = rawData[:][0]
# firstCol = rawData[:][1]
# # print(rawData)
# # print(firstCol)

# defaultVal = 0
# mean = np.mean(zerothCol)
# median = np.median(zerothCol)
# # Array like values ( with input )
# from scipy import stats
# # No numpy way of caluclation here
# modeVec = stats.mode(zerothCol)
# mode = modeVec[0]
# modeCount = modeVec[1]
# # print("Mean,median,mode = ", mean, median, mode)
# print(f"Mean = {mean} \t median = {median} \t modeVal = {mode} \t modeCount = {modeCount}.")

# '''
# Feature Scaling techniques
# '''

# '''
# Normalize each column by {xmin,xmax}
# Dataset columns are features F1,...,Fn
# Normalize for only the columns given; not all columns are strictly numerical.
# '''
# def normalize(dataset,cols):
#     xmin = min()
#     xmax = max()
#     for col in cols:
#         print(col)
        
    
# '''
# z-score normalize
# Fix data distributions : change (mean,sigma) from original values 
# to (0,1). 
# z = ( x - mu ) / sigma
# '''
# def standardize(dataset,cols):
    

# # Datasets sourced from Kaggle { https://www.kaggle.com/ } 

# rawDataset = [[1,2,3],[4,3,2],[1,2,3]]
# normalizedDS = normalize(dataset)
# print(normalizedDS)

# '''
# Dataset CSV loading techniques :
# gaaah first thing to learn in MLE -> how the fuck to even load data

# Also ensure to restart the kernel on code executions. It's not just clean code that matters.
# Note : Jupyter is web browser -> must drag-and-drop.
# '''

# # Damn gotta learn python standard libraries.
# # https://machinelearningmastery.com/load-machine-learning-data-python/#:~:text=The%20Python%20API%20provides%20the,directory%20(download%20from%20here).
# import csv
# import numpy as np
# import pandas as pd

# # Output : raw data by < indices, columns > ; select kth Col to solve mean on a col
# # How to handle huge dataset file reading too?
# # Woah may run into data limitations with this ( that and multimodality )
# # Dataframe handles bigger dimensionality
# def loadMLRawDataStdPyLib(filename):
#     raw_data = open(filename, 'rt')
#     reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
#     # List on an iterator style -> then array-ify.
#     x = list(reader)
#     # Notice conv : integers -> Pythonic floats. Facilitates storage?
#     data = np.array(x).astype('float')
#     print(data.shape)

# def loadMLRawDataPandasDataframe(filename):
#     print(f"Starting read of filename {filename}")
#     df = pd.read_csv(filename)
#     # print(df.head(2))
#     print(f"Finished read of filename {filename}")
#     return df

# # filename="C:\Users\haris\OneDrive\Desktop\KAGGLE_PUBLIC_DATASETS\car_prices.csv\car_prices.csv"
# filename="car_prices.csv"
# rawData = loadMLRawDataPandasDataframe(filename)
# print(rawData)

# '''
# Feature Scaling techniques
# '''

# '''
# Normalize each column by {xmin,xmax}
# Dataset columns are features F1,...,Fn
# Normalize for only the columns given; not all columns are strictly numerical.
# '''
# def normalize(dataset,cols):
#     xmin = min()
#     xmax = max()
#     for col in cols:
#         print(col)
        
    
# '''
# z-score normalize
# Fix data distributions : change (mean,sigma) from original values 
# to (0,1). 
# z = ( x - mu ) / sigma
# '''
# def standardize(dataset,cols):
    

# # Datasets sourced from Kaggle { https://www.kaggle.com/ } 

# rawDataset = [[1,2,3],[4,3,2],[1,2,3]]
# normalizedDS = normalize(dataset)
# print(normalizedDS)

# '''
# Pandas dataframe selections
# Huh <object> over <string> preferred in pythonic type system.
# '''

# def manipulateBasicPandasDataFrames():
#     df = pd.DataFrame(
#      [
#       (73, 15, 55, 33, 'foo'),
#       (63, 64, 11, 11, 'bar'),
#       (56, 72, 57, 55, 'foo'),
#       (63, 64, 11, 11, 'bat'),
#      ],
#      # columns=['A', 'B', 'C', 'D', 'E'],
#     )
    
#     colNames = df.columns.tolist()
#     print(f"colNames of dataframe = {colNames}")
    
#     # df
#     # print(df[df['B'] == 64])
#     # https://stackoverflow.com/questions/64307431/get-second-column-of-a-data-frame-using-pandas
#     # df[df['C'] == 11]
#     # df['B'].values
#     # df['E'].values
#     df[2].values

# manipulateBasicPandasDataFrames()

# '''
# Dataset CSV loading techniques :
# gaaah first thing to learn in MLE -> how the fuck to even load data

# Also ensure to restart the kernel on code executions. It's not just clean code that matters.
# Note : Jupyter is web browser -> must drag-and-drop.
# '''

# # Damn gotta learn python standard libraries.
# # https://machinelearningmastery.com/load-machine-learning-data-python/#:~:text=The%20Python%20API%20provides%20the,directory%20(download%20from%20here).
# import csv
# import numpy as np
# import pandas as pd

# # Output : raw data by < indices, columns > ; select kth Col to solve mean on a col
# # How to handle huge dataset file reading too?
# # Woah may run into data limitations with this ( that and multimodality )
# # Dataframe handles bigger dimensionality
# def loadMLRawDataStdPyLib(filename):
#     raw_data = open(filename, 'rt')
#     reader = csv.reader(raw_data, delimiter=',', quoting=csv.QUOTE_NONE)
#     # List on an iterator style -> then array-ify.
#     x = list(reader)
#     # Notice conv : integers -> Pythonic floats. Facilitates storage?
#     data = np.array(x).astype('float')
#     print(data.shape)

# # How to cache dataframe reading here?
# def loadMLRawDataPandasDataframe(filename):
#     print(f"Starting read of filename {filename}")
#     df = pd.read_csv(filename)
#     # print(df.head(2))
#     print(f"Finished read of filename {filename}")
#     return df

# def viewRawDataPandasDataframe(df,colName):
#     dims = df.shape
#     numRows = len(df)
#     nR = dims[0]
#     if(numRows != nR):
#         print("Error in number rows reading")
#     numCols = dims[1]
#     print(f"Num rows = {numRows} \t num cols = {numCols}")
#     # if(colName >= numCols):
#         # Use most specific exception constructor semantically befitting
#         # raise ValueError('User column given exceeds the number of cols')
#     colView = df[colName].values # this is a numpy ndarray object ; not callable()
#     print("For column name = {colName}, col entries = ")
#     print(colView)

# # filename="C:\Users\haris\OneDrive\Desktop\KAGGLE_PUBLIC_DATASETS\car_prices.csv\car_prices.csv"
# filename="car_prices.csv"
# rawData = loadMLRawDataPandasDataframe(filename)
# colNames = rawData.columns.tolist()
# print(f"colNames of dataframe = {colNames}")
# viewRawDataPandasDataframe(rawData,"year")
# # print(rawData)

'''
Bag of Words feature engineering technique
Reference = https://en.wikipedia.org/wiki/Bag-of-words_model

Pros :
- Simple to implement
- Effective for text classification tasks
- Works well with traditional ML algorithms
- Fast to compute
Cons :
- Ignores word order and context
- High dimensionality for large vocabularies
- Sparsity of feature vectors
- No semantic understanding
'''
def computeBagOfWords(vocabulary, document):
    """Compute the Bag of Words representation for a given document.
    
    Args:
        vocabulary (list of str): The list of unique words in the vocabulary.
        document (str): The input document as a string ( assumes space-separated words ).
        
    Returns:
        np.ndarray: A 1D array representing the Bag of Words vector.
    """
    # Initialize the Bag of Words vector with zeros
    if(len(vocabulary) == 0):
        print("Error : vocabulary length is zero.")
        return np.array([])
    bow_vector = np.zeros(len(vocabulary), dtype=int)
    
    # Tokenize the document into words
    words = document.split()
    
    # Count occurrences of each word in the vocabulary
    for word in words:
        if word in vocabulary:
            index = vocabulary.index(word)
            bow_vector[index] += 1
            
    return bow_vector

def test_computeBagOfWords():
    print(f"Testing computeBagOfWords function")
    DEFAULT_VOCAB =["the", "cat", "sat", "on", "mat"]
    vocabulary = DEFAULT_VOCAB
    # absolutePath= "/Users/harisrid/Desktop/CODEBASES/MACHINE_LEARNING_ENGINEERING_SIDE_PROJECTS/PREPROCESSING_STEPS/FEATURE_ENGINEERING
    # vocabFileNames = ["vocabulary1.txt", "vocabulary2.txt"]
    # for vocabFile in vocabFileNames:
    #     actualVocabFilePath = absolutePath + vocabFile
    #     try:
    #         print(f"Loading vocabulary from file: {actualVocabFilePath}")
    #         with open(actualVocabFilePath, 'r') as f:
    #             vocabulary = [line.strip() for line in f.readlines()]
    #         print(f"Vocabulary loaded: {vocabulary}")
    #     except FileNotFoundError:
    #         print(f"actualVocabFilePath {actualVocabFilePath} not found. Using default vocabulary for testing.")
    document = "the cat sat on the mat the cat"
    bow_vector = computeBagOfWords(vocabulary, document)
    print("Vocabulary:", vocabulary)
    print("Document:", document)
    print("Bag of Words vector:", bow_vector)
    for index, word in enumerate(vocabulary):
        print(f"'{word}' \t {bow_vector[index]}")

'''
TF-IDF feature engineering technique
Reference = https://en.wikipedia.org/wiki/Tf%E2%80%93idf

Goals :
- Weigh terms by importance
- Reduce impact of common words

Pros :
- Weighs terms by importance
- Reduces impact of common words
- Improves text classification performance
Cons :
- More complex to compute than Bag of Words
- Still ignores word order and context
- Sensitive to document length
'''
def compute_tf_idf(vocabulary, documents, tf, idf):
    """Compute the TF-IDF representation for a set of documents.
    
    Args:
        vocabulary (list of str): The list of unique words in the vocabulary.
        documents (list of str): The input documents as a list of strings.
        
    Returns:
        np.ndarray: A 2D array representing the TF-IDF matrix.
    """
    import numpy as np
    
    # Initialize the TF-IDF matrix
    tf_idf_matrix = np.zeros((len(documents), len(vocabulary)))
    
    # Compute TF-IDF for each document
    for doc_index, document in enumerate(documents):
        tf = compute_tf(vocabulary, document) # can we cache this ahead of time? Maybe?
        for word_index, word in enumerate(vocabulary):
            tf_value = tf.get(word, 0)
            idf_value = idf.get(word, 0)
            tf_idf_matrix[doc_index][word_index] = tf_value * idf_value
            
    return tf_idf_matrix

'''
    Compute the Term Frequency (TF) for a given document.
'''
def compute_tf(vocabulary, document):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    
    # Step 1: Compute term frequency (TF)
    # TF = (Number of times term t appears in a document) / (Total number of terms in the document)
    tf = defaultdict(int)
    for word in document:
        numWords = len(document)
        if word in vocabulary:
            tf[word] += 1 
    for word, wordCount in tf.items():
        tf[word] = wordCount / numWords  # Normalize by total words in document
    return tf

def compute_idf(vocabulary, documents):
    """Compute the Inverse Document Frequency (IDF) for a set of documents.
    
    Args:
        documents (list of str): The input documents as a list of strings.
        vocabulary (list of str): The list of unique words in the vocabulary.
        
    Returns:
        dict: A dictionary mapping each word to its IDF value.
        IDF = log( N / (1 + df) ) where
        N = total number of documents
        df = number of documents containing the word

        Penalizes words that appear in many documents ( common words ).
        Incentives words that appear in fewer documents.
    """

    N = len(documents)
    df = defaultdict(int)
    vocabSet = set(vocabulary) # fast computation
    
    # Calculate document frequency for each word
    for doc in documents:
        words = set(doc.split())
        for word in words:
            if word in vocabulary:
                df[word] += 1
    
    # Calculate IDF for each word
    idf = {}
    for word, freq in df.items():
        idf[word] = math.log(N / (1 + freq))  # Adding 1 to avoid division by zero
    
    return idf

def compute_tf_idf_with_vectorizer(vocabulary, documents):
    """Compute the TF-IDF representation for a set of documents.
    
    Args:
        vocabulary (list of str): The list of unique words in the vocabulary.
        documents (list of str): The input documents as a list of strings.
        
    Returns:
        np.ndarray: A 2D array representing the TF-IDF matrix.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    
    # Fit and transform the documents to get the TF-IDF matrix
    tf_idf_matrix = vectorizer.fit_transform(documents)
    
    return tf_idf_matrix.toarray()
        
def main():
    print(f"Inside main of commonMLPreproc.py : executing common ML preprocessing steps")
    test_computeBagOfWords()

main()

