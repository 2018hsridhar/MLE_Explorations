'''
Working with Python Jupyter Notebooks
Goal : To Explore Supervised Learning Methods -> in this case, Naive Bayes Classification
We predicate many SL methods off of the NB method

Links : 
1. https://www.opinosis-analytics.com/knowledge-base/term-frequency-explained/#:~:text=Term%20frequency%20(TF)%20means%20how,about%20how%20you%20define%20it.



'''

'''
Normalization method : less systematic bias from longer documents.
'''

'''
Better if can make in-place instead
Extends by passing stopWords ( versus preDefList ) 

'''

def testNaturalLanguageProcessingMethods() -> bool:
    nlpMethodsWork = True
    stopWordsListOne = ['the','and','or','but', 'this', 'that']
    stopWordsListTwo = ['a','at','is','are','to','be']
    return nlpMethodsWork


'''
Make topic words ( not stop words ) appear more often

'''
def stopWordRemoval(stopWords:, doc:string) -> List[String]:
    stopWordsRemovedDoc = []
    for token in doc:
        if token not in stopWords:
    return stopWordsRemovedDoc

'''
'''
def computeInverseDocumentFrequency(targetTerm: string, docSet:[string]) -> int:
    idf = 0
    for doc in docSet:
    
    return idf

'''
TF : scoped down to a single document
'''
def computeTermFrequency(targetTerm:string, doc:string) -> int:
    delimeter = "\\s+"
    docTokens = doc.split(delimeter)
    totalNumberDocTerms = len(docTokens)
    targetTermFreq = 0
    for token in docTokens:
        if token == targetTerm:
            targetTermFreq += 1
    return (targetTermFreq / totalNumberDocTerms)

