'''
Working with Python Jupyter Notebooks
Goal : To Explore Supervised Learning Methods -> in this case, Naive Bayes Classification
We predicate many SL methods off of the NB method

Links : 
1. https://www.opinosis-analytics.com/knowledge-base/term-frequency-explained/#:~:text=Term%20frequency%20(TF)%20means%20how,about%20how%20you%20define%20it.



Setting up Pythonic Naive Bayes Classification.
Let's explore how to evolve and iterate on a classificier from two class ( binomial ) to multi class.
We'll also explore the methods of TF ( Total Frequency ), IDF ( Inverse Document Frequency ), and TF-IDF ( which combines the two as a product ).
Lastly, we'll also explore vectorization of counts based on a vocabulary of words, and representing non-mathematical inputs more mathematically.
Links : 
1. https://www.opinosis-analytics.com/knowledge-base/term-frequency-explained/#:~:text=Term%20frequency%20(TF)%20means%20how,about%20how%20you%20define%20it.
Good canon case studies :
1. UberEats Market Campaigns -> targeting users : can we make non-Habitual Users Habitual Users, following a promitions campaign? Do we have to promote to a massive user base, or a cohort group only?
- maximize probability of conversions
- collect billags : [ byMonth, byYear, averages ]
- given a finite marketing budget ( $100,000 ), can we target just the specific cohort of potential conversions - and thus, maximize revenue?
- model metric : number of converted users grows postDeployment

custVec(x) = [ srvcUsage(pastWeek), srvcUsage(pastMonth), avgSpendAmount ] 
yearlyUsage not predictive enough

Customer promotions are Gaussian-model based ; not Spam ( Bernoulli ) or Articles ( Multinomial ).


Need to calculate likeliehoods for categorial and numerical information

Assignment One Goal : 
1. Leverage the Scikit-Learn library to set up NBC : Naive-Bayes Classifier.
    - but what should our NBC even do?
2. Set up feature hashing, stop words, and n-grams
3. Ensure the original input data is Gaussian -> to use Gaussian likeliehoods. Do not use data with Power Distributions.
4. Use Scikit KDE - Kernel Density Estimation - to estimate a dataset's probability distriubtion.

Larger Docs with NBC
- multinomial likeliehood

Eliminate vocabulary : set up feature hashing.

Assignment Two Goal :
1. Set up a classifier, using whichever desired approach, for document classification.





'''

'''
Replace with scikit later?
Vectorize features
Coded with just one hash function in mind -> can extend to a one-bit second hash func to reduce collisions.
'''
def featureHashing(features : List[string], n:int):
    featureVecLen = 100
    featureHashVec = [0 for i in range(featureVecLen)]
    for feature in features:
        featureHash := hash(f) % featureVecLen
        featureHashVec[featureHash] += 1
    return featureHashVec

'''
Compute means and stdDevs for each feature ( f1,...,fm) for each data point (d1,...,dn)
But why are these two summary stats crucial?
Note : for GTL assignation -> how do we even store this information?
Filter and compute summStats for specific labels or class Types.
'''
def computeMeansAndStdDeviationsOfFeatureVectors(self, dataPoints: List[int], targetClassLabel:int) -> List[List[int]]:
    featureSpaceSize = len(dataPoints[0]) - 1
    sums = [0 for i in range(featureSpaceSize)]
    averages = [0 for i in range(featureSpaceSize)]
    stdDevs = [0 for i in range(featureSpaceSize)]
    numDataPoints = len(dataPoints)
    for dataPoint in dataPoints:
        curClassLabel = dataPoint[0]
        if(curClassLabel == targetClassLabel):
            for index, featureVal in enumerate(dataPoint[1:]):
                sums[index] += featureVal
    for i in range(len(featureSpaceSize)):
        averages[i] = sums[i] / numDataPoints
        stdDevs[i] = self.computeStdDev(sums[i], averages[i], numDataPoints - 1)
    return [[averages[1:], stdDevs[1:]]]        

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
Can we add optimizations atop original NB algorithm?
'''
def optimizeOriginalNaiveBayes(self) -> None:


'''
Make topic words ( not stop words ) appear more often

'''
def stopWordRemoval(stopWords:, doc:string) -> List[String]:
    stopWordsRemovedDoc = []
    for token in doc:
        if token not in stopWords:
    return stopWordsRemovedDoc

def computeTFIDF(self, targetTerm: string, doc:string, docSet:[string]) -> int:
    tf = self.computeTermFrequency(targetTerm, doc)
    idf = self.computeInverseDocumentFrequency(targetTerm, docSet)
    return (tf * idf )

'''
IDF : generalizes across a large document corpus
If a word is more common -> we log scale that down
But can we have log_b(0) in the first place -> break log ( is UNDEFINED )
Need smoothening op for always positive log_b(...) input
'''
def computeInverseDocumentFrequency(self, targetTerm: string, docSet:[string]) -> int:
    termFreqAcrossDocs = 0
    numDocs = len(docSet)
    for doc in docSet:
        if(self.isInDocument(targetTerm,doc)):
            termFreqAcrossDocs += 1
    laplaceSmoothening = 0.1
    idf = log((numDocs / termFreqAcrossDocs) + laplaceSmoothening)
    return idf

def isInDocument(self, targetTerm: string, doc:string) -> bool:
    isInDoc = False
    delimeter = "\\s+"
    docTokens = doc.split(delimeter)
    for token in docTokens:
        if(token === targetTerm);
            isInDoc = True
            break
    return isInDoc

'''
TF : scoped down to a single document
'''
def computeTermFrequency(self, targetTerm:string, doc:string) -> int:
    delimeter = "\\s+"
    docTokens = doc.split(delimeter)
    totalNumberDocTerms = len(docTokens)
    targetTermFreq = 0
    for token in docTokens:
        if token == targetTerm:
            targetTermFreq += 1
    return (targetTermFreq / totalNumberDocTerms)
