'''
Py spark logistic regression

Hyperparams :
    10 iterations
    Threshold hyperparam
    L1 reg

Assume : each userId unique and counts = the last observed day
Binary labeling only

Churn problem : cancellation = func(user features )
Vectorize for PySpark pipelines please

25 mins and passed :-) ( return spark type here ) 

'''
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import col

from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, DoubleType


# train and predicted: same dataframe
def predict_cancellations(user_interaction_df):


    NUM_ITERATIONS = 10
    DECISION_THRESHOLD = 0.6
    L1_REGULARIZATION = 0.1
    REGULARIZATION_TYPE = 1.0 # L1

    # 1. Assemble features into a single vector column
    # The resulting DataFrame will have BOTH 'features' vector AND 'label.'
    # minimize merge work
    # group features together
    feature_cols = ["month_interaction_count", "week_interaction_count", "day_interaction_count"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    training_dataframe = assembler.transform(user_interaction_df)

    # 2. Initialize and train Logistic Regression with hyperparameters
    lr = LogisticRegression(
        featuresCol="features",
        labelCol="cancelled_within_week",
        maxIter=NUM_ITERATIONS,             # number of training iterations
        threshold=DECISION_THRESHOLD,          # decision threshold for prediction
        regParam=L1_REGULARIZATION,           # regularization strength
        elasticNetParam=REGULARIZATION_TYPE     # L1 regularization (1.0 = L1, 0.0 = L2)
    )

    # Train the model
    model = lr.fit(training_dataframe)

    # 3. Make predictions on the training dataframe
    # Vectors desired for rawPrediction, probability
    predictions = model.transform(training_dataframe)
    # Select the columns you want, including the vectors
    # Convert PySpark DataFrame to pandas DataFrame?
    spark_output_dataframe = predictions.select([
        "user_id",          # your identifier column
        "rawPrediction",   # vector of raw scores for each class
        "probability",     # vector of predicted probabilities for each class
        "prediction"       # predicted label
    ])
   
    # # If you want to see the first few rows
    spark_output_dataframe.show(truncate=False)
    return spark_output_dataframe


    # Start Spark session ( once per application ) 
    # spark = SparkSession.builder \
    #     .appName("MyApp") \
    #     .getOrCreate()

    # Free up Spark cluster resources
    # spark.stop()
    

     # # Convert vectors to arrays before toPandas()
    # vector_to_array = udf(lambda v: v.toArray().tolist(), ArrayType(DoubleType()))

    # # panda_output_dataframe = spark_output_dataframe.toPandas() # conversion
    # panda_output_dataframe = predictions.select(
    #     "user_id",
    #     vector_to_array("rawPrediction").alias("rawPrediction"),
    #     vector_to_array("probability").alias("probability"),
    #     "prediction"
    # ).toPandas()
    
