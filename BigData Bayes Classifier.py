import random
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, concat_ws, trim, lower
from sparknlp.annotator import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import Stemmer
from pyspark.ml import Pipeline
from pyspark.sql.functions import col


spark = SparkSession.builder \
  .appName("Spark review Bayes Classifier") \
  .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:5.1.0") \
  .config("spark.network.timeout", "600s") \
  .config("spark.executor.heartbeatInterval", "120s") \
  .getOrCreate()

# Increasing network timeout and heartbeat intervals from configuration helped local execution.
# Because of my computer's limited resources, there wasn't enough time for the executors to respond.


# Verify Spark NLP is available.
print(spark.sparkContext.getConf().get("spark.jars.packages"))


# Specify input data path.
input_path = sys.argv[1]
dataFrame = spark.read.json(input_path)


#Clean column text from white spaces and lowercase so capitalization doesn't make the same words be counted many times
dataFrame = dataFrame.withColumn("trimmed_text", trim(lower(col("text"))))


# Filter out reviews with a score of 3.
filtered_data = dataFrame.filter(col("rating") != 3)


# Create labels based on review scores.
# Reviews with scores 1-2 are labeled as 0 (negative), and those with scores 4-5 as 1 (positive).
# The `when` function applies conditional transformations to the instructed columns.
labeled_data = filtered_data.withColumn( "label", when(col("rating") <= 2, 0).otherwise(1)).cache()


# Remove duplicate reviews based on asin and trimmed text columns.
deduplicated_data = labeled_data.dropDuplicates(['asin', 'trimmed_text'])


# Remove stop words (common words like "and", "the", etc.).
# Stop words are removed because they don't provide meaningful information for sentiment analysis.
# The stop word remover works on string arrays and not just a string (e.g. the review strings on the reviewText column).
# Use tokenizer to split the sentences into string arrays.
reviewTokenizer = RegexTokenizer(inputCol="trimmed_text", outputCol="reviewTokens")
tokenized_data = reviewTokenizer.transform(deduplicated_data)


# Use the remover.
remover = StopWordsRemover().setInputCol("reviewTokens").setOutputCol("noStopWord_review")


cleaned_data = remover.transform(tokenized_data)


#The filtered_review column contains the review texts without the stop words, each one in a single string, and will serve as input to the document assembler.
final_data = cleaned_data.withColumn("filtered_review", concat_ws(" ", "noStopWord_review"))


#STEMMING AND LEMMATIZATION PROCESS
# Assemble the text into a "document" format for NLP.
# DocumentAssembler converts raw text into a format that Spark NLP can process.
document_assembler = DocumentAssembler().setInputCol("filtered_review").setOutputCol("document")


#Tokenize for the lemmatization and Stemming processes.
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")


# Lemmatize with the pretrained model.
# Lemmatization reduces words to their base form (lemma) e.g., "studies" -> "study".
lemmatizer = LemmatizerModel.pretrained().setInputCols(["token"]).setOutputCol("lemma")


# Stem the tokens.
# Stemming reduces words to their root form, e.g., "running" -> "run".
stemmer = Stemmer().setInputCols(["lemma"]).setOutputCol("stem")


pipeline = Pipeline(stages=[document_assembler, tokenizer, lemmatizer, stemmer])
dataModel = pipeline.fit(final_data)


result = dataModel.transform(final_data)


vectorData= result.withColumn('stemData', col('stem.result'))
# Transform the processed text into feature vectors with count vectorizer.
vectorizer = CountVectorizer(inputCol="stemData", outputCol="features")
vectorized_model = vectorizer.fit(vectorData)
vectorized_data = vectorized_model.transform(vectorData)


#NAIVE BAYES CLASSIFICATION
# Randomly split the training and testing sets to the ratio specified of the execution command.
split = [float(sys.argv[2]), float(sys.argv[3])] # train/test data split ratio.
train_data, test_data = vectorized_data.randomSplit(split, seed=random.randint(0,2**32-1))  # Different seed for each execution


# Create and train the Naive Bayes classifier.
nb = NaiveBayes(smoothing = 1.0, modelType = "multinomial", featuresCol = "features", labelCol = "label")
model = nb.fit(train_data)  # Train the model


# Test data predictions
predictions = model.transform(test_data)


# Precision and Recall evaluation
evaluator_precision = MulticlassClassificationEvaluator(
  labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
)
evaluator_recall = MulticlassClassificationEvaluator(
  labelCol="label", predictionCol="prediction", metricName="weightedRecall"
)
precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)


results = []
# Save run data for the execution
results.append({
  "split": split,
  "precision": precision,
  "recall": recall,
})

# Print precision and recall
print(results)

# Terminate SparkSession and free resources
spark.stop()