import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col, udf, when}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}

object Model extends App {
  val spark = SparkSession
    .builder()
    .appName("Analysis of the titanic passenger dataset")
    .master("local[*]")
    .getOrCreate()

  import spark.implicits._

  val testingDf = spark.read.option("header", "true").csv("assignment-spark-2/src/main/resources/test.csv")
  val trainingDf = spark.read.option("header", "true").csv("assignment-spark-2/src/main/resources/train.csv")
  trainingDf.show(10) // Display the first 10 rows
  trainingDf.printSchema() // Print the schema of the dataset
  trainingDf.describe().show() // Compute basic statistics for each column

  // Count of each categorical value in the Pclass column
  trainingDf.groupBy("Pclass").count().show()

  // Count of each categorical value in the Embarked column
  trainingDf.groupBy("Embarked").count().show()

  // Mean age of passengers by sex and class
  trainingDf.groupBy("Sex", "Pclass")
    .agg(avg("Age").alias("mean_age"))
    .show()


  /** Feature Engineering */
  // Extraction of the titles from the Name column and create a new column called Title
  val titleRegex = """([\w]+)\. """.r
  val getTitle = udf((name: String) => titleRegex.findFirstMatchIn(name).map(_.group(1)).getOrElse(""))
  val trainingDFWithTitle = trainingDf.withColumn("Title", getTitle($"Name"))
  val testDFWithTitle = trainingDf.withColumn("Title", getTitle($"Name"))

  // making a new column for size of family
  val trainingDFWithFamilySize = trainingDFWithTitle.withColumn("FamilySize", $"SibSp" + $"Parch" + 1)
  val testDFWithFamilySize = testDFWithTitle.withColumn("FamilySize", $"SibSp" + $"Parch" + 1)

  // making a new column to check if passenger is alone
  val trainDFWithIsAlone = trainingDFWithFamilySize.withColumn("IsAlone", when($"FamilySize" === 1, 1).otherwise(0))
  val testDFWithIsAlone = testDFWithFamilySize.withColumn("IsAlone", when($"FamilySize" === 1, 1).otherwise(0))

  trainDFWithIsAlone.show()
  testDFWithIsAlone.show()

  /** Prediction */
  // Select the required columns for training and testing
  val selectedTrain = trainingDf.select("Survived", "Pclass", "Sex", "Age", "Fare","Cabin", "Embarked")
  val selectedTest = testingDf.select("PassengerId", "Pclass", "Sex", "Age", "Fare","Cabin", "Embarked")

  // data cleaning
  val cleaningTrain = selectedTrain.na.drop()
  val cleanTest = selectedTest.na.drop()

  // Cast the data to required types
  val finalTrain = cleaningTrain
    .withColumn("Age", col("Age").cast("Double"))
    .withColumn("Fare", col("Fare").cast("Double"))
    .withColumn("Pclass", col("Pclass").cast("Integer"))
    .withColumn("Survived", col("Survived").cast("Integer"))
    .withColumn("Cabin", col("Cabin").cast("Integer"))
  val finalTest = cleanTest
    .withColumn("Age", col("Age").cast("Double"))
    .withColumn("Fare", col("Fare").cast("Double"))
    .withColumn("Pclass", col("Pclass").cast("Integer"))
    .withColumn("PassengerId", col("PassengerId").cast("Integer"))
    .withColumn("Cabin", col("Cabin").cast("Integer"))
  //taking the mean of all values and filling incomplete data
  val ageMeanTrain = finalTrain.agg(avg("Age")).first()(0).asInstanceOf[Double]
  val fareMeanTrain = finalTrain.agg(avg("Fare")).first()(0).asInstanceOf[Double]
  val trainFilled = finalTrain.na.fill(ageMeanTrain, Seq("Age")).na.fill(fareMeanTrain, Seq("Fare"))

  val ageMeanTest = finalTest.agg(avg("Age")).first()(0).asInstanceOf[Double]
  val fareMeanTest = finalTest.agg(avg("Fare")).first()(0).asInstanceOf[Double]
  val testFilled = finalTest.na.fill(ageMeanTest, Seq("Age")).na.fill(fareMeanTest, Seq("Fare"))

 
  val sexIndexer = new StringIndexer()
    .setInputCol("Sex")
    .setOutputCol("SexIndex")

  val sexEncoder = new OneHotEncoder()
    .setInputCol("SexIndex")
    .setOutputCol("SexVec")

  val embarkedIndexer = new StringIndexer()
    .setInputCol("Embarked")
    .setOutputCol("EmbarkedIndex")

  // OneHotEncoder
  val embarkedEncoder = new OneHotEncoder()
    .setInputCol("EmbarkedIndex")
    .setOutputCol("EmbarkedVec")

  val assembler = new VectorAssembler()
    .setInputCols(Array("Pclass", "SexVec", "Age", "Fare", "EmbarkedVec"))
    .setOutputCol("features")

  val logisticRegModel = new LogisticRegression()
    .setFeaturesCol("features")
    .setLabelCol("Survived")

  val pipeline = new Pipeline()
    .setStages(Array(sexIndexer, embarkedIndexer, sexEncoder, embarkedEncoder,
      assembler, logisticRegModel))

  val modelFit: PipelineModel = pipeline.fit(trainFilled)
  val results = modelFit.transform(testFilled)
  val predictionResults = results.select("PassengerId", "prediction")
  predictionResults.show()
  predictionResults.write
    .format("csv")
    .option("header", "true")
    .option("delimiter", ",")
    .mode("overwrite")
    .save("/Users/shami/Desktop/assignment-spark-2/src/main/scala/survivor.csv")
}
