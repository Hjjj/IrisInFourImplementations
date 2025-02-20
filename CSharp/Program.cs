using System;
using System.IO;
using System.Linq;
using System.Net.Http;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace IrisClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create a new ML context for ML.NET operations
            var mlContext = new MLContext();

            // Load the data from URL
            string dataUrl = "https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv";
            IDataView dataView = LoadDataFromUrl(mlContext, dataUrl);

            // Split the data into training and test sets
            var trainTestData = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            var trainData = trainTestData.TrainSet;
            var testData = trainTestData.TestSet;

            // Define the data preparation and model training pipeline
            var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label", "Species")
                .Append(mlContext.Transforms.Concatenate("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy("Label", "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            // Train the model
            var model = pipeline.Fit(trainData);

            // Evaluate the model
            var predictions = model.Transform(testData);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            // Output the evaluation metrics
            Console.WriteLine($"Log-loss: {metrics.LogLoss}");
            Console.WriteLine($"Per class log-loss: {string.Join(" ", metrics.PerClassLogLoss.Select(l => l.ToString()))}");
        }

        private static IDataView LoadDataFromUrl(MLContext mlContext, string url)
        {
            using (var client = new HttpClient())
            {
                var data = client.GetStringAsync(url).Result;
                var dataPath = "iris_data.csv";
                File.WriteAllText(dataPath, data);
                return mlContext.Data.LoadFromTextFile<IrisData>(dataPath, hasHeader: true, separatorChar: ',');
            }
        }
    }

    // Define the IrisData class to represent the input data
    public class IrisData
    {
        [LoadColumn(0)]
        public float SepalLength;

        [LoadColumn(1)]
        public float SepalWidth;

        [LoadColumn(2)]
        public float PetalLength;

        [LoadColumn(3)]
        public float PetalWidth;

        [LoadColumn(4)]
        public string Species;
    }

    // Define the IrisPrediction class to represent the prediction output
    public class IrisPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedSpecies;
    }
}
