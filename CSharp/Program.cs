using System;
using System.Configuration;
using System.IO;
using System.Linq;
using System.Net.Http;
using Microsoft.ML;
using Microsoft.ML.Data;
using Python.Runtime;

namespace IrisClassification
{
    class Program
    {
        static void Main(string[] args)
        {
            // Read the Python DLL path and data URL from the configuration file
            string environmentVar = ConfigurationManager.AppSettings["EnvironmentVar"];
            string pythonDllPath = ConfigurationManager.AppSettings["PythonDllPath"];
            string dataUrl = ConfigurationManager.AppSettings["DataUrl"];
            string plotFilePath = ConfigurationManager.AppSettings["PlotFilePath"];

            // Set the environment variable for Python DLL
            Environment.SetEnvironmentVariable(environmentVar, pythonDllPath);

            // Initialize the Python runtime
            PythonEngine.Initialize();

            // Test python.net first
            using (Py.GIL())
            {
                dynamic np = Py.Import("numpy");
                dynamic plt = Py.Import("matplotlib.pyplot");

                // Generate data
                dynamic x = np.linspace(0, 10, 100);
                dynamic y = np.sin(x);

                // Create plot
                plt.plot(x, y);
                plt.title("Sine Wave from C#");
                plt.xlabel("X-axis");
                plt.ylabel("Y-axis");

                // Save to file
                plt.savefig(plotFilePath);
                plt.close(); // Close to avoid display attempts
            }

            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = plotFilePath,
                UseShellExecute = true
            });

            // Create a new ML context for ML.NET operations
            var mlContext = new MLContext();

            // Load the data from URL
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
