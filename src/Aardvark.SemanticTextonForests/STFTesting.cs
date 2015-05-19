using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Aardvark.Base;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Aardvark.SemanticTextonForests
{
    /// <summary>
    /// Result of one test case
    /// </summary>
    public class TestCaseResult
    {
        public string Name;
        public ClassifierTestResult TrainingSetResult;
        public ClassifierTestResult TestSetResult;
    }

    /// <summary>
    /// Represents one test case of the Forest classification system. Given a parameter object, the test case can generate a forest 
    /// from an image set, textonize the images and generate a trained classifier. After the test is run, the result of the classifier 
    /// test is stored.
    /// </summary>
    public class TestCase
    {
        public string Name;

        private LabeledImage[] images;
        public TrainingParams parameters;
        public TestingParams testParameters;
        private FilePaths filePaths;

        private LabeledImage[] trainingSet;
        private LabeledImage[] testSet;
        private Forest forest;

        private TextonizedLabeledImage[] textonTrainingSet;
        private TextonizedLabeledImage[] textonTestSet;
        private Classifier svm;


        public TestCase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, LabeledImage[] inputImages, string name)
        {
            this.parameters = parameters;
            this.images = inputImages;
            this.testParameters = testParameters;
            this.Name = name;
            this.filePaths = FilePaths;

            Report.BeginTimed(2, "Preparing test case: " + name);

            //select subset of classes
            if (testParameters.SubClass)
            {
                var ras = new List<LabeledImage>();

                for (int i = 0; i <= testParameters.ClassLimit; i++)
                {
                    ras.AddRange(images.Where(x => x.Label.Index == i));
                }

                images = new List<LabeledImage>(ras).ToArray();
            }

            //split images for training and testing (currently 50/50)
            images.SplitIntoSets(out trainingSet, out testSet);

            //take the entire training set to build the vocabulary
            if (testParameters.TrainForestWithEntireSet)
            {
                trainingSet = images;
            }

            //set parameter objects - old code
            parameters.FeatureProviderFactory.SelectProvider(parameters.FeatureType, parameters.SamplingWindow);
            parameters.SamplingProviderFactory.SelectProvider(parameters.SamplingType, parameters.SamplingWindow, parameters.RandomSamplingCount);

            Report.End(2);
        }

        /// <summary>
        /// Runs the test case.
        /// </summary>
        /// <param name="writeTempFilesToDisk">Whether or not to write temporary files to disk. If yes,
        /// the system can attempt to read a previously generated forest from disk.</param>
        /// <returns>Result of the test run.</returns>
        public TestCaseResult Run(bool writeTempFilesToDisk)
        {
            var Result = new TestCaseResult();

            Report.BeginTimed(1, "Test case " + Name + ": Execution.");

            Report.Line(1, "Test case " + Name + ": Generating Forest.");
            if (writeTempFilesToDisk)
            {
                if (testParameters.GenerateNewForest)
                {
                    HelperFunctions.CreateNewForestAndSaveToFile(filePaths.ForestFilePath, trainingSet, parameters);
                }
                forest = HelperFunctions.ReadForestFromFile(filePaths.ForestFilePath);
            }
            else
            {
                forest = HelperFunctions.CreateNewForest(trainingSet, parameters);
            }

            Report.Line(1, "Test case " + Name + ": Textonizing Images.");

            //fresh split (50/50)
            images.SplitIntoSets(out trainingSet, out testSet);

            if (writeTempFilesToDisk)
            {
                if (testParameters.GenerateNewTextonization)
                {
                    HelperFunctions.CreateTextonizationAndSaveToFile(filePaths.TrainingTextonsFilePath, forest, trainingSet, parameters);
                    HelperFunctions.CreateTextonizationAndSaveToFile(filePaths.TestTextonsFilePath, forest, testSet, parameters);
                }
                textonTrainingSet = HelperFunctions.ReadTextonizedImagesFromFile(filePaths.TrainingTextonsFilePath);
                textonTestSet = HelperFunctions.ReadTextonizedImagesFromFile(filePaths.TestTextonsFilePath);
            }
            else
            {
                textonTrainingSet = HelperFunctions.CreateTextonization(forest, trainingSet, parameters);
                textonTestSet = HelperFunctions.CreateTextonization(forest, testSet, parameters);
            }

            Report.Line(1, "Test case " + Name + ": Training SVM.");                                                        

            svm = new Classifier(filePaths.WorkDir);
            svm.Train(textonTrainingSet, parameters);

            Result.TrainingSetResult = svm.TestRecall(parameters, "Test case " + this.Name + ": training set");
            Result.TestSetResult = svm.Test(textonTestSet, parameters, "Test case " + this.Name + ": test set");
            Result.Name = "Result of test case " + Name;

            Report.End(1);

            return Result;
        }
    }

    /// <summary>
    /// Result of a test series.
    /// </summary>
    public class TestSeriesResult
    {
        /// <summary>
        /// collected output string
        /// </summary>
        public string OutputString;

        /// <summary>
        /// individual testing results (including all multi-runs)
        /// </summary>
        public TestCaseResult[] TestCaseResults;

        /// <summary>
        /// parameter sets used for the individual test cases (not including multi-run duplicates)
        /// </summary>
        public TrainingParams[] TestCaseTrainingparams;

        /// <summary>
        /// Index of best test Parameter object (according to a scoring function)
        /// </summary>
        public int IndexOfBestParams;

        /// <summary>
        /// Write the result into a JSON string
        /// </summary>
        /// <returns>JSON serialization</returns>
        public string writeJSON()
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full

            };

            var s = JsonConvert.SerializeObject(this, Formatting.Indented, settings);

            return s;
        }
    }

    /// <summary>
    /// A collection of test cases and statistics.
    /// </summary>
    public class TestSeries
    {
        private Dictionary<int, TestCase> TestCases = new Dictionary<int, TestCase>();
        private Dictionary<int, int> TestRunCounts = new Dictionary<int, int>();
        private int Length = 0;
        private FilePaths GlobalFilepaths;
        private LabeledImage[] GlobalImageSet;
        private string Name;
        private string historyFolderPath;
        private bool readwriteTempFiles;
        private Label[] Labels;

        /// <summary>
        /// Creates a new test series.
        /// </summary>
        /// <param name="name">Name of the test series</param>
        /// <param name="globalFilepaths">File path object to be used for all tests</param>
        /// <param name="globalImageSet">Data set to be used for all tests</param>
        /// <param name="historyFolderPath">Folder to store the testing history. If this is null, no history is stored</param>
        /// <param name="readwriteTempFiles">Whether or not to write temporary files to disk</param>
        public TestSeries(string name, FilePaths globalFilepaths, LabeledImage[] globalImageSet, Label[] allLabels, string historyFolderPath, bool readwriteTempFiles = false)
        {
            this.Name = name;
            this.GlobalFilepaths = globalFilepaths;
            this.GlobalImageSet = globalImageSet;
            this.historyFolderPath = historyFolderPath;
            this.readwriteTempFiles = readwriteTempFiles;
            this.Labels = allLabels;
        }

        /// <summary>
        /// Completely specify a new test case.
        /// </summary>
        /// <param name="parameters">Parameters for the classification system</param>
        /// <param name="testParameters">Parameters for the test series and statistics</param>
        /// <param name="FilePaths">All file paths for input and output</param>
        /// <param name="inputImages">Data set used for classification testing</param>
        /// <param name="runCount">The times each test should be run</param>
        /// <param name="name">Name for identifying the test series</param>
        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, LabeledImage[] inputImages, int runCount, string name)
        {
            var test = new TestCase(parameters, testParameters, FilePaths, inputImages, name);
            TestCases.Add(Length, test);
            TestRunCounts.Add(Length, runCount);
            Length++;
        }

        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, LabeledImage[] inputImages, string name)
        {
            AddTestcase(parameters, testParameters, FilePaths, inputImages, 1, name);
        }
        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, LabeledImage[] inputImages, string name)
        {
            AddTestcase(parameters, testParameters, GlobalFilepaths, inputImages, name);
        }
        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, string name)
        {
            AddTestcase(parameters, testParameters, FilePaths, GlobalImageSet, name);
        }
        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, string name)
        {
            AddTestcase(parameters, testParameters, GlobalFilepaths, GlobalImageSet, name);
        }
        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, LabeledImage[] inputImages, int runCount, string name)
        {
            AddTestcase(parameters, testParameters, GlobalFilepaths, inputImages, runCount, name);
        }
        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, int runCount, string name)
        {
            AddTestcase(parameters, testParameters, FilePaths, GlobalImageSet, runCount, name);
        }
        public void AddTestcase(TrainingParams parameters, TestingParams testParameters, int runCount, string name)
        {
            AddTestcase(parameters, testParameters, GlobalFilepaths, GlobalImageSet, runCount, name);
        }

        /// <summary>
        /// Specifies only the most important parameters for a test run. 
        /// </summary>
        /// <param name="name">Friendly Name of the test run.</param>
        /// <param name="treesCount">Number of Trees.</param>
        /// <param name="treeDepth">Maximum depth of a Tree.</param>
        /// <param name="imageSubsetCount">Size of the subset used for training of each Tree.</param>
        /// <param name="samplingWindow">Size of the (square) sampling window of an image in pixels.</param>
        /// <param name="classesCount">Only use the first [0-n] classes.</param>
        /// <param name="enableGridsearch">Enable searching for the best internal classifier parameter using cross validation.</param>
        /// <param name="runCount">How many times should this test be executed?</param>
        /// <param name="maxSamples">The total count of data point samples for each Tree's training. </param>
        public void AddSimpleTestcase(string name, int treesCount, int treeDepth, int imageSubsetCount, int samplingWindow, int classesCount, bool enableGridsearch, int runCount, int maxSamples = 999999999)
        {
            TestingParams testParameters = new TestingParams()
            {
                SubClass = (classesCount == -1)?false:true,
                ClassLimit = classesCount - 1,
                TrainForestWithEntireSet = true,
                GenerateNewForest = true,
                GenerateNewTextonization = true,
                GenerateNewSVMKernel = true
            };

            var parameters = new TrainingParams(treesCount, treeDepth, imageSubsetCount, samplingWindow, Labels, maxSamples)
            {
                ForestName = "STForest of testcase " + name,
                FeatureType = FeatureType.SelectRandom,
                SamplingType = SamplingType.RegularGrid,
                FeatureProviderFactory = new FeatureProviderFactory(FeatureType.SelectRandom, samplingWindow),
                SamplingProviderFactory = new SamplingProviderFactory(),
                RandomSamplingCount = 50,
                ThresholdCandidateNumber = 15,
                ThresholdInformationGainMinimum = 0.001d,
                ClassificationMode = ClassificationMode.Semantic,
                ForcePassthrough = false,
                EnableGridSearch = enableGridsearch
            };

            AddTestcase(parameters, testParameters, runCount, name);
        }

        /// <summary>
        /// Specifies only the most important parameters for a test run. 
        /// </summary>
        /// <param name="name">Friendly Name of the test run.</param>
        /// <param name="treesCount">Number of Trees.</param>
        /// <param name="treeDepth">Maximum depth of a Tree.</param>
        /// <param name="imageSubsetCount">Size of the subset used for training of each Tree.</param>
        /// <param name="samplingWindow">Size of the (square) sampling window of an image in pixels.</param>
        /// <param name="classesCount">Only use the first [0-n] classes.</param>
        /// <param name="runCount">How many times should this test be executed?</param>
        /// <param name="maxSamples">The total count of data point samples for each Tree's training. </param>
        public void AddSimpleTestcase(string name, int treesCount, int treeDepth, int imageSubsetCount, int samplingWindow, int runCount = 1, int maxSamples = 999999999, int classesCount = -1 )
        {
            AddSimpleTestcase(name, treesCount, treeDepth, imageSubsetCount, samplingWindow, classesCount, false, runCount, maxSamples);
        }

        /// <summary>
        /// Runs an individual Test Case.
        /// </summary>
        /// <param name="index">Index of the test to run.</param>
        /// <returns>Test Result.</returns>
        public TestCaseResult RunTestcase(int index)
        {
            return TestCases[index].Run(readwriteTempFiles);
        }

        /// <summary>
        /// Runs all the previously specified Test Cases in sequence.
        /// </summary>
        /// <returns>Result of the Test Series.</returns>
        public TestSeriesResult RunAllTestcases()
        {
            var result = new TestSeriesResult();
            var resultTests = new List<TestCaseResult>();
            var resultParams = new List<TrainingParams>();

            Report.BeginTimed(0, "Running test series " + Name);

            int outputspaces = 24;
            int mediumspaces = 12;
            int shortspaces = 8;
            var resultString = new StringBuilder();
            resultString.Append("Result of test series " + Name);
            resultString.Append(Environment.NewLine);
            resultString.Append(Environment.NewLine);

            //build matrix columns
            resultString.Append(HelperFunctions.Spaces("Nr.", shortspaces));
            resultString.Append(HelperFunctions.Spaces("test case", outputspaces));
            resultString.Append(HelperFunctions.Spaces("classes", shortspaces));
            resultString.Append(HelperFunctions.Spaces("trees", shortspaces));
            resultString.Append(HelperFunctions.Spaces("depth", shortspaces));
            resultString.Append(HelperFunctions.Spaces("subset", shortspaces));
            resultString.Append(HelperFunctions.Spaces("window", shortspaces));
            resultString.Append(HelperFunctions.Spaces("#feats", shortspaces));
            resultString.Append(HelperFunctions.Spaces("avg. precision", outputspaces));
            resultString.Append(HelperFunctions.Spaces("variance", mediumspaces));
            resultString.Append(HelperFunctions.Spaces("avg. recall", mediumspaces));
            resultString.Append(HelperFunctions.Spaces("runcount", shortspaces));
            resultString.Append(Environment.NewLine);

            TestCaseResult bestCase = null;
            int bestIndex = -1;
            double bestScore = 0;

            int testcount = 0;

            int totaltests = TestRunCounts.Sum(x => x.Value);   //one entry per test cotaining the number of times this test is run

            for (int i = 0; i < Length; i++ )
            {
                var test = TestCases[i];

                int runcount = TestRunCounts[i];
               

                double minprec = 1.0;
                double maxprec = 0.0;
                double meanprecision = 0.0;
                double meanrecall = 0.0;
                double resultvariance = 0.0;

                List<TestCaseResult> currentMultirunResults = new List<TestCaseResult>();

                //run multiple times
                for (int j = 0; j < runcount; j++)
                {
                    Report.Progress(0, (double)testcount / (double)totaltests);
                    testcount++;

                    var testResult = test.Run(readwriteTempFiles);
                    resultTests.Add(testResult);

                    currentMultirunResults.Add(testResult);

                }
                
                //get some statistics
                foreach(var tr in currentMultirunResults)
                {
                    var prec = tr.TestSetResult.Precision;
                    if (prec < minprec)
                    {
                        minprec = prec;
                    }
                    if (prec > maxprec)
                    {
                        maxprec = prec;
                    }

                    meanprecision += prec;
                    meanrecall += tr.TrainingSetResult.Precision;
                }
                meanprecision = meanprecision / (double)runcount;
                meanrecall = meanrecall / (double)runcount;
                foreach (var tr in currentMultirunResults)
                {
                    resultvariance += Math.Pow((tr.TestSetResult.Precision - meanprecision), 2.0);
                }
                resultvariance = Math.Sqrt(resultvariance / (double)runcount);

                resultParams.Add(test.parameters);

                //high precision is good, high variance is bad, high recall is sometimes good
                //TODO improve this scoring formula!!
                double testScoreValue = (meanprecision - resultvariance) + meanrecall;       

                if (testScoreValue > bestScore)
                {
                    bestScore = testScoreValue;
                    bestCase = currentMultirunResults.First();
                    bestIndex = i;
                }

                string precstring = (runcount > 1) ?
                    String.Format(CultureInfo.InvariantCulture, "{0:0.0000}  [{1:0.00}-{2:0.00}]", meanprecision, minprec, maxprec) :
                    String.Format(CultureInfo.InvariantCulture, "{0:0.0000} ", meanprecision);

                string varstring = (runcount > 1) ?
                    String.Format(CultureInfo.InvariantCulture, "{0:0.0000}", resultvariance) :
                    String.Format(CultureInfo.InvariantCulture, " ");

                string numfeaturesstring = (test.parameters.MaxSampleCount >= 999999999) ?
                    String.Format(CultureInfo.InvariantCulture, "all") :
                    String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.MaxSampleCount);

                string classesstring = (test.testParameters.ClassLimit + 1 == -1) ?
                    String.Format(CultureInfo.InvariantCulture, "all") :
                    String.Format(CultureInfo.InvariantCulture, "{0}", test.testParameters.ClassLimit + 1);

                //insert matrix values, one row per testcase
                resultString.Append(HelperFunctions.Spaces(String.Format(CultureInfo.InvariantCulture, "{0}", i), shortspaces));
                resultString.Append(HelperFunctions.Spaces(test.Name, outputspaces));
                resultString.Append(HelperFunctions.Spaces(classesstring, shortspaces));
                resultString.Append(HelperFunctions.Spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.TreesCount), shortspaces));
                resultString.Append(HelperFunctions.Spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.MaxTreeDepth), shortspaces));
                resultString.Append(HelperFunctions.Spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.ImageSubsetCount), shortspaces));
                resultString.Append(HelperFunctions.Spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.SamplingWindow), shortspaces));
                resultString.Append(HelperFunctions.Spaces(numfeaturesstring, shortspaces));
                resultString.Append(HelperFunctions.Spaces(precstring, outputspaces));
                resultString.Append(HelperFunctions.Spaces(varstring, mediumspaces));
                resultString.Append(HelperFunctions.Spaces(String.Format(CultureInfo.InvariantCulture, "{0:0.0000}", meanrecall), mediumspaces));
                resultString.Append(HelperFunctions.Spaces(String.Format(CultureInfo.InvariantCulture, "{0}", runcount), shortspaces));
                resultString.Append(Environment.NewLine);
            }

            Report.End(0);

            if(bestCase == null)    //exception catching
            {
                return null;
            }

            resultString.Append(Environment.NewLine);
            resultString.Append("Test series " + Name + " finished. Best test case is ("+bestIndex+") " + ((bestCase == null) ? "none" : bestCase.Name)+" with a score of "+String.Format(CultureInfo.InvariantCulture,"{0:0.000}", bestScore));
            resultString.Append(Environment.NewLine);
            resultString.Append(Environment.NewLine);
            resultString.Append(" ... with output (of the first run): ");
            resultString.Append(Environment.NewLine);
            resultString.Append(bestCase.TrainingSetResult.OutputString + bestCase.TestSetResult.OutputString);

            result.OutputString = resultString.ToString();
            result.TestCaseResults = resultTests.ToArray();
            result.TestCaseTrainingparams = resultParams.ToArray();
            result.IndexOfBestParams = bestIndex;

            WriteHistory(result);

            return result;
        }

        /// <summary>
        /// Stores a test series result in the history folder, in a subfolder with the current timestamp. 
        /// </summary>
        /// <param name="output">The test series result to be written to disk.</param>
        private void WriteHistory(TestSeriesResult output)
        {
            if(historyFolderPath == null)   //if path is null, do nothing
            {
                return;
            }

            //else, create a subfolder with the current datetime and write the current result string plus the collection of parameters and individual results
            string historyfolder = Path.Combine(historyFolderPath, DateTime.Now.ToString("yyyyMMdd_HHmmss"));
            Directory.CreateDirectory(historyfolder);
            string historyfilepath = Path.Combine(historyfolder, "output.txt");
            string historycasesfilepath = Path.Combine(historyfolder, "individualResults.json");

            File.WriteAllText(historyfilepath, output.OutputString);

            File.WriteAllText(historycasesfilepath, output.writeJSON());
        }

        /// <summary>
        /// loads a previously generated test series result from file
        /// </summary>
        /// <param name="path">File path of previous test result</param>
        /// <returns></returns>
        public static TestSeriesResult LoadFromJSON(string path)
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full
            };
            return JsonConvert.DeserializeObject<TestSeriesResult>(File.ReadAllText(path), settings);
        }
    }

    #region Parameter Class
    /// <summary>
    /// The test case parameter class, which stores the parameters important for one test case.
    /// </summary>
    public class TestingParams
    {
        /// <summary>
        /// Only use the first few classes?
        /// </summary>
        public bool SubClass;                     

        /// <summary>
        /// The number of classes to use if SubClass is set to true
        /// </summary>
        public int ClassLimit;

        /// <summary>
        /// train forest with entire image set? If false, train forest only with training set
        /// </summary>
        public bool TrainForestWithEntireSet;     

        /// <summary>
        /// Create a new forest? If false, attempt to read an existing one from disk.
        /// </summary>
        public bool GenerateNewForest;         

        /// <summary>
        /// Create a new textonization set? If false, attempt to read an existing one from disk.
        /// </summary>
        public bool GenerateNewTextonization;    

        /// <summary>
        /// Create new SVM kernels? If false, attempt to read an existing one from disk.
        /// </summary>
        public bool GenerateNewSVMKernel;        
    }
#endregion
}
