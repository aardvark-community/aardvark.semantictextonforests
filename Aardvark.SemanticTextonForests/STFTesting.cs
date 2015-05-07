using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Aardvark.Base;
using Newtonsoft.Json;

namespace Aardvark.SemanticTextonForests
{
    //result of one test case
    public class TestCaseResult
    {
        public string Name;
        public SVMTestResult TrainingSetResult;
        public SVMTestResult TestSetResult;
    }

    //represents one test case of the STF program
    //given a parameter object, the test case can generate a forest from an image set, textonize the images and generate a trained SVM
    //after the test is run, the result of the SVM test is stored
    public class TestCase
    {
        public string Name;

        private STLabelledImage[] images;
        public TrainingParams parameters;
        public TestingParams testParameters;
        private FilePaths filePaths;

        private STLabelledImage[] trainingSet;
        private STLabelledImage[] testSet;
        private STForest forest;

        private STTextonizedLabelledImage[] textonTrainingSet;
        private STTextonizedLabelledImage[] textonTestSet;
        private STFSVM svm;


        public TestCase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, STLabelledImage[] inputImages, string name)
        {
            this.parameters = parameters;
            this.images = inputImages;
            this.testParameters = testParameters;
            this.Name = name;
            this.filePaths = FilePaths;

            Report.BeginTimed(2, "Preparing test case: " + name);

            //select subset of classes
            if (testParameters.subClass)
            {
                var ras = new List<STLabelledImage>();

                for (int i = 0; i <= testParameters.classLimit; i++)
                {
                    ras.AddRange(images.Where(x => x.ClassLabel.Index == i));
                }

                images = new List<STLabelledImage>(ras).ToArray();
            }

            //split images for training and testing (currently 50/50)
            images.splitIntoSets(out trainingSet, out testSet);

            //take the entire training set to build the vocabulary
            if (testParameters.trainForestWithEntireSet)
            {
                trainingSet = images;
            }

            //set parameter objects
            parameters.featureProviderFactory.selectProvider(parameters.featureType, parameters.samplingWindow);
            parameters.samplingProviderFactory.selectProvider(parameters.samplingType, parameters.samplingWindow, parameters.randomSamplingCount);

            Report.End(2);
        }

        public TestCaseResult run(bool writeTempFilesToDisk)
        {
            var Result = new TestCaseResult();

            Report.BeginTimed(1, "Test case " + Name + ": Execution.");

            Report.Line(1, "Test case " + Name + ": Generating Forest.");
            if (writeTempFilesToDisk)
            {
                if (testParameters.generateNewForest)
                {
                    HelperFunctions.createNewForestAndSaveToFile(filePaths.forestFilePath, trainingSet, parameters);
                }
                forest = HelperFunctions.readForestFromFile(filePaths.forestFilePath);
            }
            else
            {
                forest = HelperFunctions.createNewForest(trainingSet, parameters);
            }

            Report.Line(1, "Test case " + Name + ": Textonizing Images.");

            //fresh split (50/50)
            images.splitIntoSets(out trainingSet, out testSet);

            if (writeTempFilesToDisk)
            {
                if (testParameters.generateNewTextonization)
                {
                    HelperFunctions.createTextonizationAndSaveToFile(filePaths.trainingTextonsFilePath, forest, trainingSet, parameters);
                    HelperFunctions.createTextonizationAndSaveToFile(filePaths.testTextonsFilePath, forest, testSet, parameters);
                }
                textonTrainingSet = HelperFunctions.readTextonizedImagesFromFile(filePaths.trainingTextonsFilePath);
                textonTestSet = HelperFunctions.readTextonizedImagesFromFile(filePaths.testTextonsFilePath);
            }
            else
            {
                textonTrainingSet = HelperFunctions.createTextonization(forest, trainingSet, parameters);
                textonTestSet = HelperFunctions.createTextonization(forest, testSet, parameters);
            }

            Report.Line(1, "Test case " + Name + ": Training SVM.");

            svm = new STFSVM(testParameters.generateNewSVMKernel);
            svm.train(textonTrainingSet, parameters, filePaths.trainingsetpath, filePaths.kernelsetpath);

            //Result.TrainingSetResult = svm.test(textonTrainingSet, parameters, filePaths.testsetpath1, filePaths.semantictestsetpath1, "Test case " + this.Name + ": training set");
            Result.TrainingSetResult = svm.testWithTrainingset(parameters, "Test case " + this.Name + ": training set");
            Result.TestSetResult = svm.test(textonTestSet, parameters, filePaths.testsetpath2, filePaths.semantictestsetpath2, "Test case " + this.Name + ": test set");
            Result.Name = "Result of test case " + Name;

            Report.End(1);

            return Result;
        }
    }

    //result of a test series
    public class TestSeriesResult
    {
        //collected output string
        public string OutputString;

        //individual testing results (including all multi-runs)
        public TestCaseResult[] TestCaseResults;
        //parameter sets used for the individual test cases (not including multi-run dublicates
        public TrainingParams[] TestCaseTrainingparams;
        public int IndexOfBestParams;   //best params index for convenience

        //write this file into a json string
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

    //a collection of many testcases, able to obtain data from them
    public class TestSeries
    {
        public Dictionary<int, TestCase> TestCases = new Dictionary<int, TestCase>();
        public Dictionary<int, int> TestRunCounts = new Dictionary<int, int>();
        public int Length = 0;
        public FilePaths GlobalFilepaths;
        public STLabelledImage[] GlobalImageSet;
        public string Name;
        public string historyFolderPath;    //path to store all testing results. does not write history if this is null
        public bool readwriteTempFiles;

        //leave historyFolderPath as null to deactivate test history
        //readwriteTempFiles - if generated forests and textonizations should be saved to disk after generation and/or (tried to) read from disk
        public TestSeries(string name, FilePaths globalFilepaths, STLabelledImage[] globalImageSet, string historyFolderPath, bool readwriteTempFiles = false)
        {
            this.Name = name;
            this.GlobalFilepaths = globalFilepaths;
            this.GlobalImageSet = globalImageSet;
            this.historyFolderPath = historyFolderPath;
            this.readwriteTempFiles = readwriteTempFiles;
        }

        //completely specify a new test case
        public void addTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, STLabelledImage[] inputImages, int runCount, string name)
        {
            var test = new TestCase(parameters, testParameters, FilePaths, inputImages, name);
            TestCases.Add(Length, test);
            TestRunCounts.Add(Length, runCount);
            Length++;
        }

        public void addTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, STLabelledImage[] inputImages, string name)
        {
            addTestcase(parameters, testParameters, FilePaths, inputImages, 1, name);
        }

        public void addTestcase(TrainingParams parameters, TestingParams testParameters, STLabelledImage[] inputImages, string name)
        {
            addTestcase(parameters, testParameters, GlobalFilepaths, inputImages, name);
        }
        public void addTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, string name)
        {
            addTestcase(parameters, testParameters, FilePaths, GlobalImageSet, name);
        }
        public void addTestcase(TrainingParams parameters, TestingParams testParameters, string name)
        {
            addTestcase(parameters, testParameters, GlobalFilepaths, GlobalImageSet, name);

        }
        public void addTestcase(TrainingParams parameters, TestingParams testParameters, STLabelledImage[] inputImages, int runCount, string name)
        {
            addTestcase(parameters, testParameters, GlobalFilepaths, inputImages, runCount, name);
        }
        public void addTestcase(TrainingParams parameters, TestingParams testParameters, FilePaths FilePaths, int runCount, string name)
        {
            addTestcase(parameters, testParameters, FilePaths, GlobalImageSet, runCount, name);
        }
        public void addTestcase(TrainingParams parameters, TestingParams testParameters, int runCount, string name)
        {
            addTestcase(parameters, testParameters, GlobalFilepaths, GlobalImageSet, runCount, name);
        }


        //specify only the most important parameters
        public void addSimpleTestcase(string name, int treesCount, int treeDepth, int imageSubsetCount, int samplingWindow, int classesCount, bool enableGridsearch, int runCount, int maxSamples = 999999999)
        {
            TestingParams testParameters = new TestingParams()
            {
                subClass = (classesCount == -1)?false:true,
                classLimit = classesCount - 1,
                trainForestWithEntireSet = true,
                generateNewForest = true,
                generateNewTextonization = true,
                generateNewSVMKernel = true
            };

            TrainingParams parameters = new TrainingParams()
            {
                forestName = "STForest of testcase " + name,
                classesCount = GlobalParams.labels.Max(x => x.Index) + 1,
                treesCount = treesCount,
                maxTreeDepth = treeDepth,
                imageSubsetCount = imageSubsetCount,
                featureType = FeatureType.SelectRandom,
                samplingType = SamplingType.RegularGrid,
                samplingWindow = samplingWindow,
                maxSampleCount = maxSamples,
                featureProviderFactory = new FeatureProviderFactory(),
                samplingProviderFactory = new SamplingProviderFactory(),
                randomSamplingCount = 50,
                thresholdCandidateNumber = 15,
                thresholdInformationGainMinimum = 0.001d,
                classificationMode = ClassificationMode.Semantic,
                forcePassthrough = false,
                enableGridSearch = enableGridsearch
            };

            addTestcase(parameters, testParameters, runCount, name);
        }

        public void addSimpleTestcase(string name, int treesCount, int treeDepth, int imageSubsetCount, int samplingWindow,int maxSamples = 999999999, int runCount = 1, int classesCount = -1 )
        {
            addSimpleTestcase(name, treesCount, treeDepth, imageSubsetCount, samplingWindow, classesCount, false, runCount, maxSamples);
        }

        public TestCaseResult runTestcase(int index)
        {
            return TestCases[index].run(readwriteTempFiles);
        }

        //run each test in sequence, build an output (currently a formatted matrix string)
        public TestSeriesResult runAllTestcases()
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
            resultString.Append(HelperFunctions.spaces("Nr.", shortspaces));
            resultString.Append(HelperFunctions.spaces("test case", outputspaces));
            resultString.Append(HelperFunctions.spaces("classes", shortspaces));
            resultString.Append(HelperFunctions.spaces("trees", shortspaces));
            resultString.Append(HelperFunctions.spaces("depth", shortspaces));
            resultString.Append(HelperFunctions.spaces("subset", shortspaces));
            resultString.Append(HelperFunctions.spaces("window", shortspaces));
            resultString.Append(HelperFunctions.spaces("#feats", shortspaces));
            resultString.Append(HelperFunctions.spaces("avg. precision", outputspaces));
            resultString.Append(HelperFunctions.spaces("variance", mediumspaces));
            resultString.Append(HelperFunctions.spaces("avg. recall", mediumspaces));
            resultString.Append(HelperFunctions.spaces("runcount", shortspaces));
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

                    var testResult = test.run(readwriteTempFiles);
                    resultTests.Add(testResult);

                    currentMultirunResults.Add(testResult);

                }
                
                //get some statistics
                foreach(var tr in currentMultirunResults)
                {
                    var prec = tr.TestSetResult.precision;
                    if (prec < minprec)
                    {
                        minprec = prec;
                    }
                    if (prec > maxprec)
                    {
                        maxprec = prec;
                    }

                    meanprecision += prec;
                    meanrecall += tr.TrainingSetResult.precision;
                }
                meanprecision = meanprecision / (double)runcount;
                meanrecall = meanrecall / (double)runcount;
                foreach (var tr in currentMultirunResults)
                {
                    resultvariance += Math.Pow((tr.TestSetResult.precision - meanprecision), 2.0);
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

                string numfeaturesstring = (test.parameters.maxSampleCount >= 999999999) ?
                    String.Format(CultureInfo.InvariantCulture, "all") :
                    String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.maxSampleCount);

                string classesstring = (test.testParameters.classLimit + 1 == -1) ?
                    String.Format(CultureInfo.InvariantCulture, "all") :
                    String.Format(CultureInfo.InvariantCulture, "{0}", test.testParameters.classLimit + 1);

                //insert matrix values, one row per testcase
                resultString.Append(HelperFunctions.spaces(String.Format(CultureInfo.InvariantCulture, "{0}", i), shortspaces));
                resultString.Append(HelperFunctions.spaces(test.Name, outputspaces));
                resultString.Append(HelperFunctions.spaces(classesstring, shortspaces));
                resultString.Append(HelperFunctions.spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.treesCount), shortspaces));
                resultString.Append(HelperFunctions.spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.maxTreeDepth), shortspaces));
                resultString.Append(HelperFunctions.spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.imageSubsetCount), shortspaces));
                resultString.Append(HelperFunctions.spaces(String.Format(CultureInfo.InvariantCulture, "{0}", test.parameters.samplingWindow), shortspaces));
                resultString.Append(HelperFunctions.spaces(numfeaturesstring, shortspaces));
                resultString.Append(HelperFunctions.spaces(precstring, outputspaces));
                resultString.Append(HelperFunctions.spaces(varstring, mediumspaces));
                resultString.Append(HelperFunctions.spaces(String.Format(CultureInfo.InvariantCulture, "{0:0.0000}", meanrecall), mediumspaces));
                resultString.Append(HelperFunctions.spaces(String.Format(CultureInfo.InvariantCulture, "{0}", runcount), shortspaces));
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
            resultString.Append(bestCase.TrainingSetResult.outputString + bestCase.TestSetResult.outputString);

            result.OutputString = resultString.ToString();
            result.TestCaseResults = resultTests.ToArray();
            result.TestCaseTrainingparams = resultParams.ToArray();
            result.IndexOfBestParams = bestIndex;

            writeHistory(result);

            return result;
        }

        private void writeHistory(TestSeriesResult output)
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

        //loads a previously generated test series result from file
        public static TestSeriesResult loadFromJSON(string path)
        {
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full

            };

            var parsed = JsonConvert.DeserializeObject<TestSeriesResult>(File.ReadAllText(path), settings);

            return parsed;
        }

    }

    #region Parameter Class
    public class TestingParams
    {
        public bool subClass;                     //only use the first few classes
        public int classLimit;                    //this is the number of classes to use if ^ is enabled
        public bool trainForestWithEntireSet;     //train forest with entire image set (instead of only the training set)
        public bool generateNewForest;            //create a new forest (instead of using the existing one -> filename)
        public bool generateNewTextonization;     //create new textonization
        public bool generateNewSVMKernel;         //create new SVM kernels
    }
#endregion
}
