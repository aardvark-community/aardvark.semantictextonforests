using Aardvark.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;
using System.Globalization;
using LibSvm;
using static System.Environment;

namespace ScratchAttila
{
    #region EntryPoint
    /// <summary>
    /// Workaround class to call Aardvark.Bootstrapper.Init() 
    /// before any other assembly needs to be loaded.
    /// </summary>
    static class EntryPoint
    {
        [STAThread]
        static void Main(string[] args)
        {
            Program.Start(args);
        }
    }
    #endregion


    class Program
    {
        public static readonly string PathTmp;
        public static readonly string PathMsrcTrainingsData = @"\\hobel\InOut\STFdata\train";

        public static readonly ClassLabel[] MsrcLabels = new[]
        {
            new ClassLabel(0, "meadow+animal"),
            new ClassLabel(1, "tree"),
            new ClassLabel(2, "house"),
            new ClassLabel(3, "plane"),
            new ClassLabel(4, "cow"),
            new ClassLabel(5, "face"),
            new ClassLabel(6, "car"),
            new ClassLabel(7, "bike"),
            new ClassLabel(8, "sheep"),
            new ClassLabel(9, "flower"),
            new ClassLabel(10, "sign"),
            new ClassLabel(11, "bird"),
            new ClassLabel(12, "bookshelf"),
            new ClassLabel(13, "books"),
            new ClassLabel(14, "cat"),
            new ClassLabel(15, "dog"),
            new ClassLabel(16, "street"),
            new ClassLabel(17, "water+boat"),
            new ClassLabel(18, "person"),
            new ClassLabel(19, "seashore"),
        };

        static Program()
        {
            PathTmp = Path.Combine(Environment.GetFolderPath(SpecialFolder.Desktop), "stftmp");
            if (!Directory.Exists(PathTmp)) Directory.CreateDirectory(PathTmp);
        }

        [STAThread]
        public static void Start(string[] args)
        {
            //0 = report load times and major results
            //1 = report training of trees, decision progress
            //2 = report numbers of class labels and images, write and read filenames, decision distribution
            //3 = report of each decision node during testing
            //4 = report of each decision node during training
            Report.Verbosity = 3;

            PredictionTestcase();

            //firstTestcase();

            //secondTestcase();

            //segmentationTestcase();

            

            Report.Line(0, "Reached end of program.");
            Console.ReadLine();

        }

        private static void segmentationTestcase()
        {

            string workingDirectory = "C:\\Users\\aszabo\\Desktop\\Aardwork\\STFu\\temp";

            GlobalParams.Labels = Program.MsrcLabels;

            // (0) Read and Prepare Data

            var images = HelperFunctions.GetLabeledImagesFromDirectory(@"\\hobel\InOut\STFdata\train");

            var tempList = new List<LabeledImage>();

            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 4));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 12));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 15));

            images = tempList.ToArray();

            LabeledImage[] train;
            LabeledImage[] test;

            images.splitIntoSets(out train, out test);

            // (1) Train Forest

            var parameters = new TrainingParams(5, 8, 30, 11, Program.MsrcLabels, 2000);

            var forest = new Forest(parameters.ForestName, parameters.TreesCount);

            forest.Train(train, parameters);

            // (2) Textonize Data

            var trainTextons = train.textonize(forest, parameters);

            // (3) Train Classifier

            var svm = new Classifier(workingDirectory);

            svm.train(trainTextons, parameters);

            // (4) Classify!

            Console.WriteLine("Type the index of a picture (max index=" + (test.Length - 1) + ") :");
            while (true)
            {
                var i = Convert.ToInt32(Console.ReadLine());

                var testData = test[i].textonize(forest, parameters);

                var prediction = svm.predictLabel(testData, parameters);

                string outputString = "Test Image " + i + ": " +
                    " Class = " + testData.Label.Index + " " + testData.Label.Name + "; " +
                    " Predicted = " + prediction.Index + " " + prediction.Name;

                Console.WriteLine(outputString);
            }



        }

        private static void PredictionTestcase()
        {
            string workingDirectory = PathTmp;
            GlobalParams.Labels = Program.MsrcLabels;


            // (0) Read and Prepare Data

            var images = HelperFunctions.GetLabeledImagesFromDirectory(PathMsrcTrainingsData);

            var tempList = new List<LabeledImage>();
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 4));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 12));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 15));
            images = tempList.ToArray();

            LabeledImage[] train;
            LabeledImage[] test;

            images.splitIntoSets(out train, out test);

            // (1) Train Forest

            var parameters = new TrainingParams(5, 8, 30, 11, Program.MsrcLabels, 2000)
            {
                EnableGridSearch = true     //enable searching optimal C using cross validation
            };

            var forest = new Forest(parameters.ForestName, parameters.TreesCount);

            forest.Train(train, parameters);
            //forest.writeToFile("foo.json");
            //var foo = HelperFunctions.readForestFromFile("foo.json");

            // (2) Textonize Data

            var trainTextons = train.textonize(forest, parameters);

            // (3) Train Classifier

            var svm = new Classifier(workingDirectory);

            svm.train(trainTextons, parameters);

            // (4) Classify!

            Console.WriteLine("Type the index of a picture (max index="+(test.Length-1)+") :");
            while(true)
            {
                var i = Convert.ToInt32(Console.ReadLine());

                var testData = test[i].textonize(forest, parameters);

                var prediction = svm.predictLabel(testData, parameters);

                var s = $"Test Image {i}:  Class = {testData.Label.Index} {testData.Label.Name};  Predicted = {prediction.Index } {prediction.Name}";

                Console.WriteLine(s);
            }
        }

        private static void firstTestcase()
        {
            #region Image IO

            string trainingPath = Path.Combine(workDir, trainingImagesSubfoler);
            string forestFilePath = Path.Combine(workDir, forestFilename);
            string trainingsetpath = Path.Combine(Program.workDir, Program.svmTrainingProblemFilename);
            string kernelsetpath = Path.Combine(Program.workDir, svmTrainingKernelFilename);
            string trainingTextonsFilePath = Path.Combine(workDir, semanticTrainingsetFilename);
            string testTextonsFilePath = Path.Combine(workDir, semanticTestSetFilename);

            var fileNames = new FilePaths()
            {
                WorkDir = workDir,
                forestFilePath = forestFilePath,
                kernelsetpath = kernelsetpath,
                testTextonsFilePath = testTextonsFilePath,
                trainingsetpath = trainingsetpath,
                trainingTextonsFilePath = trainingTextonsFilePath
            };

            GlobalParams.Labels = Program.MsrcLabels;

            LabeledImage[] images = HelperFunctions.GetLabeledImagesFromDirectory(trainingPath);

            #endregion

            #region Create and run Test Cases

            string outputFilepath = Path.Combine(outputDir, outputFilename);

            var testSeries = new TestSeries("functionality test series", fileNames, images, outputHistoryDir);

            //testSeries.addSimpleTestcase("test test", 2, 4, 30, 21, 2);

            for (int j = 10; j < 11; j = j + 5) //trees
            {
                for (int i = 5; i < 6; i = i + 3) //levels
                {
                    //5 trees * 5 levels = 25 tests
                    //testSeries.addSimpleTestcase("test " + i +" "+5, 10, i, (int)(((double)3 * 25.0) / 4.0), 15, 3, 5, 500);
                }
            }

            for (int i = 5000; i < 100000; i += 10000)
            {
                //testSeries.addSimpleTestcase("test " + (i), 5, 12, (int)(((double)0.75 * 25.0) / 2.0), 17, 10, 1, i);
            }

            testSeries.addSimpleTestcase("test " + (1), 2, 6, (int)(((double)3 * 25.0) / 4.0), 11, 1000, 1, 3);
            //testSeries.addSimpleTestcase("bla ", 3, 12, (int)(((double)3 * 25.0) / 4.0), 11, 999999999, 1,3);


            var testResult = testSeries.runAllTestcases();

            File.WriteAllText(outputFilepath, testResult.OutputString);

            #endregion
        }

        private static void secondTestcase()
        {

            string trainingPath = Path.Combine(workDir, trainingImagesSubfoler);
            string forestFilePath = Path.Combine(workDir, forestFilename);
            string testsetpath1 = Path.Combine(Program.workDir, Program.svmTestProblemFilename1);
            string testsetpath2 = Path.Combine(Program.workDir, Program.svmTestProblemFilename2);
            string semantictestsetpath1 = Path.Combine(Program.workDir, Program.svmSemanticProblemFilename1);
            string semantictestsetpath2 = Path.Combine(Program.workDir, Program.svmSemanticProblemFilename2);
            string trainingsetpath = Path.Combine(Program.workDir, Program.svmTrainingProblemFilename);
            string kernelsetpath = Path.Combine(Program.workDir, svmTrainingKernelFilename);
            string trainingTextonsFilePath = Path.Combine(workDir, semanticTrainingsetFilename);
            string testTextonsFilePath = Path.Combine(workDir, semanticTestSetFilename);

            var fileNames = new FilePaths()
            {
                forestFilePath = forestFilePath,
                kernelsetpath = kernelsetpath,
                semantictestsetpath1 = semantictestsetpath1,
                semantictestsetpath2 = semantictestsetpath2,
                testsetpath1 = testsetpath1,
                testsetpath2 = testsetpath2,
                testTextonsFilePath = testTextonsFilePath,
                trainingsetpath = trainingsetpath,
                trainingTextonsFilePath = trainingTextonsFilePath
            };
            
            GlobalParams.Labels = new[] {
                new ClassLabel(0, "NOT OK"),
                new ClassLabel(1, "OK"),
            };

            var images = HelperFunctions.getTDatasetFromDirectory(@"C:\Users\aszabo\Desktop\T\TOD\20130314_2VR Vis\Artikel 3"); 

            images =
                images.Where(x => x.ClassLabel.Index == 0).ToArray().GetRandomSubset(12).Concat(
                images.Where(x => x.ClassLabel.Index == 1).ToArray().GetRandomSubset(12))
                .ToArray();
            
            //select about 1200 from each
            //-> results in approx 1 000 000 data points using 29px sampling window

            string outputFilepath = Path.Combine(outputDir, outputFilename);

            var testSeries = new TestSeries("functionality test series", fileNames, images, outputHistoryDir);

            testSeries.addSimpleTestcase("test " + (1), 2, 16, (int)(((double)2.0 * 12.0) / 4.0), 41, 1000, 3);
            testSeries.addSimpleTestcase("bla ", 3, 10, (int)(((double)2.0 * 12.0) / 4.0), 41, 999999999, 1);

            var testResult = testSeries.runAllTestcases();

            File.WriteAllText(outputFilepath, testResult.OutputString);
        }

        #region File Paths
        public const string workDir = "C:\\Users\\aszabo\\Desktop\\Aardwork\\STFu";
        public const string outputDir = "C:\\Users\\aszabo\\Desktop\\Aardwork\\STFu\\output";
        public const string outputHistoryDir = "C:\\Users\\aszabo\\Desktop\\Aardwork\\STFu\\output\\history";
        public const string svmTrainingProblemFilename = "trainingProblem.ds";
        public const string svmTrainingKernelFilename = "trainingKernel.ds";
        public const string svmTestProblemFilename2 = "testProblem2.ds";
        public const string svmTestProblemFilename1 = "testProblem1.ds";
        public const string svmSemanticProblemFilename2 = "semanticProblem2.ds";
        public const string svmSemanticProblemFilename1 = "semanticProblem1.ds";
        const string forestFilename = "currentForest.json";
        const string textonizedTrainingsetFilename = "TrainingTextonsLeafonly.json";
        const string textonizedTestSetFilename = "TestTextonsLeafonly.json";
        const string semanticTrainingsetFilename = "TrainingTextonsSemantic.json";
        const string semanticTestSetFilename = "TestTextonsSemantic.json";
        static string outputFilename = "output.txt";

        const string trainingImagesSubfoler = "train";
        #endregion
    }
}


            