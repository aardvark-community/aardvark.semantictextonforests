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

            GlobalParams.Labels = new ClassLabel[] {
                new ClassLabel() {Index=0, Name="meadow+animal"},
                new ClassLabel() {Index=1, Name="tree"},
                new ClassLabel() {Index=2, Name="house"},
                new ClassLabel() {Index=3, Name="plane"},
                new ClassLabel() {Index=4, Name="cow"},
                new ClassLabel() {Index=5, Name="face"},
                new ClassLabel() {Index=6, Name="car"},
                new ClassLabel() {Index=7, Name="bike"},
                new ClassLabel() {Index=8, Name="sheep"},
                new ClassLabel() {Index=9, Name="flower"},
                new ClassLabel() {Index=10, Name="sign"},
                new ClassLabel() {Index=11, Name="bird"},
                new ClassLabel() {Index=12, Name="bookshelf"},
                new ClassLabel() {Index=13, Name="books"},
                new ClassLabel() {Index=14, Name="cat"},
                new ClassLabel() {Index=15, Name="dog"},
                new ClassLabel() {Index=16, Name="street"},
                new ClassLabel() {Index=17, Name="water+boat"},
                new ClassLabel() {Index=18, Name="person"},
                new ClassLabel() {Index=19, Name="seashore"},
            };

            // (0) Read and Prepare Data

            var images = HelperFunctions.GetLabeledImagesFromDirectory(@"\\hobel\InOut\STFdata\train");

            var tempList = new List<STLabeledImage>();

            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 4));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 12));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 15));

            images = tempList.ToArray();

            STLabeledImage[] train;
            STLabeledImage[] test;

            images.splitIntoSets(out train, out test);

            // (1) Train Forest

            var parameters = new TrainingParams(5, 8, 30, 11, 2000);

            var forest = new STForest(parameters);

            forest.train(train, parameters);

            // (2) Textonize Data

            var trainTextons = train.textonize(forest, parameters);

            // (3) Train Classifier

            var svm = new STFSVM(workingDirectory);

            svm.train(trainTextons, parameters);

            // (4) Classify!

            Console.WriteLine("Type the index of a picture (max index=" + (test.Length - 1) + ") :");
            while (true)
            {
                var i = Convert.ToInt32(Console.ReadLine());

                var testData = test[i].textonize(forest, parameters);

                var prediction = svm.predictLabel(testData, parameters);

                string outputString = "Test Image " + i + ": " +
                    " Class = " + testData.ClassLabel.Index + " " + testData.ClassLabel.Name + "; " +
                    " Predicted = " + prediction.Index + " " + prediction.Name;

                Console.WriteLine(outputString);
            }



        }

        private static void PredictionTestcase()
        {
            string workingDirectory = PathTmp;
            
            GlobalParams.Labels = new ClassLabel[] {
                new ClassLabel() {Index=0, Name="meadow+animal"},
                new ClassLabel() {Index=1, Name="tree"},
                new ClassLabel() {Index=2, Name="house"},
                new ClassLabel() {Index=3, Name="plane"},
                new ClassLabel() {Index=4, Name="cow"},
                new ClassLabel() {Index=5, Name="face"},
                new ClassLabel() {Index=6, Name="car"},
                new ClassLabel() {Index=7, Name="bike"},
                new ClassLabel() {Index=8, Name="sheep"},
                new ClassLabel() {Index=9, Name="flower"},
                new ClassLabel() {Index=10, Name="sign"},
                new ClassLabel() {Index=11, Name="bird"},
                new ClassLabel() {Index=12, Name="bookshelf"},
                new ClassLabel() {Index=13, Name="books"},
                new ClassLabel() {Index=14, Name="cat"},
                new ClassLabel() {Index=15, Name="dog"},
                new ClassLabel() {Index=16, Name="street"},
                new ClassLabel() {Index=17, Name="water+boat"},
                new ClassLabel() {Index=18, Name="person"},
                new ClassLabel() {Index=19, Name="seashore"},
            };

            // (0) Read and Prepare Data

            var images = HelperFunctions.GetLabeledImagesFromDirectory(PathMsrcTrainingsData);

            var tempList = new List<STLabeledImage>();

            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 4));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 12));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 15));

            images = tempList.ToArray();

            STLabeledImage[] train;
            STLabeledImage[] test;

            images.splitIntoSets(out train, out test);

            // (1) Train Forest

            var parameters = new TrainingParams(5, 8, 30, 11, 2000);

            var forest = new STForest(parameters);

            forest.train(train, parameters);

            // (2) Textonize Data

            var trainTextons = train.textonize(forest, parameters);

            // (3) Train Classifier

            var svm = new STFSVM(workingDirectory);

            svm.train(trainTextons, parameters);

            // (4) Classify!

            Console.WriteLine("Type the index of a picture (max index="+(test.Length-1)+") :");
            while(true)
            {
                var i = Convert.ToInt32(Console.ReadLine());

                var testData = test[i].textonize(forest, parameters);

                var prediction = svm.predictLabel(testData, parameters);

                string outputString = "Test Image " + i + ": "+
                    " Class = " + testData.ClassLabel.Index + " " + testData.ClassLabel.Name + "; "+
                    " Predicted = " + prediction.Index + " " + prediction.Name;

                Console.WriteLine(outputString);
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

            ClassLabel[] labels = new ClassLabel[] {
                new ClassLabel() {Index=0, Name="meadow+animal"},
                new ClassLabel() {Index=1, Name="tree"},
                new ClassLabel() {Index=2, Name="house"},
                new ClassLabel() {Index=3, Name="plane"},
                new ClassLabel() {Index=4, Name="cow"},
                new ClassLabel() {Index=5, Name="face"},
                new ClassLabel() {Index=6, Name="car"},
                new ClassLabel() {Index=7, Name="bike"},
                new ClassLabel() {Index=8, Name="sheep"},
                new ClassLabel() {Index=9, Name="flower"},
                new ClassLabel() {Index=10, Name="sign"},
                new ClassLabel() {Index=11, Name="bird"},
                new ClassLabel() {Index=12, Name="bookshelf"},
                new ClassLabel() {Index=13, Name="books"},
                new ClassLabel() {Index=14, Name="cat"},
                new ClassLabel() {Index=15, Name="dog"},
                new ClassLabel() {Index=16, Name="street"},
                new ClassLabel() {Index=17, Name="water+boat"},
                new ClassLabel() {Index=18, Name="person"},
                new ClassLabel() {Index=19, Name="seashore"},
            };
            GlobalParams.Labels = labels;

            STLabeledImage[] images = HelperFunctions.GetLabeledImagesFromDirectory(trainingPath);

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

            ClassLabel[] labels = new ClassLabel[] {
                new ClassLabel() {Index=0, Name="NOT OK"},
                new ClassLabel() {Index=1, Name="OK"},
            };
            GlobalParams.Labels = labels;

            STLabeledImage[] images = HelperFunctions.getTDatasetFromDirectory(@"C:\Users\aszabo\Desktop\T\TOD\20130314_2VR Vis\Artikel 3"); 

            var l1 = images.Where(x => x.ClassLabel.Index == 0)/*.RandomOrder()*/.Take(12);
            var l2 = images.Where(x => x.ClassLabel.Index == 1)/*.RandomOrder()*/.Skip(12).Take(12);

            var result = new List<STLabeledImage>();

            result.AddRange(l1);
            result.AddRange(l2);

            images = result.ToArray();
            
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


            