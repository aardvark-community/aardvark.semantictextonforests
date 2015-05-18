using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Aardvark.Base;
using Aardvark.SemanticTextonForests;
using LibSvm;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace Examples
{
    static class Program
    {
        public static readonly string PathTmp;
        public static readonly string PathMsrcTrainingData = @"\\hobel\InOut\STFdata\train";

        public static readonly Dictionary<int, Label> MsrcLabels = new Dictionary<int, Label>()
        {
            {  0, new Label(0, "meadow+animal") },
            {  1, new Label(1, "tree") },
            {  2, new Label(2, "house") },
            {  3, new Label(3, "plane") },
            {  4, new Label(4, "cow") },
            {  5, new Label(5, "face") },
            {  6, new Label(6, "car") },
            {  7, new Label(7, "bike") },
            {  8, new Label(8, "sheep") },
            {  9, new Label(9, "flower") },
            { 10, new Label(10, "sign") },
            { 11, new Label(11, "bird") },
            { 12, new Label(12, "bookshelf") },
            { 13, new Label(13, "books") },
            { 14, new Label(14, "cat") },
            { 15, new Label(15, "dog") },
            { 16, new Label(16, "street") },
            { 17, new Label(17, "water+boa" ) },
            { 18, new Label(18, "person") },
            { 19, new Label(19, "seashore") },
        };

        static Program()
        {
            PathTmp = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "stftmp");
            if (!Directory.Exists(PathTmp)) Directory.CreateDirectory(PathTmp);
        }

        [STAThread]
        static void Main(string[] args)
        {
            //0 = report load times and major results
            //1 = report training of trees, decision progress
            //2 = report numbers of class labels and images, write and read filenames, decision distribution
            //3 = report of each decision node during testing
            //4 = report of each decision node during training
            Report.Verbosity = 2;

            //PredictionTest();

            QuickieTest();

            Report.Line(0, "Reached end of program.");
            Console.ReadLine();
        }
        
        private static void QuickieTest()
        {
            var blapath = Path.Combine(PathTmp, "bla");
            var hpath = Path.Combine(blapath, "h");
            if (!Directory.Exists(blapath)) Directory.CreateDirectory(blapath);
            if (!Directory.Exists(hpath)) Directory.CreateDirectory(hpath);

            var parameters = new TrainingParams(16, 10, 25, 11, Program.MsrcLabels.Values.ToArray(), 5000);
            var images = HelperFunctions.GetMsrcImagesFromDirectory(PathMsrcTrainingData, parameters);

            var ts = new TestSeries("quick", new FilePaths(blapath), images, parameters.Labels, hpath);

            ts.AddSimpleTestcase("fast test", 5, 8, 200, 21, 5);
            
            var tsr = ts.RunAllTestcases();

            Report.Line(tsr.OutputString);
        }

        private static void PredictionTest()
        {
            string workingDirectory = PathTmp;

            var parameters = new TrainingParams(16, 25, 25, 11, Program.MsrcLabels.Values.ToArray(), 5000);

            // (0) Read and Prepare Data

            var images = HelperFunctions.GetMsrcImagesFromDirectory(PathMsrcTrainingData, parameters);

            var tempList = new List<LabeledImage>();
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 4));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 12));
            tempList.AddRange(images.Where(x => x.ClassLabel.Index == 15));
            images = tempList.ToArray();

            LabeledImage[] train;
            LabeledImage[] test;

            images.SplitIntoSets(out train, out test);

            // (1) Train Forest

            var forest = new Forest(parameters.ForestName, parameters.TreesCount);

            forest.Train(train, parameters);
            //forest.writeToFile("foo.json");
            //var foo = HelperFunctions.readForestFromFile("foo.json");

            // (2) Textonize Data

            var trainTextons = train.Textonize(forest, parameters);

            // (3) Train Classifier

            var svm = new Classifier(workingDirectory);

            svm.Train(trainTextons, parameters);

            // (4) Classify!

            Console.WriteLine("Type the index of a picture (max index=" + (test.Length - 1) + ") :");
            while (true)
            {
                var i = Convert.ToInt32(Console.ReadLine());

                var testData = test[i].Textonize(forest, parameters);

                var prediction = svm.PredictLabel(testData, parameters);

                var s = $"Test Image {i}:  Class = {testData.Label.Index} {testData.Label.Name};  Predicted = {prediction.Index } {prediction.Name}";

                Console.WriteLine(s);
            }
        }
    }
}


