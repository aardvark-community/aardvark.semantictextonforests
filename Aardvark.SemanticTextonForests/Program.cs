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

            Report.Line(0, "Reached end of program.");
            Console.ReadLine();

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

            images.SplitIntoSets(out train, out test);

            // (1) Train Forest

            var parameters = new TrainingParams(3, 6, 25, 19, Program.MsrcLabels, 2000);

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

            Console.WriteLine("Type the index of a picture (max index="+(test.Length-1)+") :");
            while(true)
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


            