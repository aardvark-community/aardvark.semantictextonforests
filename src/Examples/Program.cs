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
        public static readonly string PathMsrcSegmentationData = @"\\hobel\InOut\STFdata\GroundTruth";

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
            Report.Verbosity = 1;

            //PredictionTest();

            //QuickieTest();

            SegmentationTest();

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

            //ts.AddSimpleTestcase("fast test", 5, 8, 200, 21, 5);

            for(int i=1; i<10; i++)
            {
                ts.AddSimpleTestcase("fast test", 5, 10, 200, (2*i+1)*2, 5, 7500);
            }
            
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
            tempList.AddRange(images.Where(x => x.Label.Index == 4));
            tempList.AddRange(images.Where(x => x.Label.Index == 12));
            tempList.AddRange(images.Where(x => x.Label.Index == 15));
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

        private static void SegmentationTest()
        {
            // (0) PREPARE DATA

            var parameters = new TrainingParams(5, 10, 20000, 3, Program.MsrcLabels.Values.ToArray());
            parameters.SegmentationLabels = MsrcSegmentationLabels.Values.ToArray();
            parameters.ColorizationRule = MrscColorizationRule;
            parameters.MappingRule = MsrcMappingRule;
            
            Report.BeginTimed(1, "Reading Patches");
            var patches = HelperFunctions.GetMsrcSegmentationDataset(PathMsrcTrainingData, PathMsrcSegmentationData, parameters);
            Report.End(1);

            //24k+ patches, which is currently too much -> select small subset
            var trainList = new List<LabeledPatch>();
            int ccount = 300;

            trainList.AddRange(patches.Where(x => x.ParentImage.Label.Index == 0).ToArray().Take(ccount));
            trainList.AddRange(patches.Where(x => x.ParentImage.Label.Index == 1).ToArray().Take(ccount));
            trainList.AddRange(patches.Where(x => x.ParentImage.Label.Index == 2).ToArray().Take(ccount));

            // (1) TRAIN CLASSIFIER

            var forest = new Forest("seg test", parameters.TreesCount);
            forest.Train(trainList.ToArray(), parameters);
            var trainTextons = trainList.ToArray().Textonize(forest, parameters);
            var svm = new Classifier(PathTmp);
            svm.Train(trainTextons, parameters);

            // (2) SEGMENT IMAGE AND GENERATE OUTPUT

            while (true)
            {
                Console.WriteLine("Type a number from 0 to 2 :");

                var i = Convert.ToInt32(Console.ReadLine());

                var p = patches;
                var pw = p.Where(x => x.ParentImage.Label.Index == i).ToArray();
                var pwr = pw.GetRandomSubset(10).First();
                var pwrp = pwr.ParentImage.Image.ImagePath;

                //var inFilename = patches.Where(x => x.ParentImage.Label.Index == i).ToArray().GetRandomSubset(10).First().ParentImage.Image.ImagePath;
                var inFilename = pwrp;

                var fn = Path.GetFileNameWithoutExtension(inFilename);
                Console.WriteLine($"Selected picture with filename {fn}");

                var PatchesOfInputImage = patches.Where(x => x.ParentImage.Image.ImagePath == inFilename).ToArray();
                var inLabels = svm.PredictLabels(PatchesOfInputImage.Textonize(forest, parameters), parameters);

                HelperFunctions.WriteSegmentationOutputOfOneImage(PatchesOfInputImage, inLabels, parameters, 
                    Path.Combine(PathTmp, $"out_{fn}.bmp"));
                Console.WriteLine($"Segmentation complete! See output in working directory.");
            }
        }

        public static readonly Dictionary<int, Label> MsrcSegmentationLabels = new Dictionary<int, Label>()
        {
            //not all labels, only from the first three classes
            {  0, new Label(0, "grass") },
            {  1, new Label(1, "tree") },
            {  2, new Label(2, "cow") },
            {  3, new Label(3, "mountain") },
            {  4, new Label(4, "horse") },
            {  5, new Label(5, "sheep") },
            {  6, new Label(6, "building") },
            {  7, new Label(7, "sky") },
            {  8, new Label(8, "road") },
            {  9, new Label(9, "unknown") }
        };

        public static readonly SegmentationMappingRule MsrcMappingRule = (labels, image, x, y) =>
        {
            Label result;

            var r = image.GetChannel(Col.Channel.Red)[x, y];
            var g = image.GetChannel(Col.Channel.Green)[x, y];
            var b = image.GetChannel(Col.Channel.Blue)[x, y];
            //var a = image.GetChannel(Col.Channel.Alpha)[x, y];

            if (r == 0 && g == 128 && b == 0 )
            {
                result = labels[0]; //grass
            }
            else if (r == 128 && g == 128 && b == 0 )
            {
                result = labels[1]; //tree
            }
            else if (r == 0 && g == 0 && b == 128 )
            {
                result = labels[2]; //cow
            }
            else if (r == 64 && g == 0 && b == 0 )
            {
                result = labels[3]; //mountain
            }
            else if (r == 128 && g == 0 && b == 128 )
            {
                result = labels[4]; //horse
            }
            else if (r == 0 && g == 128 && b == 128 )
            {
                result = labels[5]; //sheep
            }
            else if (r == 128 && g == 0 && b == 0 )
            {
                result = labels[6]; //building
            }
            else if (r == 128 && g == 128 && b == 128 )
            {
                result = labels[7]; //sky
            }
            else if (r == 128 && g == 64 && b == 128 )
            {
                result = labels[8]; //road
            }
            else
            {
                result = labels[9]; //unknown or unlabeled
            }

            return result;
        };

        public static readonly SegmentationColorizationRule MrscColorizationRule = (label) =>
        {
            C3b result;

            if (label.Index == 0)
            {
                result = C3b.Green; //grass
            }
            else if (label.Index == 1)
            {
                result = C3b.DarkGreen; //tree
            }
            else if (label.Index == 2)
            {
                result = C3b.Blue; //cow
            }
            else if (label.Index == 3)
            {
                result = C3b.DarkBlue; //mountain
            }
            else if (label.Index == 4)
            {
                result = C3b.DarkYellow; //horse
            }
            else if (label.Index == 5)
            {
                result = C3b.White; //sheep
            }
            else if (label.Index == 6)
            {
                result = C3b.Gray; //building
            }
            else if (label.Index == 7)
            {
                result = C3b.Cyan; //sky
            }
            else if (label.Index == 8)
            {
                result = C3b.Magenta; //road
            }
            else
            {
                result = C3b.Black; //unknown or unlabeled
            }

            return result;
        };
    }
}


