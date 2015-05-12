using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Aardvark.Base;
using Newtonsoft.Json;

namespace ScratchAttila
{
    #region STF training

    public static class Algo
    {
        public static Random Rand = new Random();
        public static int TreeCounter = 0;        //progress counter
        public static int NodeProgressCounter = 0;               //progress report

        public static int NodeIndexCounter = 0;          //the counter variable to determine a node's global index
        
        public static void Train(this Forest forest, LabeledImage[] trainingImages, TrainingParams parameters)
        {
            NodeIndexCounter = -1;

            Report.BeginTimed(0, "Training Forest of " + forest.Trees.Length + " trees with " + trainingImages.Length + " images.");

            TreeCounter = 0;

            Parallel.ForEach(forest.Trees, tree =>
            //foreach (var tree in forest.Trees)
            {
                var currentSubset = trainingImages.GetRandomSubset(parameters.ImageSubsetCount);

                Report.BeginTimed(1, "Training tree " + (tree.Index + 1) + " of " + forest.Trees.Length + ".");

                tree.Train(currentSubset, parameters);

                Report.Line(2, "Finished training tree with " + NodeProgressCounter + " nodes.");

                Report.End(1);
            }
            );

            forest.NumNodes = forest.Trees.Sum(x => x.NumNodes);

            Report.End(0);
        }

        public static void Train(this Tree tree, LabeledImage[] trainingImages, TrainingParams parameters)
        {
            var nodeCounterObject = new NodeCountObject();
            var provider = parameters.SamplingProviderFactory.GetNewProvider();
            var baseDPS = provider.GetDataPoints(trainingImages);
            var baseClassDist = new LabelDistribution(GlobalParams.Labels, baseDPS);

            tree.Root.TrainRecursive(null, baseDPS, parameters, 0, baseClassDist, nodeCounterObject);
            tree.NumNodes = nodeCounterObject.Counter;

            NodeProgressCounter = nodeCounterObject.Counter;
        }

        public static void TrainRecursive(this Node node, Node parent, DataPointSet currentData, TrainingParams parameters, int depth, LabelDistribution currentClassDist, NodeCountObject currentNodeCounter)
        {
            currentNodeCounter.Increment();

            node.GlobalIndex = Interlocked.Increment(ref NodeIndexCounter);

            //create a decider object and train it on the incoming data
            node.Decider = new Decider();

            //only one sampling rule per tree (currently only one exists, regular sampling)
            if(parent == null)
            {
                node.Decider.SamplingProvider = parameters.SamplingProviderFactory.GetNewProvider();
            }
            else
            {
                node.Decider.SamplingProvider = parent.Decider.SamplingProvider;
            }

            node.ClassDistribution = currentClassDist;

            //get a new feature provider for this node
            node.Decider.FeatureProvider = parameters.FeatureProviderFactory.GetNewProvider();
            node.DistanceFromRoot = depth;
            int newdepth = depth + 1;

            DataPointSet leftRemaining;
            DataPointSet rightRemaining;
            LabelDistribution leftClassDist;
            LabelDistribution rightClassDist;

            //training step: the decider finds the best split threshold for the current data
            var trainingResult = node.Decider.InitializeDecision(currentData, currentClassDist, parameters, out leftRemaining, out rightRemaining, out leftClassDist, out rightClassDist);

            bool passthroughDeactivated = (!parameters.ForcePassthrough && trainingResult == DeciderTrainingResult.PassThrough);

            if (trainingResult == DeciderTrainingResult.Leaf //node is a leaf (empty)
                || depth >= parameters.MaxTreeDepth - 1        //node is at max level
                || passthroughDeactivated)                   //this node should pass through (information gain too small) but passthrough mode is deactivated
            //-> leaf
            {
                Report.Line(3, "->LEAF remaining dp=" + currentData.Count + "; depth=" + depth);
                node.isLeaf = true;
                return;
            }

            if (trainingResult == DeciderTrainingResult.PassThrough) //too small information gain and not at max level -> create pass through node (copy from parent)
            {
                if (parent == null)      //empty tree or empty training set
                {
                    node.isLeaf = true;
                    return;
                }

                Report.Line(3, "PASS THROUGH NODE dp#=" + currentData.Count + "; depth=" + depth + " t=" + parent.Decider.DecisionThreshold);

                node.Decider.DecisionThreshold = parent.Decider.DecisionThreshold;
                node.ClassDistribution = parent.ClassDistribution;
                node.Decider.FeatureProvider = parent.Decider.FeatureProvider;
                node.Decider.SamplingProvider = parent.Decider.SamplingProvider;
            }

            var rightNode = new Node();
            var leftNode = new Node();

            TrainRecursive(rightNode, node, rightRemaining, parameters, newdepth, rightClassDist, currentNodeCounter);
            TrainRecursive(leftNode, node, leftRemaining, parameters, newdepth, leftClassDist, currentNodeCounter);

            node.RightChild = rightNode;
            node.LeftChild = leftNode;
        }

        public static TextonizedLabeledImage Textonize(this LabeledImage image, Forest forest, TrainingParams parameters)
        {
            return new[] { image }.Textonize(forest, parameters)[0]; ;
        }

        public static TextonizedLabeledImage[] Textonize(this LabeledImage[] images, Forest forest, TrainingParams parameters)
        {
            var result = new TextonizedLabeledImage[images.Length];

            Report.BeginTimed(0, "Textonizing " + images.Length + " images.");

            int count = 0;
            Parallel.For(0, images.Length, i =>
            //for (int i = 0; i < images.Length; i++)
            {
                //Report.Progress(0, (double)i / (double)images.Length);
                var img = images[i];

                var dist = forest.GetTextonRepresentation(img.Image, parameters);

                result[i] = new TextonizedLabeledImage(img, dist);

                Report.Line("{0} of {1} images textonized", Interlocked.Increment(ref count), images.Length);
            }
            );

            Report.End(0);

            return result;
        }

        public static TextonizedLabeledImage[] NormalizeInvDocFreq(this TextonizedLabeledImage[] images)
        {
            //assumes each feature vector has the same length, which is always the case currently
            var result = images;

            //for each column
            for (int i = 0; i < images[0].Textonization.Values.Length; i++)
            {
                double nonzerocounter = 0;
                //for each row
                for (int datapoint = 0; datapoint < images.Length; datapoint++)
                {
                    if(images[datapoint].Textonization.Values[i] != 0)
                    {
                        //nonzerocounter += images[datapoint].Textonization.Values[i];
                        nonzerocounter += 1.0;
                    }
                }
                double normalizationFactor = Math.Log((double)nonzerocounter / (double)images.Length);
                for (int datapoint = 0; datapoint < images.Length; datapoint++)
                {
                    result[datapoint].Textonization.Values[i] *= normalizationFactor;
                }
            }

            return result;
        }

        public enum DeciderTrainingResult
        {
            Leaf,           //become a leaf
            PassThrough,    //copy the parent
            InnerNode       //continue training normally
        }
    }
    #endregion

    #region Temp. Helper Functions

    public class NodeCountObject
    {
        public int Counter = 0;
        public void Increment()
        {
            Counter++;
        }
    }

    public static class HelperFunctions     //temporary helper functions
    {
        public static void SplitIntoSets(this LabeledImage[] images, out LabeledImage[] training, out LabeledImage[] test)
        {
            ////50/50 split
            var tro = new List<LabeledImage>();
            var teo = new List<LabeledImage>();

            foreach (var img in images)
            {
                var rn = Algo.Rand.NextDouble();
                if (rn >= 0.5)
                {
                    tro.Add(img);
                }
                else
                {
                    teo.Add(img);
                }
            }

            training = tro.ToArray();
            test = teo.ToArray();
        }

        public static double ToDouble(this byte par)
        {
            return ((double)par) * (1d / 255d);
        }

        public static void WriteToFile(this Forest forest, string filename)
        {
            Report.Line(2, "Writing forest to file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full

            };
            var s = JsonConvert.SerializeObject(forest, Formatting.Indented, settings);
            File.WriteAllText(filename, s);

        }

        public static Forest ReadForestFromFile(string filename)
        {
            Report.Line(2, "Reading forest from file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
            };

            var parsed = JsonConvert.DeserializeObject<Forest>(File.ReadAllText(filename), settings);
            return parsed;
        }

        public static void WriteToFile(this TextonizedLabeledImage[] images, string filename)
        {
            Report.Line(2, "Writing textonized image set to file " + filename);

            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full

            };


            var s = JsonConvert.SerializeObject(images, Formatting.Indented, settings);

            File.WriteAllText(filename, s);
        }

        public static TextonizedLabeledImage[] ReadTextonizedImagesFromFile(string filename)
        {
            Report.Line(2, "Reading textonized image set from file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
            };

            var parsed = JsonConvert.DeserializeObject<TextonizedLabeledImage[]>(File.ReadAllText(filename), settings);
            return parsed;
        }

        public static void CreateNewForestAndSaveToFile(string filename, LabeledImage[] trainingSet, TrainingParams parameters)
        {
            var forest = new Forest(parameters.ForestName, parameters.TreesCount);
            Report.Line(2, "Creating new forest " + forest.Name + ".");
            forest.Train(trainingSet, parameters);

            Report.Line(2, "Saving forest " + forest.Name + " to file.");
            forest.WriteToFile(filename);
        }
        
        public static Forest CreateNewForest(LabeledImage[] trainingSet, TrainingParams parameters)
        {
            var forest = new Forest(parameters.ForestName, parameters.TreesCount);
            Report.Line(2, "Creating new forest " + forest.Name + ".");
            forest.Train(trainingSet, parameters);
            return forest;
        }
        
        public static void CreateTextonizationAndSaveToFile(string filename, Forest forest, LabeledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.Textonize(forest, parameters);
            Report.Line(2, "Saving textonization to file.");
            texImgs.WriteToFile(filename);
        }

        public static TextonizedLabeledImage[] CreateTextonization(Forest forest, LabeledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.Textonize(forest, parameters);
            return texImgs;
        }

        //string formatting -> add spaces until the total length of the string = number
        public static string Spaces(string prevValue, int number)
        {
            int numchar = prevValue.Length;
            var result = new StringBuilder();
            result.Append(prevValue);

            for (int i = prevValue.Length; i < number; i++)
            {
                result.Append(" ");
            }

            return result.ToString();
        }

        public static LabeledImage[] GetLabeledImagesFromDirectory(string directoryPath)
        {
            string[] picFiles = Directory.GetFiles(directoryPath);
            var result = new LabeledImage[picFiles.Length];
            for (int i = 0; i < picFiles.Length; i++)
            {
                var s = picFiles[i];
                string currentFilename = Path.GetFileNameWithoutExtension(s);
                string[] filenameSplit = currentFilename.Split('_');
                int fileLabel = Convert.ToInt32(filenameSplit[0]);
                Label currentLabel = GlobalParams.Labels.First(x => x.Index == fileLabel - 1);
                result[i] = new LabeledImage(s, currentLabel);
            }
            return result;
        }

        public static LabeledImage[] GetTDatasetFromDirectory(string directoryPath)
        {
            string nokpath = Path.Combine(directoryPath, "NOK");
            string okpath = Path.Combine(directoryPath, "OK");
            string[] nokFiles = Directory.GetFiles(nokpath);
            string[] okFiles = Directory.GetFiles(okpath);

            var result = new LabeledImage[okFiles.Length + nokFiles.Length];

            for (int i = 0; i < nokFiles.Length; i++)
            {
                var s = nokFiles[i];
                result[i] = new LabeledImage(s, GlobalParams.Labels[0]);
            }

            for (int i = 0; i < okFiles.Length; i++)
            {
                var s = okFiles[i];
                result[nokFiles.Length + i] = new LabeledImage(s, GlobalParams.Labels[1]);
            }

            return result;
        }
    }

    #endregion

    #region Providers

    public enum ClassificationMode
    {
        LeafOnly,
        Semantic
    }

    public enum FeatureType
    {
        RandomPixelValue,
        RandomTwoPixelSum,
        RandomTwoPixelDifference,
        RandomTwoPixelAbsDiff,
        SelectRandom
    };

    public enum SamplingType
    {
        RegularGrid,
        RandomPoints
    };

    public class FeatureProviderFactory
    {
        IFeatureProvider CurrentProvider;
        int PixWinSize;
        FeatureType CurrentChoice;

        public void SelectProvider(FeatureType featureType, int pixelWindowSize)
        {
            CurrentChoice = featureType;
            PixWinSize = pixelWindowSize;
        }

        public IFeatureProvider GetNewProvider()
        {
            switch (CurrentChoice)
            {
                case FeatureType.RandomPixelValue:
                    CurrentProvider = new ValueOfPixelFeatureProvider();
                    CurrentProvider.Init(PixWinSize);
                    break;
                case FeatureType.RandomTwoPixelSum:
                    CurrentProvider = new PixelSumFeatureProvider();
                    CurrentProvider.Init(PixWinSize);
                    break;
                case FeatureType.RandomTwoPixelAbsDiff:
                    CurrentProvider = new AbsDiffOfPixelFeatureProvider();
                    CurrentProvider.Init(PixWinSize);
                    break;
                case FeatureType.RandomTwoPixelDifference:
                    CurrentProvider = new PixelDifferenceFeatureProvider();
                    CurrentProvider.Init(PixWinSize);
                    break;
                case FeatureType.SelectRandom:      //select one of the three providers at random - equal chance
                    var choice = Algo.Rand.Next(4);
                    switch(choice)
                    {
                        case 0:
                            CurrentProvider = new ValueOfPixelFeatureProvider();
                            CurrentProvider.Init(PixWinSize);
                            break;
                        case 1:
                            CurrentProvider = new PixelSumFeatureProvider();
                            CurrentProvider.Init(PixWinSize);
                            break;
                        case 2:
                            CurrentProvider = new AbsDiffOfPixelFeatureProvider();
                            CurrentProvider.Init(PixWinSize);
                            break;
                        case 3:
                            CurrentProvider = new PixelDifferenceFeatureProvider();
                            CurrentProvider.Init(PixWinSize);
                            break;
                        default:
                            return null;
                    }
                    break;
                default:
                    CurrentProvider = new ValueOfPixelFeatureProvider();
                    CurrentProvider.Init(PixWinSize);
                    break;

            }
            return CurrentProvider;
        }
    }

    public class PixelSumFeatureProvider : IFeatureProvider
    {
        public int FX;
        public int FY;
        public int SX;
        public int SY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(FX, FY); }
            set { FX = value.X; FY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(SX, SY); }
            set { SX = value.X; SY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = Algo.Rand.Next(pixelWindowSize) - half;
            int firstY = Algo.Rand.Next(pixelWindowSize) - half;
            int secondX = Algo.Rand.Next(pixelWindowSize) - half;
            int secondY = Algo.Rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override Feature GetFeature(DataPoint point)
        {
            Feature result = new Feature();

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().ToDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().ToDouble();

            var op = (sample1 + sample2) / 2.0; //divide by two for normalization

            result.Value = op;

            return result;
        }
    }

    internal static class MatrixCache
    {
        private static Dictionary<PixImage<byte>, Matrix<byte, C3b>> s_cache = new Dictionary<PixImage<byte>, Matrix<byte, C3b>>();

        public static Matrix<byte, C3b> GetMatrixFrom(PixImage<byte> image)
        {
            Matrix<byte, C3b> result;
            if (s_cache.TryGetValue(image, out result)) return result;

            result = image.GetMatrix<C3b>();
            s_cache[image] = result;
            return result;
        }
    }

    public class PixelDifferenceFeatureProvider : IFeatureProvider
    {
        public int FX;
        public int FY;
        public int SX;
        public int SY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(FX, FY); }
            set { FX = value.X; FY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(SX, SY); }
            set { SX = value.X; SY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = Algo.Rand.Next(pixelWindowSize) - half;
            int firstY = Algo.Rand.Next(pixelWindowSize) - half;
            int secondX = Algo.Rand.Next(pixelWindowSize) - half;
            int secondY = Algo.Rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }
        
        public override Feature GetFeature(DataPoint point)
        {
            Feature result = new Feature();

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);
            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().ToDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().ToDouble();

            var op = ((sample1 - sample2)+1.0) / 2.0; //normalize to [0,1]

            result.Value = op;

            return result;
        }
    }

    public class ValueOfPixelFeatureProvider : IFeatureProvider
    {
        public int X;
        public int Y;

        [JsonIgnore]
        public V2i PixelOffset
        {
            get { return new V2i(X, Y); }
            set { X = value.X; Y = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {

            int half = (int)(pixelWindowSize / 2);
            int x = Algo.Rand.Next(pixelWindowSize) - half;
            int y = Algo.Rand.Next(pixelWindowSize) - half;

            PixelOffset = new V2i(x, y);
        }

        public override Feature GetFeature(DataPoint point)
        {
            Feature result = new Feature();

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);

            var sample = pi[point.PixelCoords + PixelOffset].ToGrayByte().ToDouble();

            result.Value = sample;

            return result;
        }
    }

    public class AbsDiffOfPixelFeatureProvider : IFeatureProvider
    {
        public int FX;
        public int FY;
        public int SX;
        public int SY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(FX, FY); }
            set { FX = value.X; FY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(SX, SY); }
            set { SX = value.X; SY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = Algo.Rand.Next(pixelWindowSize) - half;
            int firstY = Algo.Rand.Next(pixelWindowSize) - half;
            int secondX = Algo.Rand.Next(pixelWindowSize) - half;
            int secondY = Algo.Rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override Feature GetFeature(DataPoint point)
        {
            Feature result = new Feature();

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().ToDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().ToDouble();

            var op = Math.Abs(sample2 - sample1);

            result.Value = op;

            return result;
        }
    }

    public class SamplingProviderFactory
    {
        ISamplingProvider CurrentProvider;
        int PixelWindowSize;
        SamplingType SamplingType;
        int RandomSampleCount = 0;

        public void SelectProvider(SamplingType samplingType, int pixelWindowSize)
        {
            this.PixelWindowSize = pixelWindowSize;
            this.SamplingType = samplingType;
        }

        public void SelectProvider(SamplingType samplingType, int pixelWindowSize, int randomSampleCount)
        {
            this.PixelWindowSize = pixelWindowSize;
            this.SamplingType = samplingType;
            this.RandomSampleCount = randomSampleCount;
        }

        public ISamplingProvider GetNewProvider()
        {

            switch (SamplingType)
            {
                case SamplingType.RegularGrid:
                    CurrentProvider = new RegularGridSamplingProvider();
                    CurrentProvider.Init(PixelWindowSize);
                    break;
                case SamplingType.RandomPoints:
                    var result = new RandomPointSamplingProvider();
                    result.Init(PixelWindowSize);
                    result.SampleCount = this.RandomSampleCount;
                    CurrentProvider = result;
                    break;
                default:
                    CurrentProvider = new RegularGridSamplingProvider();
                    CurrentProvider.Init(PixelWindowSize);
                    break;
            }

            return CurrentProvider;
        }
    }

    public class RegularGridSamplingProvider : ISamplingProvider
    {
        public int PixWinSize;

        public override void Init(int pixWindowSize)
        {
            PixWinSize = pixWindowSize;
        }

        public override DataPointSet GetDataPoints(Image image)
        {
            //currently, this gets a regular grid starting from the top left and continuing as long as there are pixels left.
            var pi = MatrixCache.GetMatrixFrom(image.PixImage);

            var result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling((double)PixWinSize / 2.0f); //ceiling cuts away too much in most cases

            int pointCounter = 0;

            for (int x = borderOffset; x < pi.SX - borderOffset; x = x + PixWinSize)
            {
                for (int y = borderOffset; y < pi.SY - borderOffset; y = y + PixWinSize)
                {
                    var newDP = new DataPoint(image, x, y);
                    result.Add(newDP);
                    pointCounter = pointCounter + 1;
                }
            }

            var bias = (double)1 / (double)pointCounter;     //weigh the sample points by the image's size (larger image = lower weight)

            var resDPS = new DataPointSet();
            resDPS.Points = result.ToArray();
            resDPS.Weight = bias;

            return resDPS;
        }

        public override DataPointSet GetDataPoints(LabeledImage[] images)
        {
            var result = new DataPointSet();
            
            foreach(var img in images)
            {
                var currentDPS = GetDataPoints(img.Image);
                result += new DataPointSet(
                    currentDPS.Points.Copy(x => x.SetLabel(img.ClassLabel.Index)),
                    currentDPS.Weight
                    );
            }
            
            return result;
        }
    }

    public class RandomPointSamplingProvider : ISamplingProvider
    {
        public int PixWinSize;
        public int SampleCount;

        public override void Init(int pixWindowSize)
        {
            PixWinSize = pixWindowSize;
        }

        /// <summary>
        /// Gets random points within the usable area of the image
        ///  (= image with a border respecting the feature window).
        /// </summary>
        public override DataPointSet GetDataPoints(Image image)
        {
            var pi = MatrixCache.GetMatrixFrom(image.PixImage);
            var borderOffset = (int)Math.Ceiling(PixWinSize / 2.0);

            var result = new DataPoint[SampleCount];
            for (int i = 0; i < SampleCount; i++)
            {
                var x = Algo.Rand.Next(borderOffset, (int)pi.SX - borderOffset);
                var y = Algo.Rand.Next(borderOffset, (int)pi.SY - borderOffset);
                result[i] = new DataPoint(image, x, y);
            }

            return new DataPointSet(result, 1.0);
        }

        public override DataPointSet GetDataPoints(LabeledImage[] labeledImages)
        {
            throw new NotImplementedException();
        }

    }

    #endregion
}
