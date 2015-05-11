using Aardvark.Base;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;
using Newtonsoft.Json;
using System.Threading;

namespace ScratchAttila
{
    #region STF training

    public static class STFAlgo
    {
        public static Random rand = new Random();
        public static int treeCounter = 0;        //progress counter
        public static int nodeProgressCounter = 0;               //progress report

        public static int nodeIndexCounter = 0;          //the counter variable to determine a node's global index


        public static void train(this STForest forest, STLabeledImage[] trainingImages, TrainingParams parameters)
        {
            nodeIndexCounter = -1;

            Report.BeginTimed(0, "Training Forest of " + forest.SemanticTextons.Length + " trees with " + trainingImages.Length + " images.");

            treeCounter = 0;

            Parallel.ForEach(forest.SemanticTextons, tree =>
            //foreach (var tree in forest.SemanticTextons)
            {
                STLabeledImage[] currentSubset = trainingImages.GetRandomSubset(parameters.ImageSubsetCount);

                Report.BeginTimed(1, "Training tree " + (tree.Index + 1) + " of " + forest.SemanticTextons.Length + ".");

                tree.train(currentSubset, parameters);

                Report.Line(2, "Finished training tree with " + nodeProgressCounter + " nodes.");

                Report.End(1);
            }
            );

            forest.numNodes = forest.SemanticTextons.Sum(x => x.NumNodes);

            Report.End(0);
        }

        public static void train(this SemanticTexton tree, STLabeledImage[] trainingImages, TrainingParams parameters)
        {
            var nodeCounterObject = new NodeCountObject();
            var provider = parameters.SamplingProviderFactory.getNewProvider();
            var baseDPS = provider.getDataPoints(trainingImages);
            var baseClassDist = new ClassDistribution(GlobalParams.Labels, baseDPS);

            tree.Root.trainRecursive(null, baseDPS, parameters, 0, baseClassDist, nodeCounterObject);
            tree.NumNodes = nodeCounterObject.counter;

            nodeProgressCounter = nodeCounterObject.counter;
        }

        public static void trainRecursive(this STNode node, STNode parent, DataPointSet currentData, TrainingParams parameters, int depth, ClassDistribution currentClassDist, NodeCountObject currentNodeCounter)
        {
            currentNodeCounter.increment();

            node.GlobalIndex = Interlocked.Increment(ref nodeIndexCounter);

            //create a decider object and train it on the incoming data
            node.Decider = new Decider();

            //only one sampling rule per tree (currently only one exists, regular sampling)
            if(parent == null)
            {
                node.Decider.SamplingProvider = parameters.SamplingProviderFactory.getNewProvider();
            }
            else
            {
                node.Decider.SamplingProvider = parent.Decider.SamplingProvider;
            }

            node.ClassDistribution = currentClassDist;

            //get a new feature provider for this node
            node.Decider.FeatureProvider = parameters.FeatureProviderFactory.getNewProvider();
            node.DistanceFromRoot = depth;
            int newdepth = depth + 1;

            DataPointSet leftRemaining;
            DataPointSet rightRemaining;
            ClassDistribution leftClassDist;
            ClassDistribution rightClassDist;

            //training step: the decider finds the best split threshold for the current data
            DeciderTrainingResult trainingResult = node.Decider.InitializeDecision(currentData, currentClassDist, parameters, out leftRemaining, out rightRemaining, out leftClassDist, out rightClassDist);

            bool passthroughDeactivated = (!parameters.ForcePassthrough && trainingResult == DeciderTrainingResult.PassThrough);

            if (trainingResult == DeciderTrainingResult.Leaf //node is a leaf (empty)
                || depth >= parameters.MaxTreeDepth - 1        //node is at max level
                || passthroughDeactivated)                   //this node should pass through (information gain too small) but passthrough mode is deactivated
            //-> leaf
            {
                Report.Line(3, "->LEAF remaining dp=" + currentData.DPSet.Length + "; depth=" + depth);
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

                Report.Line(3, "PASS THROUGH NODE dp#=" + currentData.DPSet.Length + "; depth=" + depth + " t=" + parent.Decider.DecisionThreshold);

                node.Decider.DecisionThreshold = parent.Decider.DecisionThreshold;
                node.ClassDistribution = parent.ClassDistribution;
                node.Decider.FeatureProvider = parent.Decider.FeatureProvider;
                node.Decider.SamplingProvider = parent.Decider.SamplingProvider;
            }

            var rightNode = new STNode();
            var leftNode = new STNode();

            trainRecursive(rightNode, node, rightRemaining, parameters, newdepth, rightClassDist, currentNodeCounter);
            trainRecursive(leftNode, node, leftRemaining, parameters, newdepth, leftClassDist, currentNodeCounter);

            node.RightChild = rightNode;
            node.LeftChild = leftNode;
        }

        public static STTextonizedLabelledImage textonize(this STLabeledImage image, STForest forest, TrainingParams parameters)
        {
            return new[] { image }.textonize(forest, parameters)[0]; ;
        }

        public static STTextonizedLabelledImage[] textonize(this STLabeledImage[] images, STForest forest, TrainingParams parameters)
        {
            var result = new STTextonizedLabelledImage[images.Length];

            Report.BeginTimed(0, "Textonizing " + images.Length + " images.");

            int count = 0;
            Parallel.For(0, images.Length, i =>
            //for (int i = 0; i < images.Length; i++)
            {
                //Report.Progress(0, (double)i / (double)images.Length);
                var img = images[i];

                var dist = forest.getTextonRepresentation(img, parameters);

                result[i] = new STTextonizedLabelledImage(img, dist);

                Report.Line("{0} of {1} images textonized", Interlocked.Increment(ref count), images.Length);
            }
            );

            Report.End(0);

            return result;
        }

        public static STTextonizedLabelledImage[] normalizeInvDocFreq(this STTextonizedLabelledImage[] images)
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
        public int counter = 0;
        public void increment()
        {
            counter++;
        }
    }

    public static class HelperFunctions     //temporary helper functions
    {
        public static void splitIntoSets(this STLabeledImage[] images, out STLabeledImage[] training, out STLabeledImage[] test)
        {
            ////50/50 split
            var tro = new List<STLabeledImage>();
            var teo = new List<STLabeledImage>();

            foreach (var img in images)
            {
                var rn = STFAlgo.rand.NextDouble();
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

        public static double toDouble(this byte par)
        {
            return ((double)par) * (1d / 255d);
        }

        public static void writeToFile(this STForest forest, string filename)
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

        public static STForest readForestFromFile(string filename)
        {
            Report.Line(2, "Reading forest from file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
            };

            var parsed = JsonConvert.DeserializeObject<STForest>(File.ReadAllText(filename), settings);
            return parsed;
        }

        public static void writeToFile(this STTextonizedLabelledImage[] images, string filename)
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

        public static STTextonizedLabelledImage[] readTextonizedImagesFromFile(string filename)
        {
            Report.Line(2, "Reading textonized image set from file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
            };

            var parsed = JsonConvert.DeserializeObject<STTextonizedLabelledImage[]>(File.ReadAllText(filename), settings);
            return parsed;
        }

        //creates a forest and saves it to file
        public static void createNewForestAndSaveToFile(string filename, STLabeledImage[] trainingSet, TrainingParams parameters)
        {
            STForest forest = new STForest(parameters.ForestName);
            Report.Line(2, "Creating new forest " + forest.name + ".");
            forest.InitializeEmptyForest(parameters.TreesCount);
            forest.train(trainingSet, parameters);

            Report.Line(2, "Saving forest " + forest.name + " to file.");
            forest.writeToFile(filename);
        }

        //deprecated
        public static STForest createNewForest(STLabeledImage[] trainingSet, TrainingParams parameters)
        {
            STForest forest = new STForest(parameters.ForestName);
            Report.Line(2, "Creating new forest " + forest.name + ".");
            forest.InitializeEmptyForest(parameters.TreesCount);
            forest.train(trainingSet, parameters);

            return forest;
        }

        //textonizes images and saves the array to file
        public static void createTextonizationAndSaveToFile(string filename, STForest forest, STLabeledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.textonize(forest, parameters);


            Report.Line(2, "Saving textonization to file.");
            texImgs.writeToFile(filename);
        }

        public static STTextonizedLabelledImage[] createTextonization(STForest forest, STLabeledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.textonize(forest, parameters);

            return texImgs;
        }

        //string formatting -> add spaces until the total length of the string = number
        public static string spaces(string prevValue, int number)
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

        //reads all images from a directory and their labels from filename
        public static STLabeledImage[] GetLabeledImagesFromDirectory(string directoryPath)
        {
            string[] picFiles = Directory.GetFiles(directoryPath);

            var result = new STLabeledImage[picFiles.Length];

            for (int i = 0; i < picFiles.Length; i++)
            {
                var s = picFiles[i];
                string currentFilename = Path.GetFileNameWithoutExtension(s);
                string[] filenameSplit = currentFilename.Split('_');
                int fileLabel = Convert.ToInt32(filenameSplit[0]);
                ClassLabel currentLabel = GlobalParams.Labels.First(x => x.Index == fileLabel - 1);
                result[i] = new STLabeledImage(s) { ClassLabel = currentLabel };
            }


            return result;
        }

        public static STLabeledImage[] getTDatasetFromDirectory(string directoryPath)
        {
            string nokpath = Path.Combine(directoryPath, "NOK");
            string okpath = Path.Combine(directoryPath, "OK");
            string[] nokFiles = Directory.GetFiles(nokpath);
            string[] okFiles = Directory.GetFiles(okpath);

            var result = new STLabeledImage[okFiles.Length + nokFiles.Length];

            for (int i = 0; i < nokFiles.Length; i++)
            {
                var s = nokFiles[i];

                result[i] = new STLabeledImage(s) { ClassLabel = GlobalParams.Labels[0] };
            }

            for (int i = 0; i < okFiles.Length; i++)
            {
                var s = okFiles[i];

                result[nokFiles.Length + i] = new STLabeledImage(s) { ClassLabel = GlobalParams.Labels[1] };
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
        IFeatureProvider currentProvider;
        int pixWinSize;
        FeatureType currentChoice;

        public void selectProvider(FeatureType featureType, int pixelWindowSize)
        {
            currentChoice = featureType;
            pixWinSize = pixelWindowSize;
        }

        public IFeatureProvider getNewProvider()
        {
            switch (currentChoice)
            {
                case FeatureType.RandomPixelValue:
                    currentProvider = new ValueOfPixelFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.RandomTwoPixelSum:
                    currentProvider = new PixelSumFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.RandomTwoPixelAbsDiff:
                    currentProvider = new AbsDiffOfPixelFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.RandomTwoPixelDifference:
                    currentProvider = new PixelDifferenceFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.SelectRandom:      //select one of the three providers at random - equal chance
                    var choice = STFAlgo.rand.Next(4);
                    switch(choice)
                    {
                        case 0:
                            currentProvider = new ValueOfPixelFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        case 1:
                            currentProvider = new PixelSumFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        case 2:
                            currentProvider = new AbsDiffOfPixelFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        case 3:
                            currentProvider = new PixelDifferenceFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        default:
                            return null;
                    }
                    break;
                default:
                    currentProvider = new ValueOfPixelFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;

            }
            return currentProvider;
        }
    }

    public class PixelSumFeatureProvider : IFeatureProvider
    {
        public int fX;
        public int fY;
        public int sX;
        public int sY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(fX, fY); }
            set { fX = value.X; fY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(sX, sY); }
            set { sX = value.X; sY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int firstY = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondY = STFAlgo.rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().toDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().toDouble();

            var op = (sample1 + sample2) / 2.0; //divide by two for normalization

            result.Value = op;

            return result;
        }
    }

    public class PixelDifferenceFeatureProvider : IFeatureProvider
    {
        public int fX;
        public int fY;
        public int sX;
        public int sY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(fX, fY); }
            set { fX = value.X; fY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(sX, sY); }
            set { sX = value.X; sY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int firstY = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondY = STFAlgo.rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().toDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().toDouble();

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
        public V2i pixelOffset
        {
            get { return new V2i(X, Y); }
            set { X = value.X; Y = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {

            int half = (int)(pixelWindowSize / 2);
            int x = STFAlgo.rand.Next(pixelWindowSize) - half;
            int y = STFAlgo.rand.Next(pixelWindowSize) - half;

            pixelOffset = new V2i(x, y);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample = pi[point.PixelCoords + pixelOffset].ToGrayByte().toDouble();

            result.Value = sample;

            return result;
        }
    }

    public class AbsDiffOfPixelFeatureProvider : IFeatureProvider
    {
        public int fX;
        public int fY;
        public int sX;
        public int sY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(fX, fY); }
            set { fX = value.X; fY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(sX, sY); }
            set { sX = value.X; sY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int firstY = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondY = STFAlgo.rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().toDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().toDouble();

            var op = Math.Abs(sample2 - sample1);

            result.Value = op;

            return result;
        }
    }


    public class SamplingProviderFactory
    {
        ISamplingProvider currentProvider;
        int pixelWindowSize;
        SamplingType samplingType;
        int randomSampleCount = 0;

        public void selectProvider(SamplingType samplingType, int pixelWindowSize)
        {
            this.pixelWindowSize = pixelWindowSize;
            this.samplingType = samplingType;
        }

        public void selectProvider(SamplingType samplingType, int pixelWindowSize, int randomSampleCount)
        {
            this.pixelWindowSize = pixelWindowSize;
            this.samplingType = samplingType;
            this.randomSampleCount = randomSampleCount;
        }

        public ISamplingProvider getNewProvider()
        {

            switch (samplingType)
            {
                case SamplingType.RegularGrid:
                    currentProvider = new RegularGridSamplingProvider();
                    currentProvider.init(pixelWindowSize);
                    break;
                case SamplingType.RandomPoints:
                    var result = new RandomPointSamplingProvider();
                    result.init(pixelWindowSize);
                    result.SampleCount = this.randomSampleCount;
                    currentProvider = result;
                    break;
                default:
                    currentProvider = new RegularGridSamplingProvider();
                    currentProvider.init(pixelWindowSize);
                    break;
            }

            return currentProvider;
        }
    }

    public class RegularGridSamplingProvider : ISamplingProvider
    {
        public int pixWinSize;

        public override void init(int pixWindowSize)
        {
            pixWinSize = pixWindowSize;
        }

        public override DataPointSet getDataPoints(STImagePatch image)
        {
            //currently, this gets a regular grid starting from the top left and continuing as long as there are pixels left.
            var pi = image.PixImage.GetMatrix<C3b>();

            List<DataPoint> result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling((double)pixWinSize / 2.0f); //ceiling cuts away too much in most cases

            int pointCounter = 0;

            for (int x = borderOffset; x < pi.SX - borderOffset; x = x + pixWinSize)
            {
                for (int y = borderOffset; y < pi.SY - borderOffset; y = y + pixWinSize)
                {
                    var newDP = new DataPoint()
                    {
                        PixelCoords = new V2i(x, y),
                        Image = image
                    };
                    result.Add(newDP);
                    pointCounter = pointCounter + 1;
                }
            }

            var bias = (double)1 / (double)pointCounter;     //weigh the sample points by the image's size (larger image = lower weight)

            var resDPS = new DataPointSet();
            resDPS.DPSet = result.ToArray();
            resDPS.SetWeight = bias;

            return resDPS;
        }

        public override DataPointSet getDataPoints(STLabeledImage[] images)
        {
            var result = new DataPointSet();
            
            foreach(var img in images)
            {
                var currentDPS = getDataPoints(img);
                foreach(var dp in currentDPS.DPSet)
                {
                    dp.label = img.ClassLabel.Index;
                }
                result = result + currentDPS;
            }
            

            return result;
        }

    }

    public class RandomPointSamplingProvider : ISamplingProvider
    {
        public int pixWinSize;
        public int SampleCount;

        public override void init(int pixWindowSize)
        {
            pixWinSize = pixWindowSize;
        }

        public override DataPointSet getDataPoints(STImagePatch image)
        {
            //Gets random points within the usable area of the image (= image with a border respecting the feature window)
            var pi = image.PixImage.GetMatrix<C3b>();

            List<DataPoint> result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling((double)pixWinSize / 2.0d);

            for (int i = 0; i < SampleCount; i++)
            {
                var x = STFAlgo.rand.Next(borderOffset, (int)pi.SX - borderOffset);
                var y = STFAlgo.rand.Next(borderOffset, (int)pi.SY - borderOffset);

                var newDP = new DataPoint()
                {
                    PixelCoords = new V2i(x, y),
                    Image = image
                };
                result.Add(newDP);

            }


            var resDPS = new DataPointSet();
            resDPS.DPSet = result.ToArray();
            resDPS.SetWeight = 1.0d;

            return resDPS;
        }

        public override DataPointSet getDataPoints(STLabeledImage[] labelledImages)
        {
            throw new NotImplementedException();
        }

    }

    #endregion
}
