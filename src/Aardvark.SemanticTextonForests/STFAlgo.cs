using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Aardvark.Base;
using Newtonsoft.Json;

namespace Aardvark.SemanticTextonForests
{
    #region Training

    /// <summary>
    /// Static class which contains several extension functions to train Forests and Trees.
    /// </summary>
    public static class Algo
    {
        /// <summary>
        /// Random Number Generator
        /// </summary>
        public static Random Rand = new Random();

        private static int NodeProgressCounter = 0; //progress report

        /// <summary>
        /// The counter variable to determine a Node's global Index.
        /// </summary>
        private static int NodeIndexCounter = 0;
        /// <summary>
        /// The counter variable to determine a Tree's Forest Index.
        /// </summary>
        public static int TreeCounter = 0;

        /// <summary>
        /// Trains an empty and initialized Forest on a set of labeled Training Images.
        /// </summary>
        /// <param name="forest">The empty Forest to be trained.</param>
        /// <param name="trainingImages">The set of labeled Training Images to train this Forest with.</param>
        /// <param name="parameters">Parameters Object.</param>
        public static void Train(this Forest forest, LabeledImage[] trainingImages, TrainingParams parameters)
        {
            NodeIndexCounter = -1;

            Report.BeginTimed(0, "Training Forest of " + forest.Trees.Length + " trees with " + trainingImages.Length + " images.");

            TreeCounter = 0;

            Parallel.ForEach(forest.Trees, tree =>
            //foreach (var tree in forest.Trees)
            {
                //get a random subset of the actual training set.
                var currentSubset = trainingImages.GetRandomSubset(parameters.ImageSubsetCount);

                Report.BeginTimed(1, "Training tree " + (tree.Index + 1) + " of " + forest.Trees.Length + ".");

                //train the tree with the subset.
                tree.Train(currentSubset, parameters);

                Report.Line(2, "Finished training tree with " + NodeProgressCounter + " nodes.");

                Report.End(1);
            }
            );

            forest.NumNodes = forest.Trees.Sum(x => x.NumNodes);

            Report.End(0);
        }

        /// <summary>
        /// Trains an empty Tree with a given set of labeled Training Images.
        /// </summary>
        /// <param name="tree">The Tree to be trained.</param>
        /// <param name="trainingImages">The set of labeled Images used for training.</param>
        /// <param name="parameters">Parameters Object.</param>
        private static void Train(this Tree tree, LabeledImage[] trainingImages, TrainingParams parameters)
        {
            var nodeCounterObject = new NodeCountObject();

            //get a new Sampling Provider for this Tree
            tree.SamplingProvider = parameters.SamplingProviderFactory.GetNewProvider();
            //extract Data Points from the training Images using the Sampling Provider
            var baseDPS = tree.SamplingProvider.GetDataPoints(trainingImages);
            var baseClassDist = new LabelDistribution(parameters.Labels.ToArray(), baseDPS, parameters);

            //recursively train the tree starting from the Root
            tree.Root.TrainRecursive(null, baseDPS, parameters, 0, baseClassDist, nodeCounterObject);
            tree.NumNodes = nodeCounterObject.Counter;

            NodeProgressCounter = nodeCounterObject.Counter;
        }

        /// <summary>
        /// Recursively trains a binary sub-tree of an input Node given an input labeled Data Point Set.
        /// </summary>
        /// <param name="node">Node for which the sub-tree is to be trained. This function is called recursively for both its children. </param>
        /// <param name="parent">Input Node's Parent Node. </param>
        /// <param name="currentData">Input labeled Data Point Set.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <param name="depth">Current distance from the Root.</param>
        /// <param name="currentLabelDist">Label Distribution corresponding to the current Data Point Set.</param>
        /// <param name="currentNodeCounter">Global Counter Object.</param>
        private static void TrainRecursive(this Node node, Node parent, DataPointSet currentData, TrainingParams parameters, int depth, LabelDistribution currentLabelDist, NodeCountObject currentNodeCounter)
        {
            currentNodeCounter.Increment();

            node.GlobalIndex = Interlocked.Increment(ref NodeIndexCounter);

            //create a decider object and train it on the incoming data
            node.Decider = new Decider();

            node.ClassDistribution = currentLabelDist;

            //get a new feature provider for this node
            node.Decider.FeatureProvider = parameters.FeatureProviderFactory.GetNewProvider();
            node.DistanceFromRoot = depth;
            int newdepth = depth + 1;

            DataPointSet leftRemaining;
            DataPointSet rightRemaining;
            LabelDistribution leftClassDist;
            LabelDistribution rightClassDist;

            //training step: the decider finds the best split threshold for the current data
            var trainingResult = node.Decider.InitializeDecision(currentData, currentLabelDist, parameters, out leftRemaining, out rightRemaining, out leftClassDist, out rightClassDist);

            bool passthroughDeactivated = (!parameters.ForcePassthrough && trainingResult == DeciderTrainingResult.PassThrough);

            if (trainingResult == DeciderTrainingResult.Leaf //node is a leaf (empty)
                || depth >= parameters.MaxTreeDepth - 1        //node is at max level
                || passthroughDeactivated)                   //this node should pass through (information gain too small) but passthrough mode is deactivated
            //-> leaf
            {
                Report.Line(3, "->LEAF remaining dp=" + currentData.Count + "; depth=" + depth);
                node.IsLeaf = true;
                return;
            }

            if (trainingResult == DeciderTrainingResult.PassThrough) //too small information gain and not at max level -> create pass through node (copy from parent)
            {
                if (parent == null)      //empty tree or empty training set
                {
                    node.IsLeaf = true;
                    return;
                }

                Report.Line(3, "PASS THROUGH NODE dp#=" + currentData.Count + "; depth=" + depth + " t=" + parent.Decider.DecisionThreshold);

                node.Decider.DecisionThreshold = parent.Decider.DecisionThreshold;
                node.ClassDistribution = parent.ClassDistribution;
                node.Decider.FeatureProvider = parent.Decider.FeatureProvider;
            }

            var rightNode = new Node();
            var leftNode = new Node();

            TrainRecursive(rightNode, node, rightRemaining, parameters, newdepth, rightClassDist, currentNodeCounter);
            TrainRecursive(leftNode, node, leftRemaining, parameters, newdepth, leftClassDist, currentNodeCounter);

            node.RightChild = rightNode;
            node.LeftChild = leftNode;
        }

        /// <summary>
        /// Textonizes an Image using an input Forest. 
        /// </summary>
        /// <param name="image">Input Image. If Label is unknown, can be an arbitrary value. </param>
        /// <param name="forest">Input Forest.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>The Labeled Image with its corresponding Textonization. </returns>
        public static TextonizedLabeledImage Textonize(this LabeledImage image, Forest forest, TrainingParams parameters)
        {
            return new[] { image }.Textonize(forest, parameters)[0]; ;
        }

        /// <summary>
        /// Textonizes a set of Labeled Images using an input Forest.
        /// </summary>
        /// <param name="image">Input Image set. If Labels are unknown, can be an arbitrary value. </param>
        /// <param name="forest">Input Forest.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>The Labeled Image set with its corresponding Textonizations. </returns>
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

                var dist = forest.GetTextonization(img.Image, parameters);

                result[i] = new TextonizedLabeledImage(img, dist);

                Report.Line("{0} of {1} images textonized", Interlocked.Increment(ref count), images.Length);
            }
            );

            Report.End(0);

            return result;
        }

        /// <summary>
        /// Represents the outcomes of the training of one Node.
        /// </summary>
        public enum DeciderTrainingResult
        {
            Leaf,           //become a leaf, stop training
            PassThrough,    //copy the parent
            InnerNode       //continue training normally
        }
    }
    #endregion

    #region Helper Functions

    /// <summary>
    /// Simple counter object.
    /// </summary>
    public class NodeCountObject
    {
        public int Counter = 0;
        public void Increment()
        {
            Counter++;
        }
    }

    /// <summary>
    /// A static class containing some helper extension functions. These functions include some file I/O and string formatting. 
    /// </summary>
    public static class HelperFunctions
    {
        /// <summary>
        /// Splits a set of labeled Images in two halves. Each image of the input set has a 50% chance of going into either set.
        /// Can be used for splitting data into training/test sets.
        /// </summary>
        /// <param name="images">Input image set.</param>
        /// <param name="training">First output set.</param>
        /// <param name="test">Second output set.</param>
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

        /// <summary>
        /// Color Byte to double.
        /// </summary>
        /// <param name="par"></param>
        /// <returns></returns>
        public static double ToDouble(this byte par)
        {
            return ((double)par) * (1d / 255d);
        }

        /// <summary>
        /// Write a forest to JSON file.
        /// </summary>
        /// <param name="forest"></param>
        /// <param name="filename">Output file path.</param>
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

        /// <summary>
        /// Open a forest from JSON file.
        /// </summary>
        /// <param name="filename">File path.</param>
        /// <returns>Forest if valid.</returns>
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

        /// <summary>
        /// Write Textonized images to JSON file. 
        /// </summary>
        /// <param name="images">Textonized image set.</param>
        /// <param name="filename">Output file path.</param>
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

        /// <summary>
        /// Open Textonized images from JSON file. The original images used must be at the same location if they are to be used for training again.
        /// </summary>
        /// <param name="filename">Input file path.</param>
        /// <returns>Set of Textonized Labeled Images.</returns>
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

        /// <summary>
        /// Creates a new Forest and writes it to JSON file.
        /// </summary>
        /// <param name="filename">Output file path.</param>
        /// <param name="trainingSet">Set of trainign images for the Forest.</param>
        /// <param name="parameters">Parameters Object.</param>
        public static void CreateNewForestAndSaveToFile(string filename, LabeledImage[] trainingSet, TrainingParams parameters)
        {
            var forest = new Forest(parameters.ForestName, parameters.TreesCount);
            Report.Line(2, "Creating new forest " + forest.Name + ".");
            forest.Train(trainingSet, parameters);

            Report.Line(2, "Saving forest " + forest.Name + " to file.");
            forest.WriteToFile(filename);
        }
        
        /// <summary>
        /// Creates a new Forest.
        /// </summary>
        /// <param name="trainingSet">Training set used for Forest.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>Newly created Forest.</returns>
        public static Forest CreateNewForest(LabeledImage[] trainingSet, TrainingParams parameters)
        {
            var forest = new Forest(parameters.ForestName, parameters.TreesCount);
            Report.Line(2, "Creating new forest " + forest.Name + ".");
            forest.Train(trainingSet, parameters);
            return forest;
        }
        
        /// <summary>
        /// Textonizes a set of Images and writes the result to JSON file.
        /// </summary>
        /// <param name="filename">Output file path.</param>
        /// <param name="forest">Input Forest.</param>
        /// <param name="imageSet">Input Image Set.</param>
        /// <param name="parameters">Parameters Object.</param>
        public static void CreateTextonizationAndSaveToFile(string filename, Forest forest, LabeledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.Textonize(forest, parameters);
            Report.Line(2, "Saving textonization to file.");
            texImgs.WriteToFile(filename);
        }

        /// <summary>
        /// Textonizes a set of Images.
        /// </summary>
        /// <param name="forest">Input Forest.</param>
        /// <param name="imageSet">Input Image Set.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>Textonized Images.</returns>
        public static TextonizedLabeledImage[] CreateTextonization(Forest forest, LabeledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.Textonize(forest, parameters);
            return texImgs;
        }

        /// <summary>
        /// String formatting -> Add spaces until the total length of the string = desired length.
        /// Used for column formatting.
        /// </summary>
        /// <param name="prevValue">Input String.</param>
        /// <param name="number">Desired Length of the String.</param>
        /// <returns>Longer String.</returns>
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

        /// <summary>
        /// Reads the files from the MSRC data set. This is a custom function and won't work in general.
        /// </summary>
        /// <param name="directoryPath"></param>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static LabeledImage[] GetMsrcImagesFromDirectory(string directoryPath, TrainingParams parameters)
        {
            string[] picFiles = Directory.GetFiles(directoryPath);
            var result = new LabeledImage[picFiles.Length];
            for (int i = 0; i < picFiles.Length; i++)
            {
                var s = picFiles[i];
                string currentFilename = Path.GetFileNameWithoutExtension(s);
                string[] filenameSplit = currentFilename.Split('_');
                int fileLabel = Convert.ToInt32(filenameSplit[0]);
                Label currentLabel = parameters.Labels.First(x => x.Index == fileLabel - 1);
                result[i] = new LabeledImage(s, currentLabel);
            }
            return result;
        }

        /// <summary>
        /// Reads the files from a custom data set. This is a custom function and won't work in general.
        /// </summary>
        /// <param name="directoryPath"></param>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public static LabeledImage[] GetTDatasetFromDirectory(string directoryPath, TrainingParams parameters)
        {
            string nokpath = Path.Combine(directoryPath, "NOK");
            string okpath = Path.Combine(directoryPath, "OK");
            string[] nokFiles = Directory.GetFiles(nokpath);
            string[] okFiles = Directory.GetFiles(okpath);

            var result = new LabeledImage[okFiles.Length + nokFiles.Length];

            for (int i = 0; i < nokFiles.Length; i++)
            {
                var s = nokFiles[i];
                result[i] = new LabeledImage(s, parameters.Labels[0]);
            }

            for (int i = 0; i < okFiles.Length; i++)
            {
                var s = okFiles[i];
                result[nokFiles.Length + i] = new LabeledImage(s, parameters.Labels[1]);
            }

            return result;
        }
    }

    /// <summary>
    /// Thread-safe Matrix Cache to minimize matrix reading operations.
    /// </summary>
    internal static class MatrixCache
    {
        private static ThreadLocal<Dictionary<PixImage<byte>, Matrix<byte, C3b>>> s_cache =
            new ThreadLocal<Dictionary<PixImage<byte>, Matrix<byte, C3b>>>(() => new Dictionary<PixImage<byte>, Matrix<byte, C3b>>());

        /// <summary>
        /// Efficiently gets the underlying byte matrix of a PixImage.
        /// </summary>
        /// <param name="image">Input PixImage.</param>
        /// <returns>The PixImage's byte matrix.</returns>
        public static Matrix<byte, C3b> GetMatrixFrom(PixImage<byte> image)
        {
            Matrix<byte, C3b> result;
            if (s_cache.Value.TryGetValue(image, out result)) return result;

            result = image.GetMatrix<C3b>();
            s_cache.Value[image] = result;
            return result;
        }
    }

    #endregion

}
