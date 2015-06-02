using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Aardvark.Base;
using System.Threading;
using static Aardvark.SemanticTextonForests.Algo;

/// <summary>
/// This File includes the necessary data structures and methods for the image segmentation method using the "Level 2" Forest.
/// </summary>
namespace Aardvark.SemanticTextonForests
{




    public class SegmentationForest
    {
        /// <summary>
        /// The collection of Trees belonging to this Forest.
        /// </summary>
        public SegmentationTree[] Trees;
        /// <summary>
        /// Friendly name.
        /// </summary>
        public string Name { get; }
        /// <summary>
        /// The total number of Trees.
        /// </summary>
        public int NumTrees { get; }
        /// <summary>
        /// The total number of nodes of all Trees.
        /// </summary>
        public int NumNodes = -1;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public SegmentationForest() { }

        public SegmentationForest(string name, int numberOfTrees)
        {
            Name = name;
            NumTrees = numberOfTrees;

            InitializeEmptyForest();
        }

        /// <summary>
        /// Initializes the forest with empty Trees.
        /// </summary>
        private void InitializeEmptyForest()
        {
            Trees = new SegmentationTree[NumTrees].SetByIndex(i => new SegmentationTree() { Index = i });
        }
    }

    public class SegmentationTree
    {
        /// <summary>
        /// This tree's root node. Every traversal starts here.
        /// </summary>
        public SegmentationNode Root;
        /// <summary>
        /// This tree's Forest index.
        /// </summary>
        public int Index = -1;
        /// <summary>
        /// The total count of nodes in this Tree.
        /// </summary>
        public int NumNodes = 0;
        /// <summary>
        /// The Sampling Provider used in this tree.
        /// </summary>
        public SegmentationSamplingProvider SamplingProvider;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public SegmentationTree()
        {
            Root = new SegmentationNode();
            Root.GlobalIndex = this.Index;
        }
    }

    public class SegmentationNode
    {
        public bool IsLeaf = false;
        public int DistanceFromRoot = 0;
        public SegmentationNode LeftChild;
        public SegmentationNode RightChild;
        /// <summary>
        /// The Decider associated with this node.
        /// </summary>
        public SegmentationDecider Decider;
        /// <summary>
        /// This node's global index in the forest.
        /// </summary>
        public int GlobalIndex = -1;

        /// <summary>
        /// The Label Distribution corresponding to the segmentation data points that reached this node - this information
        /// is in the points' label maps.
        /// </summary>
        public LabelDistribution LabelDistribution;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public SegmentationNode() { }
    }

    public class SegmentationDecider
    {
        /// <summary>
        /// The feature provider used in this Decider.
        /// </summary>
        public SegmentationFeatureProvider FeatureProvider;
        /// <summary>
        /// This Decider's trained decision threshold.
        /// </summary>
        public double DecisionThreshold;
        /// <summary>
        /// The certainty measure of the trained decision (= expected gain in information).
        /// </summary>
        public double Certainty;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public SegmentationDecider() { }

        /// <summary>
        /// Decides whether the data point's feature value corresponds to Left (less than) or Right (greater than).
        /// </summary>
        /// <param name="dataPoint">Input data point.</param>
        /// <returns>Left/Right Decision.</returns>
        public Decision Decide(SegmentationDataPoint dataPoint)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Trains this Decider on a training set. The Decider tries out many different thresholds to split the training set and chooses
        /// the one that maximizes the expected gain in information ("Certainty"). 
        /// </summary>
        /// <param name="currentDatapoints">Training set of labeled data points.</param>
        /// <param name="classDist">Class Distribution of training set.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <param name="leftRemaining">Output "Left" subset of the input set.</param>
        /// <param name="rightRemaining">Output "Right" subset of the input set.</param>
        /// <param name="leftClassDist">Class Distribution corresponding to the "Left" output.</param>
        /// <param name="rightClassDist">Class Distribution corresponding to the "Right" output.</param>
        /// <returns></returns>
        public Algo.DeciderTrainingResult InitializeDecision(
            List<SegmentationDataPoint> currentDatapoints, LabelDistribution classDist, int thresholdCandidateNumber,
            out List<SegmentationDataPoint> leftRemaining, out List<SegmentationDataPoint> rightRemaining,
            out LabelDistribution leftClassDist, out LabelDistribution rightClassDist
            )
        {

            //find the best threshold
            var bestThreshold = -1.0;
            var bestScore = double.MinValue;
            var bestLeftSet = new List<SegmentationDataPoint>();
            var bestRightSet = new List<SegmentationDataPoint>();
            LabelDistribution bestLeftClassDist = null;
            LabelDistribution bestRightClassDist = null;

            bool inputIsEmpty = currentDatapoints.Count == 0; //there is no image, no split is possible -> leaf
            bool inputIsOne = currentDatapoints.Count == 1;   //there is exactly one image, no split is possible -> leaf (or passthrough)

            if (!inputIsEmpty && !inputIsOne)
            {
                //generate random candidates for threshold
                var threshCandidates = new double[thresholdCandidateNumber];
                for (int i = 0; i < threshCandidates.Length; i++)
                {
                    threshCandidates[i] = Algo.Rand.NextDouble();
                }

                //for each candidate, try the split and calculate its expected gain in information
                //scale the candidates to the current problem
                for(var i=0; i<threshCandidates.Length; i++)
                {
                    var curThresh = threshCandidates[i];

                    var currentLeftSet = new List<SegmentationDataPoint>();
                    var currentRightSet = new List<SegmentationDataPoint>();
                    LabelDistribution currentLeftClassDist = null;
                    LabelDistribution currentRightClassDist = null;

                    SplitDatasetWithThreshold(currentDatapoints, ref curThresh, out currentLeftSet, out currentRightSet, out currentLeftClassDist, out currentRightClassDist);
                    double leftEntr = CalcEntropy(currentLeftClassDist);
                    double rightEntr = CalcEntropy(currentRightClassDist);

                    var leftsum = currentLeftClassDist.GetLabelDistSum();
                    var rightsum = currentRightClassDist.GetLabelDistSum();

                    double leftWeight = (-1.0d) * ((leftsum == 0) ? (float.MaxValue) : (leftsum)) / classDist.GetLabelDistSum();
                    double rightWeight = (-1.0d) * ((rightsum == 0) ? (float.MaxValue) : (rightsum)) / classDist.GetLabelDistSum();
                    double score = leftWeight * leftEntr + rightWeight * rightEntr;

                    if (score > bestScore) //new best threshold found
                    {
                        bestScore = score;
                        bestThreshold = curThresh;

                        bestLeftSet = currentLeftSet;
                        bestRightSet = currentRightSet;
                        bestLeftClassDist = currentLeftClassDist;
                        bestRightClassDist = currentRightClassDist;
                    }
                }
            }

            //generate output

            Certainty = bestScore;

            bool isLeaf = (bestScore > -0.01) || inputIsEmpty;   //no images reached this node or not enough information gain => leaf

            if (isLeaf)
            {
                leftRemaining = null;
                rightRemaining = null;
                leftClassDist = null;
                rightClassDist = null;
                return Algo.DeciderTrainingResult.Leaf;
            }

            if (!isLeaf)
            {
                Report.Line(3, $"NN t:{bestThreshold}(c={FeatureProvider.Channel}) s:{bestScore}; dp={currentDatapoints.Count} l/r={bestLeftSet.Count}/{bestRightSet.Count}");
            }

            this.DecisionThreshold = bestThreshold;
            leftRemaining = bestLeftSet;
            rightRemaining = bestRightSet;
            leftClassDist = bestLeftClassDist;
            rightClassDist = bestRightClassDist;

            return Algo.DeciderTrainingResult.InnerNode;
        }

        /// <summary>
        /// Splits up the dataset using a threshold.
        /// </summary>
        /// <param name="dps">Input data point set.</param>
        /// <param name="threshold">Decision threshold.</param>
        /// <param name="leftSet">Output Left subset.</param>
        /// <param name="rightSet">Output Right subset.</param>
        /// <param name="leftDist">Class Distribution belonging to Left output.</param>
        /// <param name="rightDist">Class Distribution belonging to Right output.</param>
        private void SplitDatasetWithThreshold(List<SegmentationDataPoint> dps, ref double threshold, out List<SegmentationDataPoint> leftSet, out List<SegmentationDataPoint> rightSet, out LabelDistribution leftDist, out LabelDistribution rightDist)
        {
            var leftList = new List<SegmentationDataPoint>();
            var rightList = new List<SegmentationDataPoint>();

            FeatureProvider.FindPopulatedChannel(dps);

            var maxValue = dps.Max(x =>
            {
                return FeatureProvider.GetFeature(x);
            });

            if(maxValue == 0)
            {
                Report.Line("Shouldn't happen");
            }

            //scale the threshold by the max value
            threshold *= maxValue;

            foreach (var dp in dps)
            {
                var feature = FeatureProvider.GetFeature(dp);

                if (feature < threshold)
                {
                    leftList.Add(dp);
                }
                else
                {
                    rightList.Add(dp);
                }

            }

            leftSet = new List<SegmentationDataPoint>(leftList);
            rightSet = new List<SegmentationDataPoint>(rightList);

            //leftDist = new LabelDistribution(leftList, dps[0].DistributionImage.numChannels);
            //rightDist = new LabelDistribution(rightList, dps[0].DistributionImage.numChannels);
            leftDist = LabelDistribution.GetSegmentationPrediction(leftList, dps[0].DistributionImage.numChannels);
            rightDist = LabelDistribution.GetSegmentationPrediction(rightList, dps[0].DistributionImage.numChannels);
        }

        /// <summary>
        /// Calculates the entropy of one Class Distribution. It is used for estimating the quality of a split.
        /// </summary>
        /// <param name="dist">Input Class Distribution.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>Entropy value of the input distribution.</returns>
        internal double CalcEntropy(LabelDistribution dist)
        {
            if (dist.GetLabelDistSum() == 0) return (float.MaxValue);

            //from http://en.wikipedia.org/wiki/ID3_algorithm

            double sum = 0;
            for (var c = 0; c < dist.Distribution.Length; c++)
            {
                var px = dist.GetLabelProbability(c);
                if (px == 0)
                {
                    continue;
                }
                var val = (px * Math.Log(px, 2));
                sum = sum + val;
            }
            sum = sum * (-1.0);

            if (Double.IsNaN(sum))
            {
                Report.Line("NaN value occured");
            }
            return sum;
        }
    }

    public class SegmentationFeatureProvider
    {
        public readonly int OffsetX, OffsetY;
        public int Channel;

        private int NumClasses;

        /// <summary>
        /// Initializes the Segmentation Feature Provider with a random offset vector and a random channel for sampling the 
        /// SegmentationDataPoint's pixel window.
        /// </summary>
        /// <param name="maximumOffsetX">The maximal value for the offset in x direction.</param>
        /// <param name="maximumOffsetY">The maximal value for the offset in y direction.</param>
        /// <param name="numClasses">The number of distinct labels (count of distribution map channels).</param>
        public SegmentationFeatureProvider(int maximumOffsetX, int maximumOffsetY, int numClasses)
        {
            var equalX = maximumOffsetX / 2;
            var equalY = maximumOffsetY / 2;
            OffsetX = Algo.Rand.Next(maximumOffsetX)-equalX;
            OffsetY = Algo.Rand.Next(maximumOffsetY)-equalY;
            Channel = Algo.Rand.Next(numClasses);
            NumClasses = numClasses;
        }

        /// <summary>
        /// if the channel is equal to 0, this feature provider is bad -> use the dataset to find
        /// a populated channel instead.
        /// </summary>
        public void FindPopulatedChannel(List<SegmentationDataPoint> dps)
        {
            if(dps.Count == 0)
            {
                return;
            }
            var test = dps.GetRandomSubset(1).First();
            for(int i=0; i<NumClasses; i++)
            {
                //is the current channel 0?
                if (test.DistributionImage.DistributionMap[test.X, test.Y, Channel] != 0)
                {
                    //no the current channel is not 0
                    //-> is the current channel only same values?
                    var testList = dps.GetRandomSubset(50);
                    bool same = true;
                    double last = GetFeature(testList[0]);
                    for(int j=1; j<testList.Count; j++)
                    {
                        var curf = GetFeature(testList[i]);
                        if(curf != last)
                        {
                            //no, at least one channel is not the same
                            same = false;
                            return;
                        }
                        same = true;
                    }
                    if(same)
                    {
                        Channel = (Channel + 1) % (NumClasses - 1);
                    }
                }
                else
                {
                    Channel = (Channel + 1) % (NumClasses - 1);
                }
            }
        }

        public double GetFeature(SegmentationDataPoint dp)
        {
            //get feature value from the resulting label distribution
            var sizeX = (int)dp.DistributionImage.DistributionMap.Size.X;
            var sizeY = (int)dp.DistributionImage.DistributionMap.Size.Y;

            var minX = dp.X + OffsetX;
            var minY = dp.Y + OffsetY;
            var maxX = dp.X + dp.SX + OffsetX;
            var maxY = dp.Y + dp.SY + OffsetY;

            //clamp values such that the window is still fully within the image
            if (outOfRange(minX, 0, sizeX - dp.SX)) clamp(ref minX, 0, sizeX - dp.SX);
            if (outOfRange(minY, 0, sizeY - dp.SY)) clamp(ref minY, 0, sizeY - dp.SY);
            if (outOfRange(maxX, dp.SX, sizeX)) clamp(ref maxX, dp.SX, sizeX);
            if (outOfRange(maxY, dp.SY, sizeY)) clamp(ref maxY, dp.SY, sizeY);

            var dist = dp.DistributionImage.GetWindowPrediction(minX, minY, maxX, maxY);

            return dist.Distribution[Channel];
        }

        internal bool outOfRange(int val, int min, int max)
        {
            return val < min || val > max;
        }

        internal void clamp(ref int val, int min, int max)
        {
            if (val < min)
            {
                val = min;
            }
            else if (val > max)
            {
                val = max;
            }
        }
    }

    public class SegmentationSamplingProvider
    {

        public SegmentationSamplingProvider()
        {

        }

        public List<SegmentationDataPoint> GetDataPoints(DistributionImage image, int pixWinSizeX, int pixWinSizeY)
        {
            var result = new List<SegmentationDataPoint>();

            int pointCounter = 0;

            for (int x = 0; x < image.DistributionMap.Size.X - pixWinSizeX; x += pixWinSizeX)
            {
                for (int y = 0; y < image.DistributionMap.Size.Y - pixWinSizeY; y += pixWinSizeY)
                {
                    var newDP = new SegmentationDataPoint(image, x, y, pixWinSizeX, pixWinSizeY);
                    result.Add(newDP);
                    pointCounter++;
                }
            }

            var bias = 1.0 / pointCounter;     //weigh the sample points by the image's size (larger image = lower weight)

            return result;
        }

        public List<SegmentationDataPoint> GetDataPoints(DistributionImage[] image, int pixWinSizeX, int pixWinSizeY)
        {
            var result = new List<SegmentationDataPoint>();

            foreach (var img in image)
            {
                result.AddRange(this.GetDataPoints(img, pixWinSizeX, pixWinSizeY));
            }

            return result;
        }

        /// <summary>
        /// Splits the images into 1/ratio many parts in both directions
        /// </summary>
        /// <param name="image"></param>
        /// <param name="ratio"></param>
        /// <returns></returns>
        public List<SegmentationDataPoint> GetDataPoints(DistributionImage[] image, double ratio)
        {
            var result = new List<SegmentationDataPoint>();

            foreach (var img in image)
            {
                var segmentsX = (int)Math.Floor(img.DistributionMap.Size.X * ratio);
                var segmentsY = (int)Math.Floor(img.DistributionMap.Size.Y * ratio);

                result.AddRange(this.GetDataPoints(img, segmentsX, segmentsY));
            }

            return result;
        }
    }

    public class SegmentationDataPoint
    {
        public readonly DistributionImage DistributionImage;
        public readonly int X;
        public readonly int Y;
        public readonly int SX, SY;


        public SegmentationDataPoint(DistributionImage pi, int x, int y, int sx, int sy)
        {
            if (x < 0 || y < 0 || x >= pi.DistributionMap.Size.X || y >= pi.DistributionMap.Size.Y) throw new IndexOutOfRangeException();

            DistributionImage = pi;
            X = x; Y = y;
            SX = sx; SY = sy;
        }
    }

    public static class SegmentationAlgo
    {
        private static int NodeIndexCounter = 0;
        public static int TreeCounter = 0;
        private static int NodeProgressCounter = 0;

        public static void Train(this SegmentationForest forest, DistributionImage[] trainingImages, SegmentationParameters parameters)
        {
            NodeIndexCounter = -1;

            Report.BeginTimed(0, $"Training SegmentationForest with {trainingImages.Length} DistributionImages.");

            TreeCounter = 0;

            //Parallel.ForEach(forest.Trees, tree =>
            foreach (var tree in forest.Trees)
            {
                //get a random subset of the actual training set.
                var currentSubset = trainingImages.GetRandomSubset(parameters.TrainingSubsetPerTree);

                Report.BeginTimed(1, $"Training tree {tree.Index + 1} of {forest.Trees.Length}.");

                //train the tree with the subset.
                tree.Train(currentSubset, parameters);

                Report.Line(2, "Finished training tree with " + NodeProgressCounter + " nodes.");

                Report.End(1);
            }
            //);

            forest.NumNodes = forest.Trees.Sum(x => x.NumNodes);

            Report.End(0);
        }

        private static void Train(this SegmentationTree tree, DistributionImage[] trainingImages, SegmentationParameters parameters)
        {
            var nodeCounterObject = new NodeCountObject();

            //get a new Sampling Provider for this Tree
            tree.SamplingProvider = new SegmentationSamplingProvider();
            //extract Data Points from the training Images using the Sampling Provider
            var baseDPS = tree.SamplingProvider.GetDataPoints(trainingImages, parameters.SegmentatioSplitRatio);
            //var baseClassDist = new LabelDistribution(baseDPS, baseDPS[0].DistributionImage.numChannels);
            var baseClassDist = LabelDistribution.GetSegmentationPrediction(baseDPS, baseDPS[0].DistributionImage.numChannels);

            Report.Line(2, $"Tree Training datapoint set size: {baseDPS.Count}");

            //recursively train the tree starting from the Root
            tree.Root.TrainRecursive(null, baseDPS, parameters, 0, baseClassDist, nodeCounterObject);
            tree.NumNodes = nodeCounterObject.Counter;

            NodeProgressCounter = nodeCounterObject.Counter;
        }

        private static void TrainRecursive(this SegmentationNode node, SegmentationNode parent, List<SegmentationDataPoint> currentData,
            SegmentationParameters parameters, int depth, LabelDistribution currentLabelDist, NodeCountObject currentNodeCounter)
        {
            currentNodeCounter.Increment();

            node.GlobalIndex = Interlocked.Increment(ref NodeIndexCounter);

            //create a decider object and train it on the incoming data
            node.Decider = new SegmentationDecider();

            node.LabelDistribution = currentLabelDist;

            var numChans = (currentData.Count == 0) ? 1 : currentData[0].DistributionImage.numChannels;

            //get a new feature provider for this node
            node.Decider.FeatureProvider = new SegmentationFeatureProvider(parameters.MaximumFeatureOffsetX, parameters.MaximumFeatureOffsetY,
                numChans);
            node.Decider.FeatureProvider.FindPopulatedChannel(currentData);
            node.DistanceFromRoot = depth;
            int newdepth = depth + 1;

            List<SegmentationDataPoint> leftRemaining;
            List<SegmentationDataPoint> rightRemaining;
            LabelDistribution leftClassDist;
            LabelDistribution rightClassDist;

            //training step: the decider finds the best split threshold for the current data
            var trainingResult = node.Decider.InitializeDecision(currentData, currentLabelDist, parameters.ThresholdCandidateNumber,
                out leftRemaining, out rightRemaining, out leftClassDist, out rightClassDist);

            node.LabelDistribution.Normalize();

            if (trainingResult == DeciderTrainingResult.Leaf   //node is a leaf (empty)
                || depth >= parameters.MaxTreeDepth - 1)        //node is at max level
                                                                //-> leaf
            {
                Report.Line(3, $"->LEAF remaining dp={currentData.Count}; depth={depth}; ftc={node.Decider.FeatureProvider.Channel}");
                node.IsLeaf = true;
                return;
            }

            var rightNode = new SegmentationNode();
            var leftNode = new SegmentationNode();

            TrainRecursive(rightNode, node, rightRemaining, parameters, newdepth, rightClassDist, currentNodeCounter);
            TrainRecursive(leftNode, node, leftRemaining, parameters, newdepth, leftClassDist, currentNodeCounter);

            node.RightChild = rightNode;
            node.LeftChild = leftNode;
        }
    }

    public class SegmentationParameters
    {
        /// <summary>
        /// Create default values.
        /// </summary>
        public SegmentationParameters(DistributionImage[] trainingImages)
        {
            TrainingSubsetPerTree = trainingImages.Length / 2;

            //get a feature offset vector about a third the avg size of an image
            trainingImages.GetRandomSubset(trainingImages.Length / 10).ForEach((el) =>
            {
                var coX = el.DistributionMap.Size.X * 0.4;
                var coY = el.DistributionMap.Size.Y * 0.4;
                if (coX > MaximumFeatureOffsetX) MaximumFeatureOffsetX = (int)coX;
                if (coY > MaximumFeatureOffsetY) MaximumFeatureOffsetY = (int)coY;
            });
        }

        public int NumberOfTrees = 8;
        public int TrainingSubsetPerTree = 10;
        public double SegmentatioSplitRatio = 0.02;
        public int MaximumFeatureOffsetX = 10;
        public int MaximumFeatureOffsetY = 10;
        public int ThresholdCandidateNumber = 20;
        public int MaxTreeDepth = 12;
    }
}
