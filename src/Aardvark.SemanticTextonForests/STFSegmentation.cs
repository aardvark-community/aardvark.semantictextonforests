using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Aardvark.Base;

/// <summary>
/// This File includes the necessary data structures and methods for the image segmentation system using the "Level 2" Forest.
/// </summary>
namespace Aardvark.SemanticTextonForests
{




    class SegmentationForest
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
        public ISamplingProvider SamplingProvider;

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
            List<SegmentationDataPoint> currentDatapoints, LabelDistribution classDist, TrainingParams parameters,
            out List<SegmentationDataPoint> leftRemaining, out List<SegmentationDataPoint> rightRemaining,
            out LabelDistribution leftClassDist, out LabelDistribution rightClassDist
            )
        {
            //generate random candidates for threshold
            var threshCandidates = new double[parameters.ThresholdCandidateNumber];
            for (int i = 0; i < threshCandidates.Length; i++)
            {
                threshCandidates[i] = Algo.Rand.NextDouble();
            }

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
                //for each candidate, try the split and calculate its expected gain in information
                foreach (var curThresh in threshCandidates)
                {
                    var currentLeftSet = new List<SegmentationDataPoint>();
                    var currentRightSet = new List<SegmentationDataPoint>();
                    LabelDistribution currentLeftClassDist = null;
                    LabelDistribution currentRightClassDist = null;

                    SplitDatasetWithThreshold(currentDatapoints, curThresh, parameters, out currentLeftSet, out currentRightSet, out currentLeftClassDist, out currentRightClassDist);
                    double leftEntr = CalcEntropy(currentLeftClassDist, parameters);
                    double rightEntr = CalcEntropy(currentRightClassDist, parameters);

                    //from original paper -> maximize the score
                    double leftWeight = (-1.0d) * currentLeftClassDist.GetLabelDistSum() / classDist.GetLabelDistSum();
                    double rightWeight = (-1.0d) * currentRightClassDist.GetLabelDistSum() / classDist.GetLabelDistSum();
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

            bool isLeaf = (Math.Abs(bestScore) < parameters.ThresholdInformationGainMinimum) || inputIsEmpty;   //no images reached this node or not enough information gain => leaf

            if (parameters.ForcePassthrough) //if passthrough mode is active, never create a leaf inside the tree (force-fill the tree)
            {
                isLeaf = false;
            }

            bool passThrough = (Math.Abs(bestScore) < parameters.ThresholdInformationGainMinimum) || inputIsOne;  //passthrough mode active => copy the parent node

            if (isLeaf)
            {
                leftRemaining = null;
                rightRemaining = null;
                leftClassDist = null;
                rightClassDist = null;
                return Algo.DeciderTrainingResult.Leaf;
            }

            if (!passThrough && !isLeaf)  //reports for passthrough and leaf nodes are printed in Node.train method
            {
                Report.Line(3, "NN t:" + bestThreshold + " s:" + bestScore + "; dp=" + currentDatapoints.Count + " l/r=" + bestLeftSet.Count + "/" + bestRightSet.Count + ((isLeaf) ? "->leaf" : ""));
            }

            this.DecisionThreshold = bestThreshold;
            leftRemaining = bestLeftSet;
            rightRemaining = bestRightSet;
            leftClassDist = bestLeftClassDist;
            rightClassDist = bestRightClassDist;

            if (passThrough || isLeaf)
            {
                return Algo.DeciderTrainingResult.PassThrough;
            }

            return Algo.DeciderTrainingResult.InnerNode;
        }

        /// <summary>
        /// Splits up the dataset using a threshold.
        /// </summary>
        /// <param name="dps">Input data point set.</param>
        /// <param name="threshold">Decision threshold.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <param name="leftSet">Output Left subset.</param>
        /// <param name="rightSet">Output Right subset.</param>
        /// <param name="leftDist">Class Distribution belonging to Left output.</param>
        /// <param name="rightDist">Class Distribution belonging to Right output.</param>
        private void SplitDatasetWithThreshold(List<SegmentationDataPoint> dps, double threshold, TrainingParams parameters, out List<SegmentationDataPoint> leftSet, out List<SegmentationDataPoint> rightSet, out LabelDistribution leftDist, out LabelDistribution rightDist)
        {
            var leftList = new List<SegmentationDataPoint>();
            var rightList = new List<SegmentationDataPoint>();

            int targetFeatureCount = Math.Min(dps.Count, parameters.MaxSampleCount);
            var actualDPS = dps.GetRandomSubset(targetFeatureCount);

            foreach (var dp in actualDPS)
            {
                //select only a subset of features
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

            leftDist = new LabelDistribution();
            rightDist = new LabelDistribution();
        }

        /// <summary>
        /// Calculates the entropy of one Class Distribution. It is used for estimating the quality of a split.
        /// </summary>
        /// <param name="dist">Input Class Distribution.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>Entropy value of the input distribution.</returns>
        private double CalcEntropy(LabelDistribution dist, TrainingParams parameters)
        {
            //from http://en.wikipedia.org/wiki/ID3_algorithm

            double sum = 0;
            //foreach(var cl in dist.ClassLabels)
            foreach (var cl in parameters.Labels)
            {
                var px = dist.GetLabelProbability(cl);
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
        public readonly int OffsetX, OffsetY, channel;

        public SegmentationFeatureProvider(int maximumOffset, int numClasses)
        {
            throw new NotImplementedException();
            //TODO: generate random offset and sample channel
        }

        public double GetFeature(SegmentationDataPoint dp)
        {
            //get feature value from the resulting label distribution
            throw new NotImplementedException();
        }
    }

    public class SegmentationDataPoint
    {
        public readonly PixImage<double> SegmentationMap;
        public readonly int X;
        public readonly int Y;
        public readonly int SX, SY;

        /// <summary>
        /// Scaling weight of this data point.
        /// </summary>
        public readonly double Weight;

        /// <summary>
        /// Index of this data point's label. Arbitrary value if unknown.
        /// </summary>
        public readonly int Label;

        public SegmentationDataPoint(PixImage<double> pi, int x, int y, int sx, int sy, double weight = 1.0, int label = -2)
        {
            if (pi == null) throw new ArgumentNullException();
            if (x < 0 || y < 0 || x >= pi.Size.X || y >= pi.Size.Y) throw new IndexOutOfRangeException();

            SegmentationMap = pi;
            X = x; Y = y;
            SX = sx; SY = sy;
            Weight = weight;
            Label = label;
        }
    }
}
