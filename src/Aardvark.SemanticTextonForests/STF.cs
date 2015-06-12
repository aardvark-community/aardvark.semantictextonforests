using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Aardvark.Base;
using Newtonsoft.Json;

namespace Aardvark.SemanticTextonForests
{
    #region Semantic Texton Forest

    /// <summary>
    /// One data point of an Image at a pixel coordinate position (X,Y).
    /// </summary>
    public class DataPoint
    {
        public readonly PixImage<byte> PixImage;
        public readonly int X;
        public readonly int Y;

        /// <summary>
        /// Scaling weight of this data point.
        /// </summary>
        public double Weight;

        /// <summary>
        /// Index of this data point's label. Arbitrary value if unknown.
        /// </summary>
        public readonly int Label;

        public DataPoint(PixImage<byte> pi, int x, int y, double weight = 1.0, int label = -2)
        {
            if (pi == null) throw new ArgumentNullException();
            if (x < 0 || y < 0 || x >= pi.Size.X || y >= pi.Size.Y) throw new IndexOutOfRangeException();

            PixImage = pi;
            X = x; Y = y;
            Weight = weight;
            Label = label;
        }

        /// <summary>
        /// Returns a new DataPoint with given label set.
        /// </summary>
        public DataPoint SetLabel(int label)
        {
            return new DataPoint(PixImage, X, Y, Weight, label);
        }

        [JsonIgnore]
        public V2i PixelCoords
        {
            get { return new V2i(X, Y); }
        }
    }

    /// <summary>
    /// A set of data points.
    /// </summary>
    public class DataPointSet
    {
        /// <summary>
        /// Collection of data points.
        /// </summary>
        public IList<DataPoint> Points;

        /// <summary>
        /// Scaling weight of this data point set (in addition to the individual point weights).
        /// </summary>
        public double Weight;

        [JsonIgnore]
        public int Count => Points.Count;

        public DataPointSet()
        {
            Points = new List<DataPoint>();
        }

        public DataPointSet(IList<DataPoint> points, double weight = 1.0)
        {
            Points = points;
            Weight = weight;
        }

        /// <summary>
        /// Adds two data point sets.
        /// </summary>
        public static DataPointSet operator +(DataPointSet current, DataPointSet other)
        {
            var ps = new List<DataPoint>();
            ps.AddRange(current.Points);
            ps.AddRange(other.Points);

            return new DataPointSet(ps, (current.Weight + other.Weight) / 2.0);
        }

        /// <summary>
        /// Calculates a weight for each label in this Data Point Set as the inverse Label frequency, i.e the ratio between the size of this label and all labels
        /// weight(label) = sum(label) / sum(all labels)
        /// </summary>
        /// <returns>Inverse Label Frequency</returns>
        public LabelDistribution GetLabelWeights()
        {
            var numClasses = Points.Max(x => x.Label) + 1;

            var labels = new double[numClasses].SetByIndex(i => i);

            var labelSums = new int[labels.Count()];
            var weights = new double[labels.Count()];

            for (int i = 0; i < labels.Count(); i++)
            {
                labelSums[i] = Points.Where(x => x.Label == labels[i]).Count();
            }

            var totalLabelSum = labelSums.Sum();

            for (int i = 0; i < labels.Count(); i++)
            {
                weights[i] = totalLabelSum / (double)((labelSums[i] <= 0) ? (1.0) : (labelSums[i]));
            }

            var weightsDist = new LabelDistribution(weights);

            //weightsDist.Normalize();

            return weightsDist;
        }
    }

    /// <summary>
    /// One feature of an image data point.
    /// </summary>
    public class Feature
    {
        /// <summary>
        /// The numerical value of this feature.
        /// </summary>
        public double Value;
    }

    /// <summary>
    /// A feature provider maps a data point to the numerical value of a feature according to the 
    /// mapping rule it implements. See implementations for details.
    /// </summary>
    public abstract class IFeatureProvider
    {
        /// <summary>
        /// Initialize the feature provider.
        /// </summary>
        /// <param name="pixelWindowSize">The size of the window from which this provider extracts the feature value</param>
        protected abstract void Init(int pixelWindowSize);

        /// <summary>
        /// Calculate the feature value from a given data point.
        /// </summary>
        /// <param name="point">Input data point.</param>
        /// <returns></returns>
        public abstract Feature GetFeature(DataPoint point);

        /// <summary>
        /// Calculate array of features from set of data points.
        /// </summary>
        /// <param name="points">Input data point set.</param>
        /// <returns></returns>
        public Feature[] GetArrayOfFeatures(DataPointSet points)
        {
            List<Feature> result = new List<Feature>();
            foreach (var point in points.Points)
            {
                result.Add(this.GetFeature(point));
            }
            return result.ToArray();
        }
    }

    /// <summary>
    /// A sampling provider retrieves a set of data points from a given image according to the sampling rule it implements.
    /// </summary>
    public abstract class ISamplingProvider
    {
        /// <summary>
        /// Initialize the sampling provider.
        /// </summary>
        /// <param name="pixWindowSize">Square sampling window size in pixels.</param>
        /// <param name="samplingFrequency">Distance from one sampling window center to the next (in both directions, in pixels).</param>
        public abstract void Init(int pixWindowSize, int samplingFrequency);
        /// <summary>
        /// Get a set of data points from a given image.
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <returns></returns>
        public abstract DataPointSet GetDataPoints(Image image);

        /// <summary>
        /// Gets a set of Data Points from a Labeled Patch (Classifier segmentation)
        /// </summary>
        /// <param name="patch"></param>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public abstract DataPointSet GetDataPoints(LabeledPatch patch, TrainingParams parameters);

        /// <summary>
        /// Get a collected set of data points from an array of images.
        /// </summary>
        /// <param name="labeledImages">Input image array.</param>
        /// <returns></returns>
        public abstract DataPointSet GetDataPoints(LabeledImage[] labeledImages, ForestLabelSource LabelSourceMode, LabelWeightingMode LabelWeightMode);

        /// <summary>
        /// Gets a collected Set of Data Points from an array of patches.
        /// </summary>
        /// <param name="labeledImages"></param>
        /// <param name="parameters"></param>
        /// <returns></returns>
        public abstract DataPointSet GetDataPoints(LabeledPatch[] labeledImages, TrainingParams parameters);
    }

    /// <summary>
    /// Result of binary decision.
    /// </summary>
    public enum Decision
    {
        Left,
        Right
    }

    /// <summary>
    /// The Decider makes a binary decision given the feature value of a datapoint. During training, the Decider learns a good
    /// decision threshold for the provided training data and feature/sampling providers in the function Decider.InitializeDecision. 
    /// Afterwards, the function Decider.Decide returns (Left/Right) if the datapoint's feature value is (Less than/Greater than) the 
    /// threshold. 
    /// </summary>
    public class Decider
    {
        /// <summary>
        /// The feature provider used in this Decider.
        /// </summary>
        public IFeatureProvider FeatureProvider;
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
        public Decider() { }

        /// <summary>
        /// Decides whether the data point's feature value corresponds to Left (less than) or Right (greater than).
        /// </summary>
        /// <param name="dataPoint">Input data point.</param>
        /// <returns>Left/Right Decision.</returns>
        public Decision Decide(DataPoint dataPoint)
        {
            return (FeatureProvider.GetFeature(dataPoint).Value < DecisionThreshold) ? Decision.Left : Decision.Right;
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
            DataPointSet currentDatapoints, LabelDistribution classDist, TrainingParams parameters,
            out DataPointSet leftRemaining, out DataPointSet rightRemaining,
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
            var bestLeftSet = new DataPointSet();
            var bestRightSet = new DataPointSet();
            LabelDistribution bestLeftClassDist = null;
            LabelDistribution bestRightClassDist = null;

            bool inputIsEmpty = currentDatapoints.Count == 0; //there is no image, no split is possible -> leaf
            bool inputIsOne = currentDatapoints.Count == 1;   //there is exactly one image, no split is possible -> leaf (or passthrough)

            if (!inputIsEmpty && !inputIsOne)
            {
                //for each candidate, try the split and calculate its expected gain in information
                foreach (var curThresh in threshCandidates)
                {
                    var currentLeftSet = new DataPointSet();
                    var currentRightSet = new DataPointSet();
                    LabelDistribution currentLeftClassDist = null;
                    LabelDistribution currentRightClassDist = null;

                    SplitDatasetWithThreshold(currentDatapoints, classDist.Distribution.Length, curThresh, parameters, 
                        out currentLeftSet, out currentRightSet, out currentLeftClassDist, out currentRightClassDist);
                    double leftEntr = CalcEntropy(currentLeftClassDist, classDist.Distribution.Length);
                    double rightEntr = CalcEntropy(currentRightClassDist, classDist.Distribution.Length);

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
        private void SplitDatasetWithThreshold(DataPointSet dps, int numLabels, double threshold, TrainingParams parameters, out DataPointSet leftSet, out DataPointSet rightSet, out LabelDistribution leftDist, out LabelDistribution rightDist)
        {
            var leftList = new List<DataPoint>();
            var rightList = new List<DataPoint>();

            int targetFeatureCount = Math.Min(dps.Count, parameters.MaxSampleCount);
            var actualDPS = dps.Points.GetRandomSubset(targetFeatureCount);

            foreach (var dp in actualDPS)
            {
                //select only a subset of features
                var feature = FeatureProvider.GetFeature(dp);

                if (feature.Value < threshold)
                {
                    leftList.Add(dp);
                }
                else
                {
                    rightList.Add(dp);
                }

            }

            leftSet = new DataPointSet(leftList);
            rightSet = new DataPointSet(rightList);

            leftDist = new LabelDistribution(numLabels, leftSet);
            rightDist = new LabelDistribution(numLabels, rightSet);
        }

        /// <summary>
        /// Calculates the entropy of one Class Distribution. It is used for estimating the quality of a split.
        /// </summary>
        /// <param name="dist">Input Class Distribution.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>Entropy value of the input distribution.</returns>
        private double CalcEntropy(LabelDistribution dist, int numClasses)
        {
            //from http://en.wikipedia.org/wiki/ID3_algorithm

            double sum = 0;
            //foreach(var cl in dist.ClassLabels)
            for(int c=0; c<numClasses; c++)
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

    /// <summary>
    /// A binary node of a Tree. The node references its left and right child in the binary tree to make traversal possible. It also
    /// represents one Decider, which makes the decision to pass a data point on in either the left or the right direction.
    /// </summary>
    public class Node
    {
        public bool IsLeaf = false;
        public int DistanceFromRoot = 0;
        public Node LeftChild;
        public Node RightChild;
        /// <summary>
        /// The Decider associated with this node.
        /// </summary>
        public Decider Decider;
        /// <summary>
        /// The Class Distribution corresponding to the training data that reached this node.
        /// </summary>
        public LabelDistribution LabelDistribution;
        /// <summary>
        /// This node's global index in the forest.
        /// </summary>
        public int GlobalIndex = -1;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public Node() { }

        /// <summary>
        /// Recursively gets a list of histogram nodes counting the (weighted) occurrences of each nodes for one data point. The results are appended 
        /// to currentList.
        /// </summary>
        /// <param name="dataPoint">Input Data Point.</param>
        /// <param name="currentHistogram">Part of a histogram corresponding to the input. It contains the occurrences of this node and each child node. </param>
        /// <param name="parameters">Parameters Object.</param>
        public void GetTextonization(DataPoint dataPoint, List<HistogramNode> currentHistogram, TrainingParams parameters)
        {
            switch (parameters.ClassificationMode)
            {
                case ClassificationMode.Semantic:

                    var rt = new HistogramNode();
                    rt.Index = GlobalIndex;
                    rt.Level = DistanceFromRoot;
                    rt.Value = 1;   //todo: weight
                    currentHistogram.Add(rt);

                    //descend left or right, or return if leaf
                    if (!this.IsLeaf)
                    {
                        if (Decider.Decide(dataPoint) == Decision.Left)   
                        {
                            LeftChild.GetTextonization(dataPoint, currentHistogram, parameters);
                        }
                        else            
                        {
                            RightChild.GetTextonization(dataPoint, currentHistogram, parameters);
                        }
                    }
                    else //break condition
                    {
                        return;
                    }
                    return;
                case ClassificationMode.LeafOnly:

                    if (!this.IsLeaf) //we are in a branching point, continue forward
                    {
                        if (Decider.Decide(dataPoint) == Decision.Left)
                        {
                            LeftChild.GetTextonization(dataPoint, currentHistogram, parameters);
                        }
                        else
                        {
                            RightChild.GetTextonization(dataPoint, currentHistogram, parameters);
                        }
                    }
                    else            //we are at a leaf, take this class distribution as result
                    {
                        var result = new HistogramNode();
                        result.Index = GlobalIndex;
                        result.Level = DistanceFromRoot;
                        result.Value = 1;   //todo: weight
                        var resList = new List<HistogramNode>();
                        resList.Add(result);
                        return;
                    }
                    return;

                default:
                    return;
            }
        }

        public void PushWeightsToLeaves(LabelDistribution weights)
        {
            if (!this.IsLeaf) //we are in a branching point, continue forward
            {
                LeftChild.PushWeightsToLeaves(weights);
                RightChild.PushWeightsToLeaves(weights);
            }
            else            //we are at a leaf, apply weights
            {
                LabelDistribution.Scale(weights);
                LabelDistribution.Normalize();
            }
        }

        /// <summary>
        /// Retrieves a label distribution (soft classification) for one data point of an image.
        /// </summary>
        /// <param name="dataPoint"></param>
        /// <returns></returns>
        public LabelDistribution GetDistribution(DataPoint dataPoint)
        {
            if (!this.IsLeaf) //we are in a branching point, continue forward
            {
                if (Decider.Decide(dataPoint) == Decision.Left)
                {
                    return LeftChild.GetDistribution(dataPoint);
                }
                else
                {
                    return RightChild.GetDistribution(dataPoint);
                }
            }
            else            //we are at a leaf, take this class distribution as result
            {
                return this.LabelDistribution;
            }
        }

        /// <summary>
        /// every node adds 0 to the histogram (=initialize the histogram configuration)
        /// </summary>
        /// <param name="initialHistogram">Histogram to be initialized.</param>
        public void InitializeEmpty(List<HistogramNode> initialHistogram)
        {
            var rt = new HistogramNode();
            rt.Index = GlobalIndex;
            rt.Level = DistanceFromRoot;
            rt.Value = 0;
            initialHistogram.Add(rt);

            //descend left and right, or return if leaf
            if (!this.IsLeaf)
            {
                LeftChild.InitializeEmpty(initialHistogram);
                RightChild.InitializeEmpty(initialHistogram);
            }
        }

        /// <summary>
        /// Returns soft classification of a data point
        /// </summary>
        /// <param name="dataPoint">input point</param>
        /// <param name="baseDistribution">empty distribution which is recursively filled</param>
        /// <returns>maximum depth of this data point's path, which is used to scale the distribution values</returns>
        internal int GetDistributionCumulative(DataPoint dataPoint, LabelDistribution baseDistribution)
        {
            if (!this.IsLeaf) //we are in a branching point, continue forward
            {
                if (Decider.Decide(dataPoint) == Decision.Left)
                {
                    var D = LeftChild.GetDistributionCumulative(dataPoint, baseDistribution);
                    var res = new LabelDistribution(LabelDistribution.Distribution);
                    res.Scale(1 /
                        Math.Pow(2, D - DistanceFromRoot + 1)
                        );
                    baseDistribution.AddDistribution(res);
                    return D;
                }
                else
                {
                    var D = RightChild.GetDistributionCumulative(dataPoint, baseDistribution);
                    var res = new LabelDistribution(LabelDistribution.Distribution);
                    res.Scale(1 /
                        Math.Pow(2, D - DistanceFromRoot + 1)
                        );
                    baseDistribution.AddDistribution(res);
                    return D;
                }
            }
            else            //we are at a leaf, take this class distribution as result
            {

                var res = new LabelDistribution(LabelDistribution.Distribution);
                res.Scale(1 / (double)2);
                baseDistribution.AddDistribution(res);
                return DistanceFromRoot;
            }
        }
    }

    /// <summary>
    /// A binary Tree, which is part of a Forest. The trained Tree provides a complete Textonization of an input Data Point Set. It contains
    /// one Root Node, from which the recursive functions for training and textonization can be called. 
    /// </summary>
    public class Tree
    {
        /// <summary>
        /// This tree's root node. Every traversal starts here.
        /// </summary>
        public Node Root;
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
        public Tree()
        {
            Root = new Node();
            Root.GlobalIndex = this.Index;
        }

        /// <summary>
        /// Creates a Textonization Histogram which contains all of this Tree's node occurrences for a given Data Point Set.
        /// </summary>
        /// <param name="dp">Input Data Point Set.</param>
        /// <param name="parameters">Parameters Object.</param>
        /// <returns>Textonization Histogram representing the input Data Point Set.</returns>
        public List<HistogramNode> GetHistogram(DataPointSet dp, TrainingParams parameters)
        {
            var result = new List<HistogramNode>();

            foreach (var point in dp.Points)
            {
                var cumulativeHist = new List<HistogramNode>();
                Root.GetTextonization(point, cumulativeHist, parameters);
                foreach (var el in cumulativeHist)        //this is redundant with initializeEmpty -> todo
                {
                    el.TreeIndex = this.Index;
                }
                result.AddRange(cumulativeHist);
            }

            return result;
        }

        public LabelDistribution GetDistribution(DataPoint dataPoint)
        {
            return Root.GetDistribution(dataPoint);
        }

        public LabelDistribution GetDistributionCumulative(DataPoint dataPoint, int numClasses)
        {
            var baseDistribution = new LabelDistribution(numClasses);

            Root.GetDistributionCumulative(dataPoint, baseDistribution);

            return baseDistribution;
        }


        /// <summary>
        /// Creates an empty Textonization Histogram. This ensures all nodes in this Tree are represented.
        /// </summary>
        /// <param name="currentList">Empty or partially filled Histogram Node List.</param>
        public void GetEmptyHistogram(List<HistogramNode> currentList)
        {
            var cumulativeHist = new List<HistogramNode>();
            Root.InitializeEmpty(cumulativeHist);
            foreach (var el in cumulativeHist)
            {
                el.TreeIndex = this.Index;
            }
            currentList.AddRange(cumulativeHist);
        }

        public void PushWeightsToLeaves(LabelDistribution weights)
        {
            Root.PushWeightsToLeaves(weights);
        }

    }

    /// <summary>
    /// A Semantic Texton Forest. After training, it contains a number of trained binary Trees. The trained Forest can evaluate an input image and 
    /// generate a Textonization representation according to its trained configuration. The resulting Textonization is the sum of individual
    /// Textonizations of the Trees.
    /// </summary>
    public class Forest
    {
        /// <summary>
        /// The collection of Trees belonging to this Forest.
        /// </summary>
        public Tree[] Trees;
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
        /// Global weights for the labels that occur in this Forest. Should be (proportional to) the inverse label frequency.
        /// </summary>
        public LabelDistribution LabelWeights;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public Forest() { }

        public Forest(string name, int numberOfTrees)
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
            Trees = new Tree[NumTrees].SetByIndex(i => new Tree() { Index = i });
        }

        /// <summary>
        /// Evaluates the forest for an input image and returns the image's Textonization. The result histogram is the sum of all Trees'
        /// individual Textonization histograms.
        /// </summary>
        /// <param name="img">Input Image.</param>
        /// <param name="parameters">Parameters Object</param>
        /// <returns>The Textonization of the input Image.</returns>
        public Textonization GetTextonization(Image img, TrainingParams parameters)
        {
            if (NumNodes <= -1)  //this part is deprecated
            {
                //NumNodes = Trees.Sum(x => x.NumNodes);
                throw new InvalidOperationException();
            }

            var result = new Textonization(NumNodes);

            var emptyHistogram = new List<HistogramNode>();
            Algo.TreeCounter = 0;

            foreach (var tree in Trees)    //for each tree, get a textonization of the data set and sum up the result
            {
                Algo.TreeCounter++;

                tree.GetEmptyHistogram(emptyHistogram);

                //add the current partial histogram to the overall result
                result.AddHistogram(tree.GetHistogram(tree.SamplingProvider.GetDataPoints(img), parameters));

            }

            //add the empty histogram (increments all nodes by 0 -> no change in values, but inserts unrepresented nodes)
            result.AddHistogram(emptyHistogram);

            return result;
        }

        public Textonization GetTextonization(LabeledPatch Patch, TrainingParams parameters)
        {
            if (NumNodes <= -1)  //this part is deprecated
            {
                //NumNodes = Trees.Sum(x => x.NumNodes);
                throw new InvalidOperationException();
            }

            var result = new Textonization(NumNodes);

            var emptyHistogram = new List<HistogramNode>();
            Algo.TreeCounter = 0;

            foreach (var tree in Trees)    //for each tree, get a textonization of the data set and sum up the result
            {
                Algo.TreeCounter++;

                tree.GetEmptyHistogram(emptyHistogram);

                //add the current partial histogram to the overall result
                result.AddHistogram(tree.GetHistogram(tree.SamplingProvider.GetDataPoints(Patch, parameters), parameters));
            }

            //add the empty histogram (increments all nodes by 0 -> no change in values, but inserts unrepresented nodes)
            result.AddHistogram(emptyHistogram);

            return result;
        }

        /// <summary>
        /// Calculates a distribution image for the input image. The Distribution image has the same size as the input image, the number of
        /// "color" channels equal to the count of labels, in which for each pixel the label distribution estimated by this forest is stored.
        /// </summary>
        /// <param name="image"></param>
        /// <returns></returns>
        public DistributionImage GetDistributionImage(LabeledImage image, ClassificationMode mode, LabelWeightingMode weightMode)
        {
            if (Trees.Length == 0 || Trees[0].Root.LabelDistribution.Distribution.Length <= 0)
            {
                return null;
            }

            //TODO: store this in a proper field
            var numClasses = Trees[0].Root.LabelDistribution.Distribution.Length;

            var result = new DistributionImage(image, numClasses);

            //for each coordinate, get a distribution for that pixel and put it into the result
            for (var x = 0; x < image.Image.PixImage.Size.X; x++)
            {
                for (var y = 0; y < image.Image.PixImage.Size.Y; y++)
                {
                    var dist = new LabelDistribution(numClasses);

                    //for each tree, get a label distribution and average the result
                    foreach (var tree in Trees)
                    {
                        var dataPoint = new DataPoint(image.Image.PixImage, x, y, 1.0, image.Label.Index);

                        if (mode == ClassificationMode.LeafOnly)
                        {
                            dist.AddDistribution(tree.GetDistribution(dataPoint));
                        }
                        else if(mode == ClassificationMode.Semantic)
                        {
                            dist.AddDistribution(tree.GetDistributionCumulative(dataPoint, numClasses));
                        }
                        
                    }

                    var scalefactor = 1.0 / Trees.Length;

                    dist.Scale(scalefactor);

                    if(weightMode == LabelWeightingMode.LabelsOnly)
                    {
                        dist.Scale(this.LabelWeights);
                    }

                    dist.Normalize();

                    //set the pixel value
                    result.SetDistributionValue(x, y, dist);
                }
            }

            //let the DistributionImage calculate its Summed Area Table, which it will then use for subsequent calls.
            result.CalculateSummedAreaMap();

            return result;
        }

        public LabelDistribution GetILP(LabeledImage image, ClassificationMode mode, LabelWeightingMode weightMode)
        {
            if (Trees.Length == 0 || Trees[0].Root.LabelDistribution.Distribution.Length <= 0)
            {
                return null;
            }

            //TODO: store this in a proper field
            var numClasses = Trees[0].Root.LabelDistribution.Distribution.Length;

            var result = new LabelDistribution(numClasses);

            //for each coordinate, get a distribution for that pixel and put it into the result
            var dist = new LabelDistribution(numClasses);

            //for each tree, get a label distribution and average the result
            foreach (var tree in Trees)
            {
                var dataPoints = tree.SamplingProvider.GetDataPoints(new LabeledImage[] { image }, ForestLabelSource.ImageGlobal, weightMode);

                foreach (var dataPoint in dataPoints.Points)
                {
                    if (mode == ClassificationMode.LeafOnly)
                    {
                        dist.AddDistribution(tree.GetDistribution(dataPoint));
                    }
                    else if (mode == ClassificationMode.Semantic)
                    {
                        dist.AddDistribution(tree.GetDistributionCumulative(dataPoint, numClasses));
                    }
                }
            }

            var scalefactor = 1.0 / Trees.Length;

            dist.Scale(scalefactor);

            if (weightMode == LabelWeightingMode.LabelsOnly)
            {
                dist.Scale(this.LabelWeights);
            }

            dist.Normalize();

            //set the pixel value
            result.AddDistribution(dist);

            result.Normalize();

            return result;
        }
    }
    #endregion

    #region Class Labels and Distributions

    /// <summary>
    /// Category/class/label.
    /// </summary>
    public class Label
    {
        /// <summary>
        /// Index in the global label list.
        /// </summary>
        public int Index { get; }

        /// <summary>
        /// Friendly Name.
        /// </summary>
        public string Name { get; }

        public Label()
        {
            Index = -1;
            Name = "";
        }

        public Label(int index)
        {
            Index = index;
            Name = $"Label {index}";
        }

        public Label(int index, string name)
        {
            Index = index;
            Name = name;
        }
    }

    /// <summary>
    /// A Label Distribution counting the occurrence of each label in a data set.
    /// </summary>
    public class LabelDistribution
    {
        /// <summary>
        /// Weighted histogram value for each label.
        /// </summary>
        public double[] Distribution;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public LabelDistribution() { }

        /// <summary>
        /// initializes all labels with a count of 0.
        /// </summary>
        /// <param name="allLabels">All labels.</param>
        public LabelDistribution(Label[] allLabels)
        {
            Distribution = new double[allLabels.Length]; ;
            for (int i = 0; i < allLabels.Length; i++) //allLabels must have a sequence of indices [0-n]
            {
                Distribution[i] = 0;
            }
        }

        public LabelDistribution(Label onlyLabel, int numClasses)
        {
            Distribution = new double[numClasses]; ;
            for (int i = 0; i < numClasses; i++)
            {
                Distribution[i] = 0;
            }
            Distribution[onlyLabel.Index] = 1;

        }

        public LabelDistribution(int numClasses)
        {
            Distribution = new double[numClasses];
            Distribution.SetByIndex(i => 0.0);
        }


        public LabelDistribution(double[] values)
        {
            Distribution = new double[values.Length];
            Distribution.SetByIndex(i => values[i]);
        }

        /// <summary>
        /// Create a distribution and set its value from a distribution map.
        /// </summary>
        /// <param name="numClasses"></param>
        /// <param name="dist"></param>
        /// <param name="x"></param>
        /// <param name="y"></param>
        public LabelDistribution(DistributionImage dist, long x, long y)
        {
            Distribution = new double[dist.DistributionMap.Size.Z];
            Distribution.SetByIndex(i => dist.DistributionMap[x,y,i]);
        }

        /// <summary>
        /// Creates a distribution from a collection of segmentation points
        /// </summary>
        /// <param name="points"></param>
        /// <param name="numClasses"></param>
        public LabelDistribution(List<SegmentationDataPoint> points, int numClasses)
        {
            if (points.Count == 0)
            {
                Distribution = new double[numClasses].Set(0);
            }
            else
            {
                Distribution = new double[points[0].DistributionImage.DistributionMap.Size.Z];
                points.ForEach((el) =>
                {
                    Distribution[el.Label] += el.Weight;
                });
            }
        }

        /// <summary>
        /// Creates a distribution for data points extracted from their underlying distribution maps.
        /// </summary>
        /// <param name="points"></param>
        /// <returns></returns>
        public static LabelDistribution GetSegmentationPrediction(List<SegmentationDataPoint> points, int numClasses)
        {
            if (points.Count == 0)
            {
                return new LabelDistribution(new double[numClasses].Set(0));
            }
            var Distribution = new double[points[0].DistributionImage.numChannels];
            points.ForEach((el) =>
            {
                var curDist = el.DistributionImage.GetWindowPrediction(el.X, el.Y, el.X + el.SX, el.Y + el.SY);
                Distribution.SetByIndex(i => curDist.Distribution[i]);
            });
            return new LabelDistribution(Distribution);
        }

        /// <summary>
        /// Initializes labels and adds the data points.
        /// </summary>
        /// <param name="allLabels">All labels.</param>
        /// <param name="dps">Labeled Data Points.</param>
        /// <param name="parameters">Parameters Object.</param>
        public LabelDistribution(Label[] allLabels, DataPointSet dps, TrainingParams parameters)
            : this(allLabels)
        {
            AddDatapoints(dps, parameters);
        }

        public LabelDistribution(int numClasses, DataPointSet dps)
        {
            Distribution = new double[numClasses];
            foreach(var dp in dps.Points)
            {
                Distribution[dp.Label] += dp.Weight;
            }
        }

        /// <summary>
        /// Adds one data point to the distribution.
        /// </summary>
        /// <param name="dp">Input Data Point.</param>
        /// <param name="parameters">Parameters Object.</param>
        public void AddDP(DataPoint dp, TrainingParams parameters)
        {
            AddClNum(parameters.Labels[dp.Label], dp.Weight);
        }

        /// <summary>
        /// Increments one distribution entry by the provided value.
        /// </summary>
        /// <param name="cl">Label.</param>
        /// <param name="num">Increment Value.</param>
        private void AddClNum(Label cl, double num)
        {
            Distribution[cl.Index] = Distribution[cl.Index] + num;
        }

        /// <summary>
        /// Adds a set of Data Points to the distribution.
        /// </summary>
        /// <param name="dps">Input Data Point Set.</param>
        /// <param name="parameters">Parameters Object.</param>
        public void AddDatapoints(DataPointSet dps, TrainingParams parameters)
        {
            foreach (var dp in dps.Points)
            {
                this.AddDP(dp, parameters);
            }
        }

        /// <summary>
        /// Returns the index of the label which has the highest probability.
        /// </summary>
        /// <returns></returns>
        public int GetMostLikelyLabel()
        {
            return Distribution.IndexOf(Distribution.Max());
        }

        public int GetSecondMostLikelyLabel()
        {
            double si = 0;
            double spi = 0;
            int mi = 0;
            for (int i = 0; i < Distribution.Length; i++)
            {
                si = Distribution[i];
                if (si > spi)
                {
                    spi = si;
                    mi = i;
                }
            }

            double si2 = 0;
            double spi2 = 0;
            int mi2 = 0;
            for (int i = 0; i < Distribution.Length; i++)
            {
                if (i == mi) continue;
                si2 = Distribution[i];
                if (si2 > spi2)
                {
                    spi2 = si2;
                    mi2 = i;
                }
            }

            return mi2;
        }

        //
        /// <summary>
        /// Returns the proportion of elements having this label versus the number of all elements in the distribution.
        /// </summary>
        /// <param name="label">Input Label.</param>
        /// <returns>Input Label Probability estimate.</returns>
        public double GetLabelProbability(Label label)
        {
            var sum = Distribution.Sum();
            if (sum == 0.0)
            {
                return 0.0;
            }
            var prob = Distribution[label.Index] / sum;
            return prob;
        }

        public double GetLabelProbability(int index)
        {
            var sum = Distribution.Sum();
            if (sum == 0.0)
            {
                return 0.0;
            }
            var prob = Distribution[index] / sum;
            return prob;
        }

        /// <summary></summary>
        /// <returns>Sum of distribution values.</returns>
        public double GetLabelDistSum()
        {
            return Distribution.Sum();
        }

        /// <summary>
        /// Normalizes this distribution.
        /// </summary>
        public void Normalize()
        {
            var sum = Distribution.Sum();

            if (sum == 0)
            {
                return;
            }

            for (int i = 0; i < Distribution.Length; i++)
            {
                Distribution[i] = Distribution[i] / sum;
            }
        }

        /// <summary>
        /// add the values of another Distribution to this one
        /// </summary>
        /// <param name="other"></param>
        public void AddDistribution(LabelDistribution other)
        {
            Distribution.SetByIndex(i => Distribution[i] + other.Distribution[i]);
        }

        /// <summary>
        /// scale the values in this distribution by a factor
        /// </summary>
        /// <param name="factor"></param>
        public void Scale(double factor)
        {
            for (int i = 0; i < Distribution.Length; i++)
            {
                this.Distribution[i] *= factor;
            }
        }

        public void Scale(LabelDistribution other)
        {
            for (int i = 0; i < Distribution.Length; i++)
            {
                if(Distribution[i].IsNaN())
                {
                    throw new Exception() { };
                }
                this.Distribution[i] *= other.Distribution[i];
            }
        }
        /// <summary>
        /// Soften this distribution by an exponent: p' = p ^ exponent
        /// </summary>
        /// <param name="exponent"></param>
        public void Soften(double exponent)
        {
            for (int i = 0; i < Distribution.Length; i++)
            {
                if (Distribution[i].IsNaN())
                {
                    throw new Exception() { };
                }
                var val = Distribution[i];
                Distribution[i] = Math.Pow(val, exponent);
            }
        }
    }
    #endregion

    #region Images and I/O

    /// <summary>
    /// The Textonization representation of a pixel region, as generated by a Forest. The Textonization is the occurrence histogram of all 
    /// Forest Tree Nodes passed by the pixel region's Features. The pixel region is sampled at each Forest Tree using its Sampling Provider to split
    /// it into a set of data points. Each data point is mapped to a numerical value at each Tree Node to decide whether to traverse the 
    /// binary subtree in the Left or Right direction. When a leaf is reached, the Textonization of one data point is the histogram of occurrences
    /// of Nodes along its path. The Textonization of the input pixel region is the sum of Textonizations of all its data points.
    /// </summary>
    public class Textonization
    {
        public HistogramNode[] Histogram; 

        public Textonization(int numNodes)
        {
            InitializeEmpty(numNodes);
        }

        /// <summary>
        /// Initializes an empty Textonization.
        /// </summary>
        /// <param name="numNodes">The number of Nodes in total.</param>
        public void InitializeEmpty(int numNodes)
        {
            Histogram = new HistogramNode[numNodes];

            for (int i = 0; i < numNodes; i++)
            {
                Histogram[i] = new HistogramNode() { Index = i, Level = 0, Value = 0 };
            }
        }

        /// <summary>
        /// Adds an unordered list of Histogram nodes to this Textonization.
        /// </summary>
        /// <param name="histogramNodes">Input Histogram nodes. The value of each node contained in this list is incremented.</param>
        public void AddHistogram(List<HistogramNode> histogramNodes)
        {
            foreach (var node in histogramNodes)
            {
                var localNode = this.Histogram[node.Index];

                localNode.Level = node.Level;
                localNode.TreeIndex = node.TreeIndex;
                localNode.Value += node.Value;
            }
        }

        /// <summary>
        /// Adds two textonizations. They must be generated by the same Forest.
        /// </summary>
        /// <param name="current"></param>
        /// <param name="other"></param>
        /// <returns></returns>
        public static Textonization operator +(Textonization current, Textonization other)     // (=be from the same forest)
        {
            var result = new Textonization(current.Histogram.Length);
            for (int i = 0; i < current.Histogram.Length; i++)
            {
                var curNode = current.Histogram[i];
                var otherNode = other.Histogram.First(t => t.Index == curNode.Index);
                var res = new HistogramNode();
                res.Index = curNode.Index;
                res.Level = curNode.Level;
                res.Value = curNode.Value + otherNode.Value;
                result.Histogram[i] = res;
            }
            return result;
        }

    }

    /// <summary>
    /// Represents an entry in a Textonization histogram. The Histogram Node corresponds to one Tree Node in a Forest, and
   ///  stores the Index, TreeIndex and Level. 
    /// </summary>
    public class HistogramNode
    {
        /// <summary>
        /// The global index of the corresponding Tree Node.
        /// </summary>
        public int Index = -1;
        /// <summary>
        /// The tree index of the corresponding Tree.
        /// </summary>
        public int TreeIndex = -1;
        /// <summary>
        /// The depth of the corresponding Tree Node.
        /// </summary>
        public int Level = -1;
        /// <summary>
        /// The value of this histogram bin. It's a double because the value may be scaled.
        /// </summary>
        public double Value = 0; 
    }

    /// <summary>
    /// Represents an image which is loaded from disk.
    /// </summary>
    public class Image
    {
        /// <summary>
        /// File Path of the Image.
        /// </summary>
        public string ImagePath;
        public V2d Scale;

        //lazy loading
        private PixImage<byte> pImage;
        private bool isLoaded = false;

        /// <summary>
        /// JSON constructor.
        /// </summary>
        public Image()
        {

        }

        /// <summary>
        /// Creates a new Image without loading the file into memory. It is loaded on first use.
        /// </summary>
        /// <param name="filePath">Path of the image file.</param>
        public Image(string filePath)
        {
            ImagePath = filePath;
            Scale = new V2d(1.0);
        }

        [JsonIgnore]
        public PixImage<byte> PixImage
        {
            get
            {
                if (!isLoaded)
                {
                    Load();
                }
                return pImage;
            }
        }

        private void Load()
        {
            pImage = new PixImage<byte>(ImagePath);

            
            //var newsize = (V2d)pImage.Size * Scale;

            //var newsizeinteger = new V2i(newsize);

            //var asdf = pImage.Copy();

            //var resizedImage = asdf.Resized(newsizeinteger, ImageInterpolation.Near);

            //pImage = resizedImage;

            isLoaded = true;
        }
    }

    /// <summary>
    /// Represents one rectangular patch of an image.
    /// </summary>
    public class LabeledPatch
    {
        public LabeledImage ParentImage;
        public SegmentationMappingRule MappingRule;
        private Image SegmentationMap;

        [JsonIgnore]
        public PixImage<byte> PixImage
        {
            get
            {
                return GetSubvolume(ParentImage.Image);
            }
        }

        [JsonIgnore]
        public PixImage<byte> SegmentationImage
        {
            get
            {
                return GetSubvolume(SegmentationMap);
            }
        }

        public Label Label;

        //image coordinates of the rectangle this patch represents
        //top left pixel
        public int X = -1;
        public int Y = -1;
        //rectangle size in pixels
        public int SX = -1;
        public int SY = -1;

        public LabeledPatch(LabeledImage parentImage, Image segmentationImage, SegmentationMappingRule mappingRule, TrainingParams parameters, int X, int Y, int SX, int SY)
        {
            MappingRule = mappingRule;
            SegmentationMap = segmentationImage;
            ParentImage = parentImage;
            this.X = X;
            this.Y = Y;
            this.SX = SX;
            this.SY = SY;

            //TODO: improve this, label is only estimated from the patch's center pixel
            Label = mappingRule(parameters.SegmentationLabels, segmentationImage.PixImage, X + (SX / 2), Y + (SY / 2));
        }

        private PixImage<byte> GetSubvolume(Image image)
        {
            int dSX = Math.Min(SX, image.PixImage.Size.X);
            int dSY = Math.Min(SY, image.PixImage.Size.Y);

            if(X+dSX >= image.PixImage.Size.X)
            {
                dSX = image.PixImage.Size.X - X;
            }

            if (Y + dSY >= image.PixImage.Size.Y)
            {
                dSY = image.PixImage.Size.Y - Y;
            }

            SX = dSX;
            SY = dSY;

            return MatrixCache.GetMatrixFrom(image.PixImage).SubMatrix(X, Y, dSX, dSY).ToPixImage<byte>();
        }
    }

    /// <summary>
    /// Image with added Label, used for training and testing.
    /// </summary>
    public class LabeledImage
    {
        public Image Image { get; }
        public Label Label { get; }

        public DistributionImage LabelMap;
        public bool HasLabelMap = false;

        /// <summary>
        /// This Image's training bias.
        /// </summary>
        public double TrainingBias = 1.0f;

        /// <summary>
        /// JSON constructor.
        /// </summary>
        public LabeledImage() { }

        //creates a new image from filename
        public LabeledImage(string imageFilename, Label label)
        {
            Image = new Image(imageFilename);
            Label = label;
        }

        public LabeledImage(string imageFilename)
        {
            Image = new Image(imageFilename);
            Label = new Label();
        }

        /// <summary>
        /// supply a label map for this image, which contains a label (distribution) for each individual pixel.
        /// 
        /// </summary>
        /// <param name="map"></param>
        public void SetLabelMap(DistributionImage map)
        {
            HasLabelMap = true;
            LabelMap = map;
        }

        /// <summary>
        /// Returns the label of the pixel according to the label map.
        /// 
        /// TODO: This with a label distribution instead of only one label.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <returns></returns>
        public int GetLabelOfPixel(long x, long y)
        {
            var res = LabelMap.GetDistributionValue(x, y);

            return res.GetMostLikelyLabel();
        }

        public int GetLabelOfRegion(int minX, int minY, int maxX, int maxY)
        {
            var res = LabelMap.GetWindowPrediction(minX, minY, maxX, maxY);

            return res.GetMostLikelyLabel();
        }
    }

    /// <summary>
    /// Labeled Image with added Textonization. If Label is unknown, it can be set to an arbitrary value.
    /// </summary>
    public class TextonizedLabeledImage : ITextonizable
    {
        public LabeledImage Image { get; }
        public Textonization Textonization { get; }

        public Label Label => Image.Label;

        /// <summary>
        /// JSON constructor.
        /// </summary>
        public TextonizedLabeledImage() { }

        public TextonizedLabeledImage(LabeledImage image, Textonization textonization)
        {
            Image = image;
            Textonization = textonization;
        }
    }

    public class TextonizedLabeledPatch : ITextonizable
    {
        public LabeledPatch Patch { get; }

        public Textonization Textonization { get; }

        public Label Label => Patch.Label;

        public TextonizedLabeledPatch() { }

        public TextonizedLabeledPatch(LabeledPatch patch, Textonization textonization)
        {
            Patch = patch;
            Textonization = textonization;
        }
    }

    public interface ITextonizable
    {
        Textonization Textonization { get; }

        Label Label { get; }
    }

    public class DistributionImage
    {
        public LabeledImage Image { get; }

        public Volume<double> DistributionMap;

        private Volume<double> SummedAreaMap;
        bool SAMIsReady = false;

        public int numChannels => (int)DistributionMap.Size.Z;

        private LabelDistribution ILP;
        private bool ILPIsCalculated = false;

        public DistributionImage(LabeledImage parentImage, int numClasses)
        {
            Image = parentImage;

            //initializes a new PixImage with the size of the parent image and a channel count of numClasses
            DistributionMap = new Volume<double>(parentImage.Image.PixImage.Size.X, parentImage.Image.PixImage.Size.Y, numClasses);
        }

        /// <summary>
        /// Sets the Distribution Map's value at the coordinates x,y to the Distribution dist.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="y"></param>
        /// <param name="dist"></param>
        public void SetDistributionValue(int x, int y, LabelDistribution dist)
        {
            for (int i = 0; i < dist.Distribution.Length; i++)
            {
                DistributionMap[x, y, i] = dist.Distribution[i];
            }
        }

        /// <summary>
        /// Creates and stores a Summed Area Table for this distribution map. After this method has finished,
        /// calls to GetWindowPrediction will be handled using the SAT.
        /// </summary>
        public void CalculateSummedAreaMap()
        {
            SummedAreaMap = new Volume<double>(DistributionMap.Size.X, DistributionMap.Size.Y, numChannels);

            //initialize the top left cell
            copyValue(DistributionMap, SummedAreaMap, 0, 0);

            //initialize top row
            SummedAreaMap.ForeachX((x) =>
            {
                if (x == 0) return;
                //var value = add(getValue(SummedAreaMap, x - 1, 0), getValue(DistributionMap, x, 0));
                var vp = SummedAreaMap.SubVector(new V3l(x - 1, 0, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
                var vc = DistributionMap.SubVector(new V3l(x, 0, 0), DistributionMap.SZ, DistributionMap.DZ);
                var res = SummedAreaMap.SubVector(new V3l(x, 0, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
                res.Apply(vp, (elc, elo) => elc + elo);
                res.Apply(vc, (elc, elo) => elc + elo);
                //setValue(SummedAreaMap, x, 0, value);
            });

            //initialize left column
            SummedAreaMap.ForeachY((y) =>
            {
                if (y == 0) return;
                //var value = add(getValue(SummedAreaMap, 0, y - 1), getValue(DistributionMap, 0, y));
                //setValue(SummedAreaMap, 0, y, value);
                var vp = SummedAreaMap.SubVector(new V3l(0, y-1, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
                var vc = DistributionMap.SubVector(new V3l(0, y, 0), DistributionMap.SZ, DistributionMap.DZ);
                var res = SummedAreaMap.SubVector(new V3l(0, y, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
                res.Apply(vp, (elc, elo) => elc + elo);
                res.Apply(vc, (elc, elo) => elc + elo);
            });

            //initialize the rest of the volume
            SummedAreaMap.ForeachXY((x, y) =>
            {
                if (x == 0 || y == 0) return;
                //var br = getValue(DistributionMap, x, y);
                //var bl = getValue(SummedAreaMap, x - 1, y);
                //var tr = getValue(SummedAreaMap, x, y - 1);
                //var tl = getValue(SummedAreaMap, x - 1, y - 1);

                //var v1 = add(br, bl);
                //var v2 = add(v1, tr);
                //var value = subtract(v2, tl);

                //setValue(SummedAreaMap, x, y, value);

                var br = DistributionMap.SubVector(new V3l(x, y, 0), DistributionMap.SZ, DistributionMap.DZ);
                var bl = SummedAreaMap.SubVector(new V3l(x - 1, y, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
                var tr = SummedAreaMap.SubVector(new V3l(x, y - 1, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
                var tl = SummedAreaMap.SubVector(new V3l(x - 1, y - 1, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);

                var res = SummedAreaMap.SubVector(new V3l(x, y, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);

                res.Apply(br, (elc, elo) => elc + elo);
                res.Apply(bl, (elc, elo) => elc + elo);
                res.Apply(tr, (elc, elo) => elc + elo);
                res.Apply(tl, (elc, elo) => elc - elo);
            });

            SAMIsReady = true;
        }

        /// <summary>
        /// elementwise addition
        /// </summary>
        /// <param name="self"></param>
        /// <param name="other"></param>
        /// <returns></returns>
        internal double[] add(double[] self, double[] other)
        {
            var result = new double[self.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = self[i] + other[i];
            }
            return result;
            //return self.Zip(other, (a, b) => (a + b)).ToArray();
        }

        internal double[] subtract(double[] self, double[] other)
        {
            var result = new double[self.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = self[i] - other[i];
            }
            return result;
            //return self.Zip(other, (a, b) => (a - b)).ToArray();
        }

        internal void copyValue(Volume<double> source, Volume<double> target, long x, long y)
        {
            //for (int i = 0; i < source.Size.Z; i++)
            //{
            //    target[x, y, i] = source[x, y, i];
            //}
            var vs = source.SubVector(new V3l(x, y, 0), source.SZ, source.DZ);
            target.SubVector(new V3l(x, y, 0), target.SZ, target.DZ).Apply(vs, (elt, els) => els);
        }

        internal double[] getValue(Volume<double> source, long x, long y)
        {
            var result = new double[source.Size.Z];
            var idx = source.Info.Index(x, y, 0);
            var dz = source.DZ;
            for (int i = 0; i < source.SZ; i++)
            {
                result[i] = source[idx];
                idx += dz;
            }
            return result;
        }

        internal void setValue(Volume<double> target, long x, long y, double[] value)
        {
            for (int i = 0; i < target.Size.Z; i++)
            {
                target[x, y, i] = value[i];
            }
        }

        public LabelDistribution GetDistributionValue(long x, long y)
        {
            return new LabelDistribution(this, x, y);
        }

        /// <summary>
        /// Gets the label distribution for the whole distribution map.
        /// </summary>
        /// <returns>Averaged global label distribution.</returns>
        public LabelDistribution GetILP()
        {
            if(ILPIsCalculated)
            {
                return ILP;
            }
            else
            {
                ILP = GetWindowPrediction(0, 0, (int)DistributionMap.Size.X - 1, (int)DistributionMap.Size.Y - 1);
                ILPIsCalculated = true;
                return ILP;
            }
        }

        /// <summary>
        /// Gets the label distribution for one sub rectangle of the distribution map.
        /// </summary>
        /// <param name="minX">Top left pixel coordinate.</param>
        /// <param name="minY">Top left pixel coordinate.</param>
        /// <param name="maxX">Bottom right pixel coordinate.</param>
        /// <param name="maxY">Bottom right pixel coordinate.</param>
        /// <returns>Averaged label distribution for the input rectangle.</returns>
        public LabelDistribution GetWindowPrediction(int minX, int minY, int maxX, int maxY)
        {
            if (SAMIsReady)
            {
                return CalculateDistributionFromSAM(minX, minY, maxX, maxY);
            }
            else
            {
                return CalculateDistributionFromMap(minX, minY, maxX, maxY);
            }
        }

        private LabelDistribution CalculateDistributionFromMap(int minX, int minY, int maxX, int maxY)
        {
            var numClasses = DistributionMap.Size.Z;
            var ILPvalues = new double[numClasses].Set(0);

            var pixCounter = 0;

            var subWindow = DistributionMap.SubVolume(minX, minY, 0, maxX - minX, maxY - minY, numClasses);

            subWindow.ForeachXY((x, y) =>
                   {
                       pixCounter++;
                       for (var c = 0; c < numClasses; c++)
                       {
                           ILPvalues[c] += subWindow[x, y, c];
                       }
                   }
            );

            ILPvalues.SetByIndex((i) => ILPvalues[i] / pixCounter);

            return new LabelDistribution(ILPvalues);
        }

        /// <summary>
        /// set the distribution image's rectangular window to the values in a label distribution
        /// </summary>
        /// <param name="minX"></param>
        /// <param name="minY"></param>
        /// <param name="maxX"></param>
        /// <param name="maxY"></param>
        /// <param name="value"></param>
        public void setRange(int minX, int minY, int maxX, int maxY, LabelDistribution value)
        {
            var numClasses = DistributionMap.Size.Z;
            var ILPvalues = new double[numClasses].Set(0);

            var subWindow = DistributionMap.SubVolume(minX, minY, 0, maxX - minX, maxY - minY, numClasses);

            subWindow.ForeachXY((x, y) =>
                    {
                        for (var c = 0; c < numClasses; c++)
                        {
                            subWindow[x, y, c] = value.Distribution[c];
                        }
                    }
            );
        }

        private LabelDistribution CalculateDistributionFromSAM(int minX, int minY, int maxX, int maxY)
        {
            //var A = getValue(SummedAreaMap, minX, minY);
            //var B = getValue(SummedAreaMap, maxX, minY);
            //var C = getValue(SummedAreaMap, minX, maxY);
            //var D = getValue(SummedAreaMap, maxX, maxY);

            //var r1 = add(D, A);
            //var r2 = subtract(r1, B);
            //var result = subtract(r2, C);

            //var scaleX = (maxX - minX) + 1.0;
            //var scaleY = (maxY - minY) + 1.0;
            //result.SetByIndex((i) => result[i] / (scaleX * scaleY)); //scale back to (0,1)

            var A = SummedAreaMap.SubVector(new V3l(minX, minY, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
            var B = SummedAreaMap.SubVector(new V3l(maxX, minY, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
            var C = SummedAreaMap.SubVector(new V3l(minX, maxY, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);
            var D = SummedAreaMap.SubVector(new V3l(maxX, maxY, 0), SummedAreaMap.SZ, SummedAreaMap.DZ);

            var result = new Vector<double>(SummedAreaMap.SZ);

            result.Apply(D, (elc, elo) => elc + elo);
            result.Apply(A, (elc, elo) => elc + elo);
            result.Apply(B, (elc, elo) => elc - elo);
            result.Apply(C, (elc, elo) => elc - elo);

            var scaleX = (maxX - minX) + 1.0;
            var scaleY = (maxY - minY) + 1.0;

            result.Apply(i => i / (scaleX * scaleY));

            return new LabelDistribution(result.Array.ToArrayOfT<double>());
        }
    }

    #endregion





    #region Providers

    /// <summary>
    /// Classification modes.
    /// </summary>
    public enum ClassificationMode
    {
        LeafOnly,   //Only use Tree leaves.
        Semantic    //Use the entire Semantic Texton Forest method.
    }

    /// <summary>
    /// Mappings of Data Points to numeric Feature Values.
    /// </summary>
    public enum FeatureType
    {
        RandomPixelValue,            //Intensity value of a pixel.
        RandomTwoPixelSum,           //Sum of two pixels.
        RandomTwoPixelDifference,    //Difference of two pixels.
        RandomTwoPixelAbsDiff,       //Absolute value of the difference of two pixels.
        SelectRandom                 //Randomly selects one of the above.
    };

    /// <summary>
    /// Methods to sample Data Points from Images.
    /// </summary>
    public enum SamplingType
    {
        RegularGrid,    //Sample the Image in a regular grid.
        RandomPoints    //Take random points within the image as samples.
    };

    /// <summary>
    /// Factory class which returns a Feature Provider according to its FeatureType setting.
    /// </summary>
    public class FeatureProviderFactory
    {
        private IFeatureProvider CurrentProvider;
        private int PixWinSize;
        private FeatureType CurrentChoice;

        /// <summary>
        /// Creates a new Factory and sets the initial settings.
        /// </summary>
        /// <param name="featureType">Feature Type of the next Provider.</param>
        /// <param name="pixelWindowSize">Pixel Window size of the next Provider.</param>
        public FeatureProviderFactory(FeatureType featureType, int pixelWindowSize)
        {
            SelectProvider(featureType, pixelWindowSize);
        }

        /// <summary>
        /// Configure the next Feature Provider to be created. 
        /// </summary>
        /// <param name="featureType">Feature Type of the next Provider.</param>
        /// <param name="pixelWindowSize">Pixel Window size of the next Provider.</param>
        public void SelectProvider(FeatureType featureType, int pixelWindowSize)
        {
            CurrentChoice = featureType;
            PixWinSize = pixelWindowSize;
        }

        /// <summary>
        /// Returns a new Feature Provider according to this Factory's settings. 
        /// </summary>
        /// <returns>New Feature Provider.</returns>
        public IFeatureProvider GetNewProvider()
        {
            switch (CurrentChoice)
            {
                case FeatureType.RandomPixelValue:
                    CurrentProvider = new ValueOfPixelFeatureProvider(PixWinSize);
                    break;
                case FeatureType.RandomTwoPixelSum:
                    CurrentProvider = new PixelSumFeatureProvider(PixWinSize);
                    break;
                case FeatureType.RandomTwoPixelAbsDiff:
                    CurrentProvider = new AbsDiffOfPixelFeatureProvider(PixWinSize);
                    break;
                case FeatureType.RandomTwoPixelDifference:
                    CurrentProvider = new PixelDifferenceFeatureProvider(PixWinSize);
                    break;
                case FeatureType.SelectRandom:      //select one of the other providers at random - equal chance
                    var choice = Algo.Rand.Next(4);
                    switch (choice)
                    {
                        case 0:
                            CurrentProvider = new ValueOfPixelFeatureProvider(PixWinSize);
                            break;
                        case 1:
                            CurrentProvider = new PixelSumFeatureProvider(PixWinSize);
                            break;
                        case 2:
                            CurrentProvider = new AbsDiffOfPixelFeatureProvider(PixWinSize);
                            break;
                        case 3:
                            CurrentProvider = new PixelDifferenceFeatureProvider(PixWinSize);
                            break;
                        default:
                            return null;
                    }
                    break;
                default:
                    CurrentProvider = new ValueOfPixelFeatureProvider(PixWinSize);
                    break;
            }
            return CurrentProvider;
        }
    }

    /// <summary>
    /// Selects two random pixels from the sampling window and returns their sum.
    /// </summary>
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

        public PixelSumFeatureProvider(int pixelWindowSize)
        {
            Init(pixelWindowSize);
        }

        protected override void Init(int pixelWindowSize)
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

            var pi = MatrixCache.GetMatrixFrom(point.PixImage);

            var p1 = point.PixelCoords + FirstPixelOffset;
            var p2 = point.PixelCoords + SecondPixelOffset;

            //clamp the coordinates
            if (p1.X<0)
            {
                p1.X = 0;
            }
            if (p1.X > pi.Size.X - 1)
            {
                p1.X = (int)pi.Size.X - 1;
            }
            if (p1.Y < 0)
            {
                p1.Y = 0;
            }
            if (p1.Y > pi.Size.Y - 1)
            {
                p1.Y = (int)pi.Size.Y - 1;
            }
            if (p2.X < 0)
            {
                p2.X = 0;
            }
            if (p2.X > pi.Size.X - 1)
            {
                p2.X = (int)pi.Size.X - 1;
            }
            if (p2.Y < 0)
            {
                p2.Y = 0;
            }
            if (p2.Y > pi.Size.Y - 1)
            {
                p2.Y = (int)pi.Size.Y - 1;
            }

            var sample1 = pi[p1].ToGrayByte().ToDouble();
            var sample2 = pi[p2].ToGrayByte().ToDouble();

            var op = (sample1 + sample2) / 2.0; //divide by two for normalization

            result.Value = op;

            return result;
        }
    }

    /// <summary>
    /// Selects two random pixels from the sampling window and returns their difference.
    /// </summary>
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

        public PixelDifferenceFeatureProvider(int pixelWindowSize)
        {
            Init(pixelWindowSize);
        }

        protected override void Init(int pixelWindowSize)
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

            var pi = MatrixCache.GetMatrixFrom(point.PixImage);

            var p1 = point.PixelCoords + FirstPixelOffset;
            var p2 = point.PixelCoords + SecondPixelOffset;

            //clamp the coordinates
            if (p1.X < 0)
            {
                p1.X = 0;
            }
            if (p1.X > pi.Size.X - 1)
            {
                p1.X = (int)pi.Size.X - 1;
            }
            if (p1.Y < 0)
            {
                p1.Y = 0;
            }
            if (p1.Y > pi.Size.Y - 1)
            {
                p1.Y = (int)pi.Size.Y - 1;
            }
            if (p2.X < 0)
            {
                p2.X = 0;
            }
            if (p2.X > pi.Size.X - 1)
            {
                p2.X = (int)pi.Size.X - 1;
            }
            if (p2.Y < 0)
            {
                p2.Y = 0;
            }
            if (p2.Y > pi.Size.Y - 1)
            {
                p2.Y = (int)pi.Size.Y - 1;
            }

            var sample1 = pi[p1].ToGrayByte().ToDouble();
            var sample2 = pi[p2].ToGrayByte().ToDouble();

            var op = ((sample1 - sample2) + 1.0) / 2.0; //normalize to [0,1]

            result.Value = op;

            return result;
        }
    }

    /// <summary>
    /// Selects one random pixel from the sampling window and returns its intensity value.
    /// </summary>
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

        public ValueOfPixelFeatureProvider(int pixelWindowSize)
        {
            Init(pixelWindowSize);
        }

        protected override void Init(int pixelWindowSize)
        {

            int half = (int)(pixelWindowSize / 2);
            int x = Algo.Rand.Next(pixelWindowSize) - half;
            int y = Algo.Rand.Next(pixelWindowSize) - half;

            PixelOffset = new V2i(x, y);
        }

        public override Feature GetFeature(DataPoint point)
        {
            Feature result = new Feature();

            var pi = MatrixCache.GetMatrixFrom(point.PixImage);

            var p = point.PixelCoords + PixelOffset;

            //clamp the coordinates
            if (p.X < 0)
            {
                p.X = 0;
            }
            if (p.X > pi.Size.X - 1)
            {
                p.X = (int)pi.Size.X - 1;
            }
            if (p.Y < 0)
            {
                p.Y = 0;
            }
            if (p.Y > pi.Size.Y - 1)
            {
                p.Y = (int)pi.Size.Y - 1;
            }

            var sample = pi[p].ToGrayByte().ToDouble();

            result.Value = sample;

            return result;
        }
    }

    /// <summary>
    /// Selects two random pixels from the sampling window and returns their absolute difference.
    /// </summary>
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

        public AbsDiffOfPixelFeatureProvider(int pixelWindowSize)
        {
            Init(pixelWindowSize);
        }

        protected override void Init(int pixelWindowSize)
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

            var pi = MatrixCache.GetMatrixFrom(point.PixImage);

            var p1 = point.PixelCoords + FirstPixelOffset;
            var p2 = point.PixelCoords + SecondPixelOffset;

            //clamp the coordinates
            if (p1.X < 0)
            {
                p1.X = 0;
            }
            if (p1.X > pi.Size.X - 1)
            {
                p1.X = (int)pi.Size.X - 1;
            }
            if (p1.Y < 0)
            {
                p1.Y = 0;
            }
            if (p1.Y > pi.Size.Y - 1)
            {
                p1.Y = (int)pi.Size.Y - 1;
            }
            if (p2.X < 0)
            {
                p2.X = 0;
            }
            if (p2.X > pi.Size.X - 1)
            {
                p2.X = (int)pi.Size.X - 1;
            }
            if (p2.Y < 0)
            {
                p2.Y = 0;
            }
            if (p2.Y > pi.Size.Y - 1)
            {
                p2.Y = (int)pi.Size.Y - 1;
            }

            var sample1 = pi[p1].ToGrayByte().ToDouble();
            var sample2 = pi[p2].ToGrayByte().ToDouble();

            var op = Math.Abs(sample2 - sample1);

            result.Value = op;

            return result;
        }
    }

    /// <summary>
    /// Factory Class which creates Sampling Providers according to its setting. 
    /// </summary>
    public class SamplingProviderFactory
    {
        private ISamplingProvider CurrentProvider;
        private int PixelWindowSize;
        private SamplingType SamplingType;
        private int RandomSampleCount = 0;
        private int SamplingFrequency = 4;

        public SamplingProviderFactory(SamplingType samplingType, int pixelWindowSize, int samplingFrequency)
        {
            SelectProvider(samplingType, pixelWindowSize, samplingFrequency);
        }

        public void SelectProvider(SamplingType samplingType, int pixelWindowSize, int samplingFrequency)
        {
            this.PixelWindowSize = pixelWindowSize;
            this.SamplingType = samplingType;
            this.RandomSampleCount = 500;
            this.SamplingFrequency = samplingFrequency;
        }

        public void SelectProvider(SamplingType samplingType, int pixelWindowSize, int randomSampleCount, int samplingFrequency)
        {
            this.PixelWindowSize = pixelWindowSize;
            this.SamplingType = samplingType;
            this.RandomSampleCount = randomSampleCount;
            this.SamplingFrequency = samplingFrequency;
        }

        public ISamplingProvider GetNewProvider()
        {

            switch (SamplingType)
            {
                case SamplingType.RegularGrid:
                    CurrentProvider = new RegularGridSamplingProvider();
                    CurrentProvider.Init(PixelWindowSize, SamplingFrequency);
                    break;
                case SamplingType.RandomPoints:
                    var result = new RandomPointSamplingProvider();
                    result.Init(PixelWindowSize, SamplingFrequency);
                    result.SampleCount = this.RandomSampleCount;
                    CurrentProvider = result;
                    break;
                default:
                    CurrentProvider = new RegularGridSamplingProvider();
                    CurrentProvider.Init(PixelWindowSize, SamplingFrequency);
                    break;
            }

            return CurrentProvider;
        }
    }

    /// <summary>
    /// Gets a regular grid starting from the top left and continuing as long as there are pixels left.
    /// </summary>
    public class RegularGridSamplingProvider : ISamplingProvider
    {
        public int PixWinSize;
        public int SamplingFrequency;

        public override void Init(int pixWindowSize, int samplingFrequency)
        {
            PixWinSize = pixWindowSize;
            SamplingFrequency = samplingFrequency;
        }

        public override DataPointSet GetDataPoints(Image image)
        {
            var pi = MatrixCache.GetMatrixFrom(image.PixImage);

            var result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling(PixWinSize / 2.0); 

            int pointCounter = 0;

            for (int x = borderOffset; x < pi.SX - borderOffset; x += SamplingFrequency)
            {
                for (int y = borderOffset; y < pi.SY - borderOffset; y += SamplingFrequency)
                {
                    var newDP = new DataPoint(image.PixImage, x, y);
                    result.Add(newDP);
                    pointCounter++;
                }
            }

            var bias = 1.0 / pointCounter;     //weigh the sample points by the image's size (larger image = lower weight)

            var resDPS = new DataPointSet();
            resDPS.Points = result.ToArray();
            resDPS.Weight = bias;

            return resDPS;
        }

        public override DataPointSet GetDataPoints(LabeledImage[] images, ForestLabelSource LabelSourceMode, LabelWeightingMode LabelWeightMode)
        {
            var result = new DataPointSet();

            //get data points and set labels
            foreach (var img in images)
            {
                var currentDPS = GetDataPoints(img.Image);
                result += new DataPointSet(
                    currentDPS.Points.Copy(x => x.SetLabel(
                        ((LabelSourceMode == ForestLabelSource.ImageGlobal)?
                        img.Label.Index:                   //set a pixel's label to the one of its parent image
                        img.GetLabelOfPixel(x.X,x.Y))      //set a pixel's label to the one from the supplied map
                        )),
                    currentDPS.Weight
                    );
            }


            if (LabelWeightMode == LabelWeightingMode.FullForest)
            {

                //set data point weights as inverse label frequency = sum of all labels / sum of this label
                var labels = result.Points.Select(x => x.Label).Distinct().ToArray();

                var labelSums = new int[labels.Count()];

                for (int i = 0; i < labels.Count(); i++)
                {
                    labelSums[i] = result.Points.Where(x => x.Label == labels[i]).Count();
                }

                var totalLabelSum = labelSums.Sum();

                for (int i = 0; i < labels.Count(); i++)
                {
                    result.Points.Where(x => x.Label == labels[i]).ForEach(x => x.Weight = totalLabelSum / (double)labelSums[i]);
                }

            }

            return result;
        }

        public override DataPointSet GetDataPoints(LabeledPatch patch, TrainingParams parameters)
        {
            var pi = MatrixCache.GetMatrixFrom(patch.PixImage);

            var result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling(PixWinSize / 2.0); //ceiling cuts away too much in most cases

            int pointCounter = 0;

            for (int x = borderOffset; x < pi.SX - borderOffset; x += SamplingFrequency)
            {
                for (int y = borderOffset; y < pi.SY - borderOffset; y += SamplingFrequency)
                {
                    var newDP = new DataPoint(patch.PixImage, x, y,1.0, 
                        patch.MappingRule(parameters.SegmentationLabels,patch.SegmentationImage,x,y).Index);
                    result.Add(newDP);
                    pointCounter++;
                }
            }

            var bias = 1.0 / pointCounter;     //weigh the sample points by the image's size (larger image = lower weight)

            var resDPS = new DataPointSet();
            resDPS.Points = result.ToArray();
            resDPS.Weight = bias;

            return resDPS;
        }

        public override DataPointSet GetDataPoints(LabeledPatch[] labeledPatches, TrainingParams parameters)
        {
            var result = new DataPointSet();

            foreach (var patch in labeledPatches)
            {
                var currentDPS = GetDataPoints(patch, parameters);
                result += currentDPS;
            }

            return result;
        }
    }


    /// <summary>
    /// Gets random points within the usable area of the image
    ///  (= image with a border respecting the feature window).
    /// 
    /// NOTE: Not yet implemented.
    /// </summary>
    public class RandomPointSamplingProvider : ISamplingProvider
    {
        public int PixWinSize;
        public int SampleCount;

        public override void Init(int pixWindowSize, int samplingFrequency)
        {
            PixWinSize = pixWindowSize;
        }

        public override DataPointSet GetDataPoints(Image image)
        {
            var pi = MatrixCache.GetMatrixFrom(image.PixImage);
            var borderOffset = (int)Math.Ceiling(PixWinSize / 2.0);

            var result = new DataPoint[SampleCount];
            for (int i = 0; i < SampleCount; i++)
            {
                var x = Algo.Rand.Next(borderOffset, (int)pi.SX - borderOffset);
                var y = Algo.Rand.Next(borderOffset, (int)pi.SY - borderOffset);
                result[i] = new DataPoint(image.PixImage, x, y);
            }

            return new DataPointSet(result, 1.0);
        }

        public override DataPointSet GetDataPoints(LabeledImage[] labeledImages, ForestLabelSource LabelSourceMode, LabelWeightingMode LabelWeightMode)
        {
            throw new NotImplementedException();
        }

        public override DataPointSet GetDataPoints(LabeledPatch patch, TrainingParams parameters)
        {
            throw new NotImplementedException();
        }

        public override DataPointSet GetDataPoints(LabeledPatch[] labeledImages, TrainingParams parameters)
        {
            throw new NotImplementedException();
        }

    }

    #endregion



    #region Parameter Classes

    /// <summary>
    /// Parameter Class containing all parameters that need to stay the same throughout the system.
    /// </summary>
    public class TrainingParams
    {
        /// <summary>
        /// Specifies common default values for most parameters.
        /// </summary>
        /// <param name="treeCount">Number of Trees.</param>
        /// <param name="maxTreeDepth">Maximum depth of Trees.</param>
        /// <param name="trainingSubsetCountPerTree">Size of training image subset used for each Tree.</param>
        /// <param name="trainingImageSamplingWindow">Size of sampling window in pixels.</param>
        /// <param name="trainingWindowSamplingFrequency">Distance from one sampling window center to the next in pixels.</param>
        /// <param name="labels">List of all Labels.</param>
        /// <param name="maxFeatureCount">Maximum number of Features per tree.</param>
        /// <param name="featureType">Feature Type.</param>
        public TrainingParams(int treeCount, int maxTreeDepth,
            int trainingSubsetCountPerTree, int trainingImageSamplingWindow,
            int trainingWindowSamplingFrequency,
            Label[] labels,
            int maxFeatureCount = 999999999,
            FeatureType featureType = FeatureType.SelectRandom
            )
        {
            this.SamplingFrequency = (trainingWindowSamplingFrequency == -1) ? (int)Math.Ceiling(trainingImageSamplingWindow / 2.0) : trainingWindowSamplingFrequency;
            this.FeatureProviderFactory = new FeatureProviderFactory(featureType, trainingImageSamplingWindow);
            this.FeatureProviderFactory.SelectProvider(featureType, trainingImageSamplingWindow);
            this.SamplingProviderFactory = new SamplingProviderFactory(this.SamplingType, trainingImageSamplingWindow, this.SamplingFrequency);
            this.TreesCount = treeCount;
            this.MaxTreeDepth = maxTreeDepth;
            this.ImageSubsetCount = trainingSubsetCountPerTree;
            this.SamplingWindow = trainingImageSamplingWindow;
            this.MaxSampleCount = maxFeatureCount;
            this.FeatureType = featureType;
            this.Labels = labels;
            this.ClassesCount = Labels.Max(x => x.Index) + 1;
        }

        public string ForestName = "new forest";       //identifier of the forest, has no usage except for readability if saving to file
        public int ClassesCount;        //how many classes
        public int TreesCount;          //how many trees should the forest have
        public int MaxTreeDepth;        //maximum depth of one tree
        public int ImageSubsetCount;    //how many images should be randomly selected from the training set for each tree's training
        public int SamplingWindow;      //side length of the square window around a pixel to be sampled; half of this size is effectively the border around the image
        public int SamplingFrequency;   //distance between sampling window center pixels
        public int MaxSampleCount;      //limit the maximum number of samples for all images (selected randomly from all samples) -> set this to 99999999 for all samples
        public FeatureType FeatureType; //the type of feature that should be extracted using the feature providers
        public SamplingType SamplingType = SamplingType.RegularGrid;//mode of sampling
        public int RandomSamplingCount = 500;  //if sampling = random sampling, how many points?
        public FeatureProviderFactory FeatureProviderFactory;       //creates a new feature provider for each decision node in the trees to apply to a sample point (window); currently value of a random pixel, sum of two random pixels, absolute difference of two random pixels
        public SamplingProviderFactory SamplingProviderFactory;     //creates a new sample point provider which is currently applied to all pictures; currently sample a regular grid with stride, sample a number of random points
        public int ThresholdCandidateNumber = 16;    //how many random thresholds should be tested in a tree node to find the best one
        public double ThresholdInformationGainMinimum = 0.01d;    //break the tree node splitting if no threshold has a score better than this
        public ClassificationMode ClassificationMode = ClassificationMode.Semantic;    //what feature representation method to use; currently: standard representation by leaves only, semantic texton representation using the entire tree
        public bool ForcePassthrough = false;   //during forest generation, force each datapoint to reach a leaf (usually bad)
        public bool EnableGridSearch = false;         //the SVM tries out many values to find the optimal C (can take a long time)
        public Label[] Labels;                  //globally used list of Image Labels
        public Label[] SegmentationLabels;      //list of Segmentation Labels
        public SegmentationMappingRule MappingRule; //mapping rule of pixel to Segmentation Label
        public SegmentationColorizationRule ColorizationRule;   //mapping rule of Segmentation Label to pixel color
        public ForestLabelSource LabelSource = ForestLabelSource.ImageGlobal;   //where to get label information of each pixel from. global is intended for classification, pixelIndividual for segmentation.
        public LabelWeightingMode LabelWeightMode = LabelWeightingMode.LabelsOnly;    //How to weigh the labels? 
        public ClassificationMode PatchPredictionMode = ClassificationMode.LeafOnly;    //use entire tree or only leaf for soft prediction? -> currently almost no difference
    }

    /// <summary>
    /// File Path Object which is used for the output in a Test Series.
    /// </summary>
    public class FilePaths
    {
        public FilePaths(string workDir)
        {
            WorkDir = workDir;
            ForestFilePath = Path.Combine(workDir, "forest.json");
            Testsetpath1 = Path.Combine(workDir, "Testset1.json");
            Testsetpath2 = Path.Combine(workDir, "Testset2.json");
            Semantictestsetpath1 = Path.Combine(workDir, "Semantictestset1.json");
            Semantictestsetpath2 = Path.Combine(workDir, "Semantictestset2.json");
            Trainingsetpath = Path.Combine(workDir, "Trainingset.ds");
            Kernelsetpath = Path.Combine(workDir, "Kernel.ds");
            TrainingTextonsFilePath = Path.Combine(workDir, "TrainingTextons.json");
            TestTextonsFilePath = Path.Combine(workDir, "TestTextons.json");
        }

        public string WorkDir;
        public string ForestFilePath;
        public string Testsetpath1;
        public string Testsetpath2;
        public string Semantictestsetpath1;
        public string Semantictestsetpath2;
        public string Trainingsetpath;
        public string Kernelsetpath;
        public string TrainingTextonsFilePath;
        public string TestTextonsFilePath;
    }

    #endregion
}
