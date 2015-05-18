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
        public readonly Image Image;
        public readonly int X;
        public readonly int Y;

        /// <summary>
        /// Scaling weight of this data point.
        /// </summary>
        public readonly double Weight;

        /// <summary>
        /// Index of this data point's label. Arbitrary value if unknown.
        /// </summary>
        public readonly int Label;

        public DataPoint(Image image, int x, int y, double weight = 1.0, int label = -2)
        {
            if (image == null) throw new ArgumentNullException();
            if (x < 0 || y < 0 || x >= image.PixImage.Size.X || y >= image.PixImage.Size.Y) throw new IndexOutOfRangeException();

            Image = image;
            X = x; Y = y;
            Weight = weight;
            Label = label;
        }

        /// <summary>
        /// Returns a new DataPoint with given label set.
        /// </summary>
        public DataPoint SetLabel(int label)
        {
            return new DataPoint(Image, X, Y, Weight, label);
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
        public abstract void Init(int pixWindowSize);
        /// <summary>
        /// Get a set of data points from a given image.
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <returns></returns>
        public abstract DataPointSet GetDataPoints(Image image);
        /// <summary>
        /// Get a collected set of data points from an array of images.
        /// </summary>
        /// <param name="labeledImages">Input image array.</param>
        /// <returns></returns>
        public abstract DataPointSet GetDataPoints(LabeledImage[] labeledImages);
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
        private void SplitDatasetWithThreshold(DataPointSet dps, double threshold, TrainingParams parameters, out DataPointSet leftSet, out DataPointSet rightSet, out LabelDistribution leftDist, out LabelDistribution rightDist)
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

            leftDist = new LabelDistribution(parameters.Labels.ToArray(), leftSet, parameters);
            rightDist = new LabelDistribution(parameters.Labels.ToArray(), rightSet, parameters);
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
        public LabelDistribution ClassDistribution;
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

        /// <summary>
        /// Adds one data point to the distribution.
        /// </summary>
        /// <param name="dp">Input Data Point.</param>
        /// <param name="parameters">Parameters Object.</param>
        public void AddDP(DataPoint dp, TrainingParams parameters)
        {
            AddClNum(parameters.Labels[dp.Label], 1.0);
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

            isLoaded = true;
        }
    }

    /// <summary>
    /// Represents one rectangular patch of an image.
    /// </summary>
    public class ImagePatch
    {
        public Image Image;

        //image coordinates of the rectangle this patch represents
        //top left pixel
        public int X = -1;
        public int Y = -1;
        //rectangle size in pixels
        public int SX = -1;
        public int SY = -1;

        public ImagePatch(Image parentImage, int X, int Y, int SX, int SY)
        {
            Image = parentImage;
            this.X = X;
            this.Y = Y;
            this.SX = SX;
            this.SY = SY;
        }

        //loading function TODO:
        //int actualSizeX = Math.Min(SX, pImage.Size.X);
        //int actualSizeY = Math.Min(SY, pImage.Size.Y);

        //var pVol = pImage.Volume;
        //var subVol = pVol.SubVolume(new V3i(X, Y, 0), new V3i(actualSizeX, actualSizeY, 3));

        //var newImageVol = subVol.ToImage();

        //var newImage = new PixImage<byte>(Col.Format.RGB, newImageVol);

        //pImage = newImage;
    }

    /// <summary>
    /// Image with added Label, used for training and testing.
    /// </summary>
    public class LabeledImage
    {
        public Image Image { get; }
        public Label ClassLabel { get; }

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
            ClassLabel = label;
        }
    }

    /// <summary>
    /// Labeled Image with added Textonization. If Label is unknown, it can be set to an arbitrary value.
    /// </summary>
    public class TextonizedLabeledImage
    {
        public LabeledImage Image { get; }
        public Textonization Textonization { get; }

        public Label Label => Image.ClassLabel;

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

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().ToDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().ToDouble();

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

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);
            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().ToDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().ToDouble();

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

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);

            var sample = pi[point.PixelCoords + PixelOffset].ToGrayByte().ToDouble();

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

            var pi = MatrixCache.GetMatrixFrom(point.Image.PixImage);

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().ToDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().ToDouble();

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


        public void SelectProvider(SamplingType samplingType, int pixelWindowSize)
        {
            this.PixelWindowSize = pixelWindowSize;
            this.SamplingType = samplingType;
            this.RandomSampleCount = 500;
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

    /// <summary>
    /// Gets a regular grid starting from the top left and continuing as long as there are pixels left.
    /// </summary>
    public class RegularGridSamplingProvider : ISamplingProvider
    {
        public int PixWinSize;

        public override void Init(int pixWindowSize)
        {
            PixWinSize = pixWindowSize;
        }

        public override DataPointSet GetDataPoints(Image image)
        {
            var pi = MatrixCache.GetMatrixFrom(image.PixImage);

            var result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling(PixWinSize / 2.0); //ceiling cuts away too much in most cases

            int pointCounter = 0;

            for (int x = borderOffset; x < pi.SX - borderOffset; x += PixWinSize)
            {
                for (int y = borderOffset; y < pi.SY - borderOffset; y += PixWinSize)
                {
                    var newDP = new DataPoint(image, x, y);
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

        public override DataPointSet GetDataPoints(LabeledImage[] images)
        {
            var result = new DataPointSet();

            foreach (var img in images)
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

        public override void Init(int pixWindowSize)
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
        /// <param name="labels">List of all Labels.</param>
        /// <param name="maxFeatureCount">Maximum number of Features per tree.</param>
        /// <param name="featureType">Feature Type.</param>
        public TrainingParams(int treeCount, int maxTreeDepth,
            int trainingSubsetCountPerTree, int trainingImageSamplingWindow,
            Label[] labels,
            int maxFeatureCount = 999999999,
            FeatureType featureType = FeatureType.SelectRandom
            )
        {
            this.FeatureProviderFactory = new FeatureProviderFactory(featureType, trainingImageSamplingWindow);
            this.FeatureProviderFactory.SelectProvider(featureType, trainingImageSamplingWindow);
            this.SamplingProviderFactory = new SamplingProviderFactory();
            this.SamplingProviderFactory.SelectProvider(this.SamplingType, trainingImageSamplingWindow);
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
        public Label[] Labels;                  //globally used list of all Labels
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
