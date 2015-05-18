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
        public abstract void Init(int pixelWindowSize);

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
                    double leftWeight = (-1.0d) * currentLeftClassDist.GetClassDistSum() / classDist.GetClassDistSum();
                    double rightWeight = (-1.0d) * currentRightClassDist.GetClassDistSum() / classDist.GetClassDistSum();
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
                var px = dist.GetClassProbability(cl);
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
        public LabelDistribution ClassDistribution;
        /// <summary>
        /// This node's global index in the forest.
        /// </summary>
        public int GlobalIndex = -1;

        public void GetClassDecisionRecursive(DataPoint dataPoint, List<TextonNode> currentList, TrainingParams parameters)
        {
            switch (parameters.ClassificationMode)
            {
                case ClassificationMode.Semantic:

                    var rt = new TextonNode();
                    rt.Index = GlobalIndex;
                    rt.Level = DistanceFromRoot;
                    rt.Value = 1;
                    currentList.Add(rt);

                    //descend left or right, or return if leaf
                    if (!this.IsLeaf)
                    {
                        if (Decider.Decide(dataPoint) == Decision.Left)   
                        {
                            LeftChild.GetClassDecisionRecursive(dataPoint, currentList, parameters);
                        }
                        else            
                        {
                            RightChild.GetClassDecisionRecursive(dataPoint, currentList, parameters);
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
                            LeftChild.GetClassDecisionRecursive(dataPoint, currentList, parameters);
                        }
                        else
                        {
                            RightChild.GetClassDecisionRecursive(dataPoint, currentList, parameters);
                        }
                    }
                    else            //we are at a leaf, take this class distribution as result
                    {
                        var result = new TextonNode();
                        result.Index = GlobalIndex;
                        result.Level = DistanceFromRoot;
                        result.Value = 1;
                        var resList = new List<TextonNode>();
                        resList.Add(result);
                        return;
                    }
                    return;

                default:
                    return;
            }
        }

        //every node adds 0 to the histogram (=initialize the histogram parameters)
        public void InitializeEmpty(List<TextonNode> currentList)
        {
            var rt = new TextonNode();
            rt.Index = GlobalIndex;
            rt.Level = DistanceFromRoot;
            rt.Value = 0;
            currentList.Add(rt);

            //descend left or right, or return if leaf
            if (!this.IsLeaf)
            {
                LeftChild.InitializeEmpty(currentList);
                RightChild.InitializeEmpty(currentList);
            }
        }
    }

    public class Tree
    {
        public Node Root;
        public int Index = -1;   //this tree's index within the forest, is set by the forest during initialization
        public int NumNodes = 0;    //how many nodes does this tree have in total

        public ISamplingProvider SamplingProvider;

        public Tree()
        {
            Root = new Node();
            Root.GlobalIndex = this.Index;
        }

        public List<TextonNode> GetClassDecision(DataPointSet dp, TrainingParams parameters)
        {
            var result = new List<TextonNode>();

            foreach (var point in dp.Points)
            {
                var cumulativeList = new List<TextonNode>();
                Root.GetClassDecisionRecursive(point, cumulativeList, parameters);
                foreach (var el in cumulativeList)        //this is redundant with initializeEmpty -> todo
                {
                    el.TreeIndex = this.Index;
                }
                result.AddRange(cumulativeList);
            }

            return result;
        }

        public void GetEmptyHistogram(List<TextonNode> currentList)
        {
            var cumulativeList = new List<TextonNode>();
            Root.InitializeEmpty(cumulativeList);
            foreach (var el in cumulativeList)
            {
                el.TreeIndex = this.Index;
            }
            currentList.AddRange(cumulativeList);
        }
    }

    public class Forest
    {
        public Tree[] Trees;
        public string Name { get; }
        public int NumTrees { get; }

        public int NumNodes = -1;

        public Forest() { }

        public Forest(string name, int numberOfTrees)
        {
            Name = name;
            NumTrees = numberOfTrees;

            InitializeEmptyForest();
        }

        private void InitializeEmptyForest()
        {
            Trees = new Tree[NumTrees].SetByIndex(i => new Tree() { Index = i });
        }

        public Textonization GetTextonRepresentation(Image img, TrainingParams parameters)
        {
            if (NumNodes <= -1)  //this part is deprecated
            {
                //NumNodes = Trees.Sum(x => x.NumNodes);
                throw new InvalidOperationException();
            }

            var result = new Textonization();
            result.InitializeEmpty(NumNodes);

            var basicNodes = new List<TextonNode>();

            Algo.TreeCounter = 0;

            foreach (var tree in Trees)    //for each tree, get a textonization of the data set and sum up the result
            {
                Algo.TreeCounter++;

                tree.GetEmptyHistogram(basicNodes);

                var imageSamples = tree.SamplingProvider.GetDataPoints(img);

                var curTex = tree.GetClassDecision(imageSamples, parameters);

                result.AddNodes(curTex);

            }

            result.AddNodes(basicNodes);    //we can add all empty nodes after calculation because it simply increments all nodes by 0 (no change) while initializing unrepresented nodes

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
        /// 
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

    //a class distribution, containing a histogram over all classes and their respective values.
    public class LabelDistribution
    {
        /// <summary>
        /// Weighted histogram value for each label.
        /// </summary>
        public double[] Histogram;

        /// <summary>
        /// Don't use this constructor, JSON only.
        /// </summary>
        public LabelDistribution() { }

        //initializes all classes with a count of 0
        public LabelDistribution(Label[] allLabels)
        {
            Histogram = new double[allLabels.Length]; ;
            for (int i = 0; i < allLabels.Length; i++) //allLabels must have a sequence of indices [0-n]
            {
                Histogram[i] = 0;
            }
        }

        //initialize classes and add the data points
        public LabelDistribution(Label[] allLabels, DataPointSet dps, TrainingParams parameters)
            : this(allLabels)
        {
            AddDatapoints(dps, parameters);
        }

        //add one data point to histogram
        public void AddDP(DataPoint dp, TrainingParams parameters)
        {
            if (dp.Label == -2) return;
            AddClNum(parameters.Labels[dp.Label], 1.0);
        }

        //add one histogram entry
        public void AddClNum(Label cl, double num)
        {
            Histogram[cl.Index] = Histogram[cl.Index] + num;
        }

        //add all data points to histogram
        public void AddDatapoints(DataPointSet dps, TrainingParams parameters)
        {
            foreach (var dp in dps.Points)
            {
                this.AddDP(dp, parameters);
            }
        }

        //returns the proportion of the elements of this class to the number of all elements in this distribution
        public double GetClassProbability(Label label)
        {
            var sum = Histogram.Sum();

            if (sum == 0)
            {
                return 0;
            }

            var prob = Histogram[label.Index] / sum;

            return prob;
        }

        //returns sum of histogram values
        public double GetClassDistSum()
        {
            return Histogram.Sum();
        }

        //normalize histogram
        public void Normalize()
        {
            var sum = Histogram.Sum();

            if (sum == 0)
            {
                return;
            }

            for (int i = 0; i < Histogram.Length; i++)
            {
                Histogram[i] = Histogram[i] / sum;
            }
        }
    }
    #endregion

    #region Images and I/O

    //the textonized form of a pixel region as returned by a STForest
    public class Textonization
    {
        public double[] Values; //old format - to be removed
        public TextonNode[] Nodes;  //new format
        public int Length;

        public Textonization()
        {

        }

        public void InitializeEmpty(int numNodes)
        {
            Length = numNodes;
            Nodes = new TextonNode[numNodes];

            for (int i = 0; i < numNodes; i++)
            {
                Nodes[i] = new TextonNode() { Index = i, Level = 0, Value = 0 };
            }
        }

        public void AddValues(double[] featureValues)
        {
            this.Values = featureValues;
            this.Length = featureValues.Length;
        }

        public void SetNodes(TextonNode[] featureNodes)
        {
            this.Nodes = featureNodes;
            this.Length = featureNodes.Length;
        }

        public void AddNodes(List<TextonNode> featureNodes)
        {
            foreach (var node in featureNodes)
            {
                var localNode = this.Nodes[node.Index];

                localNode.Level = node.Level;
                localNode.TreeIndex = node.TreeIndex;
                localNode.Value += node.Value;
            }
        }

        public static Textonization operator +(Textonization current, Textonization other)     //adds two textonizations. must have same length and same node indices (=be from the same forest)
        {
            var result = new Textonization();

            result.Length = current.Length;

            for (int i = 0; i < current.Length; i++)
            {
                var curNode = current.Nodes[i];
                var otherNode = other.Nodes.First(t => t.Index == curNode.Index);

                var res = new TextonNode();
                res.Index = curNode.Index;
                res.Level = curNode.Level;
                res.Value = curNode.Value + otherNode.Value;
                result.Nodes[i] = res;
            }

            return result;
        }

    }

    public class TextonNode
    {
        public int Index = -1; //the tree node's global identifier
        public int TreeIndex = -1; //the index of the tree this node belongs to
        public int Level = -1; //the level of this node in the tree
        public double Value = 0;   //"histogram" value
    }

    /// <summary>
    /// Represents an image which is loaded from disk
    /// </summary>
    public class Image
    {
        public string ImagePath;

        //the image will be loaded into memory on first use
        private PixImage<byte> pImage;
        private bool isLoaded = false;

        //don't use, JSON only
        public Image()
        {

        }

        //Creates a new image without loading it into memory
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
    /// Represents one rectangular patch of an image
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
    /// STImage with added class label, used for training and testing.
    /// </summary>
    public class LabeledImage
    {
        public Image Image { get; }
        public Label ClassLabel { get; }

        //this value can be changed if needed different image bias during training
        public double TrainingBias = 1.0f;

        //don't use, JSON only
        public LabeledImage() { }

        //creates a new image from filename
        public LabeledImage(string imageFilename, Label label)
        {
            Image = new Image(imageFilename);
            ClassLabel = label;
        }
    }

    public class TextonizedLabeledImage
    {
        public LabeledImage Image { get; }
        public Textonization Textonization { get; }

        public Label Label => Image.ClassLabel;

        //don't use, JSON only
        public TextonizedLabeledImage() { }

        //copy constructor
        public TextonizedLabeledImage(LabeledImage image, Textonization textonization)
        {
            Image = image;
            Textonization = textonization;
        }
    }


    #endregion

    #region Parameter Classes

    public class TrainingParams
    {

        public TrainingParams(int treeCount, int maxTreeDepth,
            int trainingSubsetCountPerTree, int trainingImageSamplingWindow,
            Label[] labels,
            int maxFeatureCount = 999999999,
            FeatureType featureType = FeatureType.SelectRandom
            )
        {
            this.FeatureProviderFactory = new FeatureProviderFactory();
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

        //todo: definitely parse this from a text file or so

        public Label[] Labels;
    }

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
