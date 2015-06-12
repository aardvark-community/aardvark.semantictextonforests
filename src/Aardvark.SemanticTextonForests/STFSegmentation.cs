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

    /// <summary>
    /// The Segmentation Forest operates on the output of a Aardvark.SemanticTextonForests.Forest and performs image segmentation.
    /// It looks similar in structure to the Forest, except that some methods are slightly different.
    /// </summary>
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
        /// Weights of all Labels that occur in this Forest. 
        /// </summary>
        public LabelDistribution LabelWeights;

        /// <summary>
        /// JSON Constructor.
        /// </summary>
        public SegmentationForest() { }

        /// <summary>
        /// Creates an empty Segmentation Forest.
        /// </summary>
        /// <param name="name">Friendly name of the Forest.</param>
        /// <param name="numberOfTrees">Number of empty Trees to be created in this Forest.</param>
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


        public DistributionImage PredictLabelDistribution(DistributionImage image, SegmentationParameters parameters)
        {
            return PredictLabelDistribution(image, parameters, null);
        }

        /// <summary>
        /// Performs Segmentation on a Distribution Image. The Distribution Image is split up into a regular grid of patches, and the 
        /// Label for each patch is predicted according to the Segmentation Forest. The resulting Label Distribution is weighted
        /// with the ILP. The ILP is the averaged distribution of predicted labels across the entire Image (see method GetILP() of Forest.)
        /// </summary>
        /// <param name="image">Input Distribution Image.</param>
        /// <param name="parameters">Trainin Parameters.</param>
        /// <param name="ILP">Image Level Prior of the according Forest. May be null if Segmentation
        /// Mode is anything else than WithILP.</param>
        /// <returns>A Distribution Image containing the predicted Label Distribution for each pixel.</returns>
        public DistributionImage PredictLabelDistribution(DistributionImage image, SegmentationParameters parameters, LabelDistribution ILP)
        {
            if(ILP == null && parameters.SegModel == SegmentationEvaluationModel.WithILP)
            {
                throw new InvalidOperationException("Can't call ILP segmentation when ILP is null.");
            }

            Report.BeginTimed(2, "Segmenting image.");
            var result = new DistributionImage(image.Image, image.numChannels);

            //split the image into data points, get a prediction for each, return a map with a soft classification for each pixel.

            //todo: move the sampling provider to the forest
            var baseDPS = Trees[0].SamplingProvider.GetDataPoints(image, parameters.SegmentatioSplitRatio).ToArray();
            var numClasses = baseDPS[0].DistributionImage.numChannels;

            //for each datapoint
            for (int i = 0; i < baseDPS.Length; i++)
            {
                var curDP = baseDPS[i];

                var curRes = new LabelDistribution(numClasses);

                //for each tree
                if (!(parameters.SegModel == SegmentationEvaluationModel.PatchPriorOnly))
                {
                    foreach (var tree in Trees)
                    {
                        LabelDistribution cDist;

                        //get the predicted distribution and add it to the result
                        if (parameters.PatchPredictionMode == ClassificationMode.LeafOnly)
                        {
                            cDist = tree.PredictLabels(curDP);
                        }
                        else //if (parameters.PatchPredictionMode == ClassificationMode.Semantic)
                        {
                            cDist = tree.PredictLabelsCumulative(curDP, numClasses);
                        }

                        cDist.Normalize();

                        curRes.AddDistribution(cDist);
                    }

                    //average and normalize
                    curRes.Scale(1.0 / NumTrees);
                    curRes.Normalize();

                    //apply the Segmentation Forest's Label weights
                    if (parameters.LabelWeightMode == LabelWeightingMode.LabelsOnly)
                    {
                        var weighDist = new LabelDistribution(this.LabelWeights.Distribution);
                        weighDist.Normalize();
                        curRes.Scale(weighDist);
                    }

                    curRes.Normalize();
                }

                //apply additional weighting according to the parameter setting
                if(parameters.SegModel == SegmentationEvaluationModel.SegmentationForestOnly)
                {
                    //no weighting
                }
                else if(parameters.SegModel == SegmentationEvaluationModel.WithPatchPrior || parameters.SegModel == SegmentationEvaluationModel.WithILP)
                {
                    //get estimated patch prior and multiply it onto the result
                    var pp = curDP.DistributionImage.GetWindowPrediction(curDP.X, curDP.Y, curDP.X + curDP.SX, curDP.Y + curDP.SY);
                    pp.Normalize();
                    curRes.Scale(pp);
                }
                else if(parameters.SegModel == SegmentationEvaluationModel.PatchPriorOnly)
                {
                    //only use the patch priors (local segmentation)
                    curRes = curDP.DistributionImage.GetWindowPrediction(curDP.X, curDP.Y, curDP.X + curDP.SX, curDP.Y + curDP.SY);
                }

                curRes.Normalize();

                //apply softened ILP
                if(parameters.SegModel == SegmentationEvaluationModel.WithILP)
                {
                    var ILPdist = new LabelDistribution(ILP.Distribution);
                    ILPdist.Soften(parameters.ILPSofteningExponent);
                    ILPdist.Normalize();
                    curRes.Scale(ILP);
                }

                curRes.Normalize();

                //set the current data point's prediction into the result map
                result.setRange(curDP.X, curDP.Y, curDP.X + curDP.SX, curDP.Y + curDP.SY, curRes);
            }

            Report.End(2);
            return result;
        }
    }

    /// <summary>
    /// A Tree of the Segmentation Forest. Similar to Aardvark.SemanticTextonForests.Tree
    /// </summary>
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

        /// <summary>
        /// Predicts the Label Distribution for one Data Point.
        /// </summary>
        /// <param name="dp">Input Segmentation Data Point.</param>
        /// <returns>The Labels that are predicted to belong to the Data Point.</returns>
        public LabelDistribution PredictLabels(SegmentationDataPoint dp)
        {
            return Root.PredictLabels(dp);
        }

        /// <summary>
        /// Predicts an averaged Label Distribution for one Data Point by summing up along the entire path of the Data Point.
        /// The resulting Distribution is more biased towards frequent Labels.
        /// </summary>
        /// <param name="dp"></param>
        /// <param name="numClasses"></param>
        /// <returns></returns>
        public LabelDistribution PredictLabelsCumulative(SegmentationDataPoint dp, int numClasses)
        {
            var baseDistribution = new LabelDistribution(numClasses);

            Root.GetDistributionCumulative(dp, baseDistribution);

            return baseDistribution;
        }

        /// <summary>
        /// to be deleted
        /// </summary>
        /// <param name="weights"></param>
        public void PushWeightsToLeaves(LabelDistribution weights)
        {
            Root.PushWeightsToLeaves(weights);
        }
    }

    /// <summary>
    /// A Node in a Segmentation Tree. Similar to Aardvark.SemanticTextonForests.Node
    /// </summary>
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

        /// <summary>
        /// Predicts the Label Distribution of one Data Point.
        /// </summary>
        /// <param name="dp"></param>
        /// <returns>The Label Distribution of the Leaf this Data Point reaches.</returns>
        public LabelDistribution PredictLabels(SegmentationDataPoint dp)
        {
            if (!this.IsLeaf)
            {
                if (Decider.Decide(dp) == Decision.Left)
                {
                    return LeftChild.PredictLabels(dp);
                }
                else
                {
                    return RightChild.PredictLabels(dp);
                }
            }
            else //break condition
            {
                return LabelDistribution;
            }
        }

        /// <summary>
        /// to be deleted
        /// </summary>
        /// <param name="weights"></param>
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
        /// Predicts the Label Distribution by performing a weighted sum over all the Distributions the Data Point encounters along its path.
        /// The weight of a Distribution increases the closer it is to a leaf.
        /// </summary>
        /// <param name="dataPoint"></param>
        /// <param name="baseDistribution"></param>
        /// <returns></returns>
        internal int GetDistributionCumulative(SegmentationDataPoint dataPoint, LabelDistribution baseDistribution)
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
    /// The Decider used in a Segmentation Node. Similar to Aardvark.SemanticTextonForests.Decider, except for a different feature provider.
    /// </summary>
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
            return (FeatureProvider.GetFeature(dataPoint) < DecisionThreshold) ? Decision.Left : Decision.Right;
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
            double minimumInformationGain, int maximumFeatureCount,
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
            bool inputIsSmall = currentDatapoints.Count <= 3;   //few data points remaining, no split makes sense -> leaf

            if (!inputIsEmpty && !inputIsSmall)
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

                    SplitDatasetWithThreshold(currentDatapoints, ref curThresh, maximumFeatureCount, out currentLeftSet, out currentRightSet, out currentLeftClassDist, out currentRightClassDist);
                    double leftEntr = CalcEntropy(currentLeftClassDist);
                    double rightEntr = CalcEntropy(currentRightClassDist);

                    var leftsum = currentLeftClassDist.GetLabelDistSum();
                    var rightsum = currentRightClassDist.GetLabelDistSum();

                    double leftWeight = (-1.0d) * leftsum / classDist.GetLabelDistSum();
                    double rightWeight = (-1.0d) * rightsum / classDist.GetLabelDistSum();
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

            bool isLeaf = (bestScore > -(1.0) * minimumInformationGain || inputIsEmpty || inputIsSmall);   //no images reached this node or not enough information gain => leaf

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
        private void SplitDatasetWithThreshold(List<SegmentationDataPoint> dps, ref double threshold, int maxSampleCount, out List<SegmentationDataPoint> leftSet, out List<SegmentationDataPoint> rightSet, out LabelDistribution leftDist, out LabelDistribution rightDist)
        {
            var leftList = new List<SegmentationDataPoint>();
            var rightList = new List<SegmentationDataPoint>();

            int targetFeatureCount = Math.Min(dps.Count, maxSampleCount);
            var actualDPS = dps.GetRandomSubset(targetFeatureCount);

            foreach (var dp in actualDPS)
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

            leftDist = new LabelDistribution(leftList, dps[0].DistributionImage.numChannels);
            rightDist = new LabelDistribution(rightList, dps[0].DistributionImage.numChannels);
            //leftDist = LabelDistribution.GetSegmentationPrediction(leftList, dps[0].DistributionImage.numChannels);
            //rightDist = LabelDistribution.GetSegmentationPrediction(rightList, dps[0].DistributionImage.numChannels);
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

    /// <summary>
    /// The Segmentation Feature Provider, which maps a Segmentation Data Point onto a numeric value.
    /// It is used similarly like in IFeatureProvider, except for the actual feature function:
    /// The Segmentation Feature Provider stores an offset vector, which is used to translate a Segmentation
    /// Data Point's pixel window. From the Distribution Image, one channel (which is not zero) is selected arbitrarily 
    /// and the averaged value of the pixel window in that channel is used as numeric result.
    /// </summary>
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
                    return;
                }
                else
                {
                    Channel = (Channel + i) % (NumClasses - 1);
                }
            }
        }

        /// <summary>
        /// Returns the numeric value associated with this SegmentationFeatureProvider for this 
        /// SegmentationDataPoint. The value is calculated by translating the Data Point by an
        /// offset vector, and returning (the average of) one image channel value.
        /// </summary>
        /// <param name="dp"></param>
        /// <returns></returns>
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
            if (outOfRange(minX, 0, sizeX-1 - dp.SX)) clamp(ref minX, 0, sizeX-1 - dp.SX);
            if (outOfRange(minY, 0, sizeY-1 - dp.SY)) clamp(ref minY, 0, sizeY-1 - dp.SY);
            if (outOfRange(maxX, dp.SX, sizeX-1)) clamp(ref maxX, dp.SX, sizeX-1);
            if (outOfRange(maxY, dp.SY, sizeY-1)) clamp(ref maxY, dp.SY, sizeY-1);

            var dist = dp.DistributionImage.GetWindowPrediction(minX, minY, maxX, maxY);

            return dist.Distribution[Channel];
        }

        /// <summary>
        /// is the value out of range?
        /// </summary>
        /// <param name="val"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        internal bool outOfRange(int val, int min, int max)
        {
            return val < min || val > max;
        }

        /// <summary>
        /// clamp the value within two values
        /// </summary>
        /// <param name="val"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
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


    /// <summary>
    /// The Sampling Provider used in the Segmentation Workflow. Similar to ISamplingProvider, except that it returns SegmentationDataPoints
    /// instead of DataPoints.
    /// </summary>
    public class SegmentationSamplingProvider
    {

        /// <summary>
        /// Instantiate new provider. Currently, there are no parameters.
        /// </summary>
        public SegmentationSamplingProvider() { }
        
        /// <summary>
        /// Returns a list of SegmentationDataPoints extracted from a Distribution image, using regular sampling with a 
        /// rectangular pixel window.
        /// </summary>
        /// <param name="image">Input image.</param>
        /// <param name="pixWinSizeX">Horizontal Pixel Window size.</param>
        /// <param name="pixWinSizeY">Vertical Pixel Window size.</param>
        /// <returns>Segmentation Data Points extracted from the input image.</returns>
        public List<SegmentationDataPoint> GetDataPoints(DistributionImage image, int pixWinSizeX, int pixWinSizeY)
        {
            var result = new List<SegmentationDataPoint>();

            int pointCounter = 0;

            //regularly place the pixel window end-to-end across the image and add the according data point to the result.
            for (int x = 0; x < image.DistributionMap.Size.X - pixWinSizeX; x += pixWinSizeX)
            {
                for (int y = 0; y < image.DistributionMap.Size.Y - pixWinSizeY; y += pixWinSizeY)
                {
                    var newDP = new SegmentationDataPoint(image, x, y, pixWinSizeX, pixWinSizeY);
                    result.Add(newDP);
                    pointCounter++;
                }
            }
            return result;
        }

        /// <summary>
        /// Returns Data Points of an array of images. 
        /// </summary>
        /// <param name="image"></param>
        /// <param name="pixWinSizeX"></param>
        /// <param name="pixWinSizeY"></param>
        /// <returns></returns>
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
        /// Returns a list of SegmentationDataPoints extracted from an array of Distribution Images. Calculates the sampling window from a ratio
        /// of one part to the whole image.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="ratio">One divided by the number of desired parts.</param>
        /// <returns></returns>
        public List<SegmentationDataPoint> GetDataPoints(DistributionImage[] image, double ratio, LabelWeightingMode LabelWeightMode)
        {
            var result = new List<SegmentationDataPoint>();

            foreach (var img in image)
            {
                result.AddRange(GetDataPoints(img, ratio));
            }
            if (LabelWeightMode == LabelWeightingMode.FullForest)
            {
                //set weight by inverse label frequency
                var labels = result.Select(x => x.Label).Distinct().ToArray();
                var labelSums = new int[labels.Count()];
                for (int i = 0; i < labels.Count(); i++)
                {
                    labelSums[i] = result.Where(x => x.Label == labels[i]).Count();
                }
                var totalLabelSum = labelSums.Sum();
                for (int i = 0; i < labels.Count(); i++)
                {
                    result.Where(x => x.Label == labels[i]).ForEach(x => x.Weight = totalLabelSum / (double)labelSums[i]);
                }
            }
            return result;
        }

        /// <summary>
        /// Same for one image.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="ratio"></param>
        /// <returns></returns>
        public List<SegmentationDataPoint> GetDataPoints(DistributionImage image, double ratio)
        {
            var segmentsX = (int)Math.Floor(image.DistributionMap.Size.X * ratio);
            var segmentsY = (int)Math.Floor(image.DistributionMap.Size.Y * ratio);
            return GetDataPoints(image, segmentsX, segmentsY);
        }
    }

    /// <summary>
    /// The Data Point used in the Segmentation workflow. This Datapoint is used in a similar fashion to Aardvark.SemanticTextonForests.DataPoint.
    /// However, this SegmentationDataPoint has no pixel location, but the coordinates of a rectangular window. It is associated with one 
    /// DistributionImage, which in turn is associated with a LabeledImage. 
    /// </summary>
    public class SegmentationDataPoint
    {
        public readonly DistributionImage DistributionImage;

        //coordinates of the rectangular window
        public readonly int X;
        public readonly int Y;
        public readonly int SX, SY;

        /// <summary>
        /// This Data Point's weight.
        /// </summary>
        public double Weight = 1.0;

        private bool labelIsReady = false;
        private int myLabel = 0;

        /// <summary>
        /// This Data Point's label. It is calculated by retrieving the associated DistributionImage's LabeledImage's 
        /// Label Distribution for this Point, averaging over the rectangular window, and picking the most likely one.
        /// TODO: This could be a Label Distribution instead of only an individual Label.
        /// </summary>
        public int Label
        {
            get
            {
                if (!labelIsReady)
                {
                    myLabel = DistributionImage.Image.GetLabelOfRegion(X, Y, X + SX, Y + SY);
                    labelIsReady = true;
                    return myLabel;
                }
                else
                {
                    return myLabel;
                }
            }
        }

        /// <summary>
        /// Creates a new Data Point.
        /// </summary>
        /// <param name="pi">Parent Distribution Image.</param>
        /// <param name="x">top left x coordinate</param>
        /// <param name="y">top left y coordinate</param>
        /// <param name="sx">x size of window</param>
        /// <param name="sy">y size of window</param>
        public SegmentationDataPoint(DistributionImage pi, int x, int y, int sx, int sy)
        {
            if (x < 0 || y < 0 || x >= pi.DistributionMap.Size.X || y >= pi.DistributionMap.Size.Y) throw new IndexOutOfRangeException();

            DistributionImage = pi;
            X = x; Y = y;
            SX = sx; SY = sy;
        }
    }

    /// <summary>
    /// Static class which contains the training and evaluation algorithms of the Segmentation Forest.
    /// </summary>
    public static class SegmentationAlgo
    {
        private static int NodeIndexCounter = 0;
        public static int TreeCounter = 0;
        private static int NodeProgressCounter = 0;

        /// <summary>
        /// Trains the Segmentation Forest with the supplied Distribution Images.
        /// </summary>
        /// <param name="forest">Segmentation Forest</param>
        /// <param name="trainingImages">Training Images</param>
        /// <param name="parameters">Training Parameters</param>
        public static void Train(this SegmentationForest forest, DistributionImage[] trainingImages, SegmentationParameters parameters)
        {
            NodeIndexCounter = -1;

            Report.BeginTimed(0, $"Training SegmentationForest with {trainingImages.Length} DistributionImages.");

            TreeCounter = 0;

            Parallel.ForEach(forest.Trees, tree =>
            //foreach (var tree in forest.Trees)
            {
                //get a random subset of the actual training set.
                var currentSubset = trainingImages.GetRandomSubset(parameters.TrainingSubsetPerTree);

                Report.BeginTimed(1, $"Training tree {tree.Index + 1} of {forest.Trees.Length}.");

                //train the tree with the subset.
                tree.Train(currentSubset, parameters);

                Report.Line(2, "Finished training tree with " + NodeProgressCounter + " nodes.");

                Report.End(1);
            }
            );

            //Calculate the Label weightings for this Data Set and store them in the Forest.
            var dps = forest.Trees[0].SamplingProvider.GetDataPoints(trainingImages, parameters.SegmentatioSplitRatio, parameters.LabelWeightMode);

            forest.LabelWeights = dps.GetLabelWeights();

            forest.NumNodes = forest.Trees.Sum(x => x.NumNodes);

            Report.End(0);
        }

        /// <summary>
        /// Calculates a Label weighting for a set of Data Points. The weight is equal to the inverse Label Frequency, i.e.
        /// weight(label) = sum(label) / sum(all labels)
        /// </summary>
        /// <param name="Points">Complete Data Point set (Training Set).</param>
        /// <returns>Inverse Label frequencies within the input set.</returns>
        internal static LabelDistribution GetLabelWeights(this List<SegmentationDataPoint> Points)
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
                weights[i] = totalLabelSum / (double)(double)((labelSums[i] <= 0) ? (1.0) : (labelSums[i]));
            }

            var weightsDist = new LabelDistribution(weights);

            //weightsDist.Normalize();

            return weightsDist;
        }

        /// <summary>
        /// Trains a Tree with a set of DistributionImages.
        /// </summary>
        /// <param name="tree"></param>
        /// <param name="trainingImages"></param>
        /// <param name="parameters"></param>
        private static void Train(this SegmentationTree tree, DistributionImage[] trainingImages, SegmentationParameters parameters)
        {
            var nodeCounterObject = new NodeCountObject();

            //get a new Sampling Provider for this Tree
            tree.SamplingProvider = new SegmentationSamplingProvider();
            //extract Data Points from the training Images using the Sampling Provider
            var baseDPS = tree.SamplingProvider.GetDataPoints(trainingImages, parameters.SegmentatioSplitRatio, parameters.LabelWeightMode);

            baseDPS = baseDPS.Where(x => x.Label != 0).ToList();

            var baseClassDist = new LabelDistribution(baseDPS, baseDPS[0].DistributionImage.numChannels);
            //var baseClassDist = LabelDistribution.GetSegmentationPrediction(baseDPS, baseDPS[0].DistributionImage.numChannels);

            Report.Line(2, $"Tree Training datapoint set size: {baseDPS.Count}");

            //recursively train the tree starting from the Root
            tree.Root.TrainRecursive(null, baseDPS, parameters, 0, baseClassDist, nodeCounterObject);
            tree.NumNodes = nodeCounterObject.Counter;

            NodeProgressCounter = nodeCounterObject.Counter;
        }

        /// <summary>
        /// Train a Node with a set of Data Points recursively until every branch ends in a Leaf.
        /// </summary>
        /// <param name="node"></param>
        /// <param name="parent"></param>
        /// <param name="currentData"></param>
        /// <param name="parameters"></param>
        /// <param name="depth"></param>
        /// <param name="currentLabelDist"></param>
        /// <param name="currentNodeCounter"></param>
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
                parameters.EntropyLimit, parameters.MaximumFeatureCount, out leftRemaining, out rightRemaining, out leftClassDist, out rightClassDist);

            //node.LabelDistribution.Normalize();

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


    /// <summary>
    /// Container for all Parameters needed for the Segmentation Forest and the segmentation workflow.
    /// The parameters have the same explanation as in Aardvark.SemanticTextonForests.TrainingParameters,
    /// except where noted.
    /// </summary>
    public class SegmentationParameters
    {
        /// <summary>
        /// Create default values. Supply the training set to calculate maximum Feature offset vector as average size * 0.5.
        /// </summary>
        public SegmentationParameters(DistributionImage[] trainingImages)
        {
            TrainingSubsetPerTree = trainingImages.Length / 2;

            if (trainingImages != null && !trainingImages.IsEmpty())
            {
                //get a feature offset vector about a third the avg size of an image
                trainingImages.GetRandomSubset(trainingImages.Length / 10).ForEach((el) =>
                {
                    var coX = el.DistributionMap.Size.X * 0.45;
                    var coY = el.DistributionMap.Size.Y * 0.45;
                    if (coX > MaximumFeatureOffsetX) MaximumFeatureOffsetX = (int)coX;
                    if (coY > MaximumFeatureOffsetY) MaximumFeatureOffsetY = (int)coY;
                });
            }
        }
    

        public int NumberOfTrees = 8;
        public int TrainingSubsetPerTree = 10;
        public double SegmentatioSplitRatio = 0.02;     //Segmentation: size of patches is a fraction of the original size
        public int MaximumFeatureOffsetX = 20;          //Segmentation Feature Provider: maximum length of the offset vector (constructor sets it to about half the image size)
        public int MaximumFeatureOffsetY = 20;
        public int ThresholdCandidateNumber = 20;
        public int MaxTreeDepth = 12;
        public double EntropyLimit = 0.1;
        public int MaximumFeatureCount = 250000;
        public SegmentationEvaluationModel SegModel = SegmentationEvaluationModel.WithPatchPrior;
        public LabelWeightingMode LabelWeightMode = LabelWeightingMode.LabelsOnly;
        public ClassificationMode PatchPredictionMode = ClassificationMode.LeafOnly;
        public double ILPSofteningExponent = 1.5;       //Segmentation Evaluation with ILP: softens the ILP with this exponent
    }

    /// <summary>
    /// Switch between the result calculation models (suggested in the paper). Does not contain all of them yet.
    /// </summary>
    public enum SegmentationEvaluationModel
    {
        SegmentationForestOnly,         //only take the prediction of the segmenatation (second) forest
        WithPatchPrior,                 //combine ^ and v 
        PatchPriorOnly,                 //only take the prediction of the first forest
        WithILP                         //full model 
    }
}
