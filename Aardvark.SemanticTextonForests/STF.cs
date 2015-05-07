using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Aardvark.Base;
using Newtonsoft.Json;

namespace Aardvark.SemanticTextonForests
{
    #region Semantic Texton Forest

    public class DataPoint
    {
        public STImage Image;
        public int X;
        public int Y;
        public double PointWeight;
        public int label = -2;      //-2 if unknown label; label index else (we take -2 because libsvm sometimes likes to assign -1 to classes)

        [JsonIgnore]
        public V2i PixelCoords
        {
            get { return new V2i(X, Y); }
            set { X = value.X; Y = value.Y; }
        }
    }

    public class DataPointSet
    {
        public DataPoint[] DPSet;
        public double SetWeight;        //<-- this is not currently used, but needs to be implemented. TODO

        public DataPointSet()
        {
            DPSet = new DataPoint[] { };
        }

        public static DataPointSet operator+(DataPointSet current, DataPointSet other)
        {
            var result = new DataPointSet();

            var resultList = new List<DataPoint>();
            resultList.AddRange(current.DPSet);
            resultList.AddRange(other.DPSet);

            result.DPSet = resultList.ToArray();
            result.SetWeight = current.SetWeight + other.SetWeight;

            return result;
        }
    }

    public class STFeature
    {
        public double Value;
    }

    public abstract class IFeatureProvider
    {
        public abstract void Init(int pixelWindowSize);

        public abstract STFeature getFeature(DataPoint point);

        public STFeature[] getArrayOfFeatures(DataPointSet points)
        {
            List<STFeature> result = new List<STFeature>();

            foreach(var point in points.DPSet)
            {
                result.Add(this.getFeature(point));
            }

            return result.ToArray();
        }
    }

    public abstract class ISamplingProvider
    {
        public abstract void init(int pixWindowSize);
        public abstract DataPointSet getDataPoints(STImage image);
        public abstract DataPointSet getDataPoints(STLabelledImage[] labelledImages);
    }

    public class Decider
    {
        public IFeatureProvider FeatureProvider;
        public ISamplingProvider SamplingProvider;
        public double DecisionThreshold;
        public double Certainty;

        //public bool Decide(STImage img)
        //{

        //    var datapoints = SamplingProvider.getDataPoints(img);
        //    var features = FeatureProvider.getArrayOfFeatures(datapoints);

        //    int leftScore = 0;
        //    int rightScore = 0;
        //    foreach (var feature in features)
        //    {
        //        var value = feature.Value;

        //        //if the feature is smaller than the threshold, vote to put it into the left set, else right set
        //        if (value < DecisionThreshold)
        //        {
        //            leftScore = leftScore + 1;
        //        }
        //        else
        //        {
        //            rightScore = rightScore + 1;
        //        }
        //    }

            

        //    if (leftScore > rightScore)
        //    {
        //        Report.Line(4, "Decided left at threshold " + DecisionThreshold + " #features = " + features.Length);
        //        return true;
        //    }
        //    else
        //    {
        //        Report.Line(4, "Decided right at threshold " + DecisionThreshold + " #features = " + features.Length);
        //        return false;
        //    }
        //}

        //true = left, false = right
        public bool Decide(DataPoint dataPoint)
        {

            //var datapoints = SamplingProvider.getDataPoints(img);
            var feature = FeatureProvider.getFeature(dataPoint);
            var value = feature.Value;


            //int leftScore = 0;
            //int rightScore = 0;
            //foreach (var feature in features)
            //{
            //    var value = feature.Value;

            //    //if the feature is smaller than the threshold, vote to put it into the left set, else right set
            //    if (value < DecisionThreshold)
            //    {
            //        leftScore = leftScore + 1;
            //    }
            //    else
            //    {
            //        rightScore = rightScore + 1;
            //    }
            //}



            if (value < DecisionThreshold)
            {
                Report.Line(4, "Decided left at threshold " + DecisionThreshold);
                return true;
            }
            else
            {
                Report.Line(4, "Decided right at threshold " + DecisionThreshold);
                return false;
            }
        }

        //returns true if this node should be a leaf and leaves the out params empty; false else and fills the out params with the split values
        //public STFAlgo.DeciderTrainingResult InitializeDecision(STLabelledImage[] images, ClassDistribution classDist, TrainingParams parameters, out STLabelledImage[] leftRemaining, out STLabelledImage[] rightRemaining, out ClassDistribution leftClassDist, out ClassDistribution rightClassDist) 
        //{
        //    //get a bunch of candidates for decision using the supplied featureProvider and samplingProvider, select the best one based on entropy, return either the 
        //    //left/right split subsets and false, or true if this node should be a leaf

        //    var threshCandidates = new double[parameters.thresholdCandidateNumber];
        //    for (int i = 0; i < threshCandidates.Length; i++ )
        //    {
        //        threshCandidates[i] = STFAlgo.rand.NextDouble();
        //    }

        //    var bestThreshold = -1.0d;
        //    var bestScore = double.MinValue;
        //    var bestLeftSet = new STLabelledImage[0];
        //    var bestRightSet = new STLabelledImage[0];
        //    ClassDistribution bestLeftClassDist = null;
        //    ClassDistribution bestRightClassDist = null;

        //    bool inputIsEmpty = images.Length == 0; //there is no image, no split is possible -> leaf
        //    bool inputIsOne = images.Length == 1;   //there is exactly one image, no split is possible -> passthrough

        //    if (!inputIsEmpty && !inputIsOne)
        //    {

        //        foreach (var curThresh in threshCandidates)
        //        {
        //            var currentLeftSet = new STLabelledImage[0];
        //            var currentRightSet = new STLabelledImage[0];
        //            ClassDistribution currentLeftClassDist = null;
        //            ClassDistribution currentRightClassDist = null;

        //            splitDatasetWithThreshold(images, curThresh, parameters, out currentLeftSet, out currentRightSet, out currentLeftClassDist, out currentRightClassDist);
        //            double leftEntr = calcEntropy(currentLeftClassDist);
        //            double rightEntr = calcEntropy(currentRightClassDist);

        //            //from semantic texton paper -> maximize the score value
        //            double leftWeight = (-1.0d) * currentLeftClassDist.getClassDistSum() / classDist.getClassDistSum();
        //            double rightWeight = (-1.0d) * currentRightClassDist.getClassDistSum() / classDist.getClassDistSum();
        //            double score = leftWeight * leftEntr + rightWeight * rightEntr;

        //            if (score > bestScore) //new best threshold found
        //            {
        //                bestScore = score;
        //                bestThreshold = curThresh;

        //                bestLeftSet = currentLeftSet;
        //                bestRightSet = currentRightSet;
        //                bestLeftClassDist = currentLeftClassDist;
        //                bestRightClassDist = currentRightClassDist;
        //            }
        //        }
        //    }


        //    bool isLeaf = inputIsEmpty;   //no images reached this node

        //    if(parameters.forcePassthrough) //if passthrough mode is active, never create a leaf inside the tree (force-fill the tree)
        //    {
        //        isLeaf = false;
        //    }

        //    bool passThrough = (Math.Abs(bestScore) < parameters.thresholdInformationGainMinimum) || inputIsOne;  //no more information gain => copy the parent node

        //    Certainty = bestScore;

        //    if (isLeaf)
        //    {
        //        leftRemaining = null;
        //        rightRemaining = null;
        //        leftClassDist = null;
        //        rightClassDist = null;
        //        return STFAlgo.DeciderTrainingResult.Leaf;
        //    }

        //    if (!passThrough && !isLeaf)  //reports for passthrough and leaf nodes are printed in Node.train method
        //    {
        //        Report.Line(3, "NN t:" + bestThreshold + " s:" + bestScore + "; img=" + images.Length + " l/r=" + bestLeftSet.Length + "/" + bestRightSet.Length + ((isLeaf) ? "->leaf" : ""));
        //    }

        //    this.DecisionThreshold = bestThreshold;
        //    leftRemaining = bestLeftSet;
        //    rightRemaining = bestRightSet;
        //    leftClassDist = bestLeftClassDist;
        //    rightClassDist = bestRightClassDist;

        //    if (passThrough || isLeaf)
        //    {
        //        return STFAlgo.DeciderTrainingResult.PassThrough;
        //    }

        //    return STFAlgo.DeciderTrainingResult.InnerNode;
        //}

        public STFAlgo.DeciderTrainingResult InitializeDecision(DataPointSet currentDatapoints, ClassDistribution classDist, TrainingParams parameters, out DataPointSet leftRemaining, out DataPointSet rightRemaining, out ClassDistribution leftClassDist, out ClassDistribution rightClassDist)
        {
            //get a bunch of candidates for decision using the supplied featureProvider and samplingProvider, select the best one based on entropy, return either the 
            //left/right split subsets and false, or true if this node should be a leaf

            var threshCandidates = new double[parameters.thresholdCandidateNumber];
            for (int i = 0; i < threshCandidates.Length; i++)
            {
                threshCandidates[i] = STFAlgo.rand.NextDouble();
            }

            var bestThreshold = -1.0d;
            var bestScore = double.MinValue;
            var bestLeftSet = new DataPointSet();
            var bestRightSet = new DataPointSet();
            ClassDistribution bestLeftClassDist = null;
            ClassDistribution bestRightClassDist = null;

            bool inputIsEmpty = currentDatapoints.DPSet.Length == 0; //there is no image, no split is possible -> leaf
            bool inputIsOne = currentDatapoints.DPSet.Length == 1;   //there is exactly one image, no split is possible -> passthrough

            if (!inputIsEmpty && !inputIsOne)
            {

                foreach (var curThresh in threshCandidates)
                {
                    var currentLeftSet = new DataPointSet();
                    var currentRightSet = new DataPointSet();
                    ClassDistribution currentLeftClassDist = null;
                    ClassDistribution currentRightClassDist = null;

                    splitDatasetWithThreshold(currentDatapoints, curThresh, parameters, out currentLeftSet, out currentRightSet, out currentLeftClassDist, out currentRightClassDist);
                    double leftEntr = calcEntropy(currentLeftClassDist);
                    double rightEntr = calcEntropy(currentRightClassDist);

                    //from semantic texton paper -> maximize the score value
                    double leftWeight = (-1.0d) * currentLeftClassDist.getClassDistSum() / classDist.getClassDistSum();
                    double rightWeight = (-1.0d) * currentRightClassDist.getClassDistSum() / classDist.getClassDistSum();
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


            bool isLeaf = inputIsEmpty;   //no images reached this node

            if (parameters.forcePassthrough) //if passthrough mode is active, never create a leaf inside the tree (force-fill the tree)
            {
                isLeaf = false;
            }

            bool passThrough = (Math.Abs(bestScore) < parameters.thresholdInformationGainMinimum) || inputIsOne;  //no more information gain => copy the parent node

            Certainty = bestScore;

            if (isLeaf)
            {
                leftRemaining = null;
                rightRemaining = null;
                leftClassDist = null;
                rightClassDist = null;
                return STFAlgo.DeciderTrainingResult.Leaf;
            }

            if (!passThrough && !isLeaf)  //reports for passthrough and leaf nodes are printed in Node.train method
            {
                Report.Line(3, "NN t:" + bestThreshold + " s:" + bestScore + "; dp=" + currentDatapoints.DPSet.Length + " l/r=" + bestLeftSet.DPSet.Length + "/" + bestRightSet.DPSet.Length + ((isLeaf) ? "->leaf" : ""));
            }

            this.DecisionThreshold = bestThreshold;
            leftRemaining = bestLeftSet;
            rightRemaining = bestRightSet;
            leftClassDist = bestLeftClassDist;
            rightClassDist = bestRightClassDist;

            if (passThrough || isLeaf)
            {
                return STFAlgo.DeciderTrainingResult.PassThrough;
            }

            return STFAlgo.DeciderTrainingResult.InnerNode;
        }

        private void splitDatasetWithThreshold(DataPointSet dps, double threshold, TrainingParams parameters, out DataPointSet leftSet, out DataPointSet rightSet, out ClassDistribution leftDist, out ClassDistribution rightDist)
        {
            var leftList = new List<DataPoint>();
            var rightList = new List<DataPoint>();

            int targetFeatureCount = Math.Min(dps.DPSet.Length, parameters.maxSampleCount);
            var actualDPS = dps.DPSet.RandomOrder().Take(targetFeatureCount).ToArray();

            foreach (var dp in actualDPS)
            {
                //var datapoints = SamplingProvider.getDataPoints(img);

                //select only a subset of features


                var feature = FeatureProvider.getFeature(dp);

                //img.TrainingBias = datapoints.SetWeight;

                //int leftScore = 0;
                //int rightScore = 0;
                //foreach (var feature in features)
                //{
                //    //if the feature is smaller than the threshold, vote to put it into the left set, else right set
                //    if (feature.Value < threshold)
                //    {
                //        leftScore = leftScore + 1;
                //    }
                //    else
                //    {
                //        rightScore = rightScore + 1;
                //    }
                //}

                if (feature.Value < threshold)
                {
                    leftList.Add(dp);
                }
                else
                {
                    rightList.Add(dp);
                }

            }

            leftSet = new DataPointSet();
            rightSet = new DataPointSet();

            leftSet.DPSet = leftList.ToArray();
            rightSet.DPSet = rightList.ToArray();

            leftDist = new ClassDistribution(GlobalParams.labels, leftSet);
            rightDist = new ClassDistribution(GlobalParams.labels, rightSet);
        }

        //splits the dataset using this threshold, from this the score should be calculated
        //private void splitDatasetWithThreshold(STLabelledImage[] imgs, double threshold, TrainingParams parameters, out STLabelledImage[] leftSet, out STLabelledImage[] rightSet, out ClassDistribution leftDist, out ClassDistribution rightDist)
        //{
        //    var leftList = new List<STLabelledImage>();
        //    var rightList = new List<STLabelledImage>();

        //    foreach(var img in imgs)
        //    {
        //        var datapoints = SamplingProvider.getDataPoints(img);

        //        //select only a subset of features
        //        int targetFeatureCount = Math.Min(datapoints.DPSet.Length, parameters.maxSampleCount);
        //        datapoints.DPSet = datapoints.DPSet.RandomOrder().Take(targetFeatureCount).ToArray();

        //        var features = FeatureProvider.getArrayOfFeatures(datapoints);

        //        img.TrainingBias = datapoints.SetWeight;
                
        //        int leftScore = 0;
        //        int rightScore = 0;
        //        foreach(var feature in features)
        //        {
        //            //if the feature is smaller than the threshold, vote to put it into the left set, else right set
        //            if(feature.Value<threshold)
        //            {
        //                leftScore = leftScore + 1;
        //            }
        //            else
        //            {
        //                rightScore = rightScore + 1;
        //            }
        //        }

        //        if(leftScore > rightScore)
        //        {
        //            leftList.Add(img);
        //        }
        //        else
        //        {
        //            rightList.Add(img);
        //        }

        //    }


        //    leftSet = leftList.ToArray();
        //    rightSet = rightList.ToArray();
        //    leftDist = new ClassDistribution(GlobalParams.labels, leftSet);
        //    rightDist = new ClassDistribution(GlobalParams.labels, rightSet);
        //}

        //calculates the entropy of one class distribution as input to the score calculation
        private double calcEntropy(ClassDistribution dist)
        {
            //from http://en.wikipedia.org/wiki/ID3_algorithm

            double sum = 0;
            //foreach(var cl in dist.ClassLabels)
            foreach (var cl in GlobalParams.labels)
            {
                var px = dist.getClassProbability(cl);
                if(px == 0)
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

    public class STNode
    {
        public bool isLeaf = false;
        public int DistanceFromRoot = 0;
        public STNode LeftChild;
        public STNode RightChild;
        public Decider Decider;
        public ClassDistribution ClassDistribution;
        public int GlobalIndex = -1;    //this node's global index in the forest 

        public void getClassDecisionRecursive(DataPoint dataPoint, List<TextonNode> currentList, TrainingParams parameters)
        {
            switch(parameters.classificationMode)
            {
                case ClassificationMode.Semantic:

                    var rt = new TextonNode();
                    rt.Index = GlobalIndex;
                    rt.Level = DistanceFromRoot;
                    rt.Value = 1;
                    currentList.Add(rt);

                    //descend left or right, or return if leaf
                    if (!this.isLeaf)
                    {
                        bool leftright = Decider.Decide(dataPoint);
                        if (leftright)   //true means left
                        {
                            LeftChild.getClassDecisionRecursive(dataPoint, currentList, parameters);
                        }
                        else            //false means right
                        {
                            RightChild.getClassDecisionRecursive(dataPoint, currentList, parameters);
                        }
                    }
                    else //break condition
                    {
                        return;
                    }
                    return;
                case ClassificationMode.LeafOnly:

                    if(!this.isLeaf) //we are in a branching point, continue forward
                    {
                        bool leftright = Decider.Decide(dataPoint);
                        if(leftright)   //true means left
                        {
                            LeftChild.getClassDecisionRecursive(dataPoint, currentList, parameters);
                        }
                        else            //false means right
                        {
                            RightChild.getClassDecisionRecursive(dataPoint, currentList, parameters);
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

        #region DELETE THIS

        public void getTestDecisionRecursive(List<TextonNode> currentList)
        {
                    var rt = new TextonNode();
                    rt.Index = GlobalIndex;
                    rt.Level = DistanceFromRoot;
                    rt.Value = 1;
                    currentList.Add(rt);
                    if (!this.isLeaf)
                    {
                        LeftChild.getTestDecisionRecursive(currentList);
                        RightChild.getTestDecisionRecursive(currentList);
                    }
                    else
                    {
                        return;
                    }

                    

        }

        #endregion

        //every node adds 0 to the histogram (=initialize the histogram parameters)
        public void initializeEmpty(List<TextonNode> currentList)
        {
            var rt = new TextonNode();
            rt.Index = GlobalIndex;
            rt.Level = DistanceFromRoot;
            rt.Value = 0;
            currentList.Add(rt);

            //descend left or right, or return if leaf
            if (!this.isLeaf)
            {
                LeftChild.initializeEmpty(currentList);
                RightChild.initializeEmpty(currentList);
            }
        }
    }

    public class SemanticTexton
    {
        public STNode Root;
        public int Index = -1;   //this tree's index within the forest, is set by the forest during initialization
        public int NumNodes = 0;    //how many nodes does this tree have in total

        public SemanticTexton()
        {
            Root = new STNode();
            Root.GlobalIndex = this.Index;
        }

        public List<TextonNode> getClassDecision(DataPointSet dp, TrainingParams parameters)
        {
            var result = new List<TextonNode>();

            foreach(var point in dp.DPSet)
            {
                var cumulativeList = new List<TextonNode>();
                Root.getClassDecisionRecursive(point, cumulativeList, parameters);
                foreach (var el in cumulativeList)        //this is redundant with initializeEmpty -> todo
                {
                    el.TreeIndex = this.Index;
                }
                result.AddRange(cumulativeList);
            }

            return result;
        }

        #region DELETE THIS - unit test

        public List<TextonNode> getTESTDecision()
        {
            var result = new List<TextonNode>();

            for (int i = 0; i < 10; i++) 
            {
                var cumulativeList = new List<TextonNode>();
                Root.getTestDecisionRecursive(cumulativeList);
                foreach (var el in cumulativeList)        //this is redundant with initializeEmpty -> todo
                {
                    el.TreeIndex = this.Index;
                }
                result.AddRange(cumulativeList);
            }

            return result;
        }

        #endregion

        public void initializeEmpty(List<TextonNode> currentList)
        {
            var cumulativeList = new List<TextonNode>();
            Root.initializeEmpty(cumulativeList);
            foreach (var el in cumulativeList)
            {
                el.TreeIndex = this.Index;
            }
            currentList.AddRange(cumulativeList);
        }
    }


    public class STForest
    {
        public SemanticTexton[] SemanticTextons;
        public string name = "unnamed";
        public int NumTrees = 0;

        public int numNodes = -1;

        public STForest()
        {

        }

        public STForest(string name)
        {
            this.name = name;
        }

        public void InitializeEmptyForest(int treeCount)
        {
            NumTrees = 0;
            SemanticTextons = new SemanticTexton[treeCount];

            for(int i=0; i<treeCount; i++)
            {
                SemanticTextons[i] = new SemanticTexton();
                SemanticTextons[i].Index = NumTrees;
                NumTrees++;
            }
        }

        public Textonization getTextonRepresentation(STImage img, TrainingParams parameters)
        {
            if(numNodes <= -1)  //this part is deprecated
            {
                numNodes = SemanticTextons.Sum(x=>x.NumNodes);
            }

            //we must use the sampling provider of a tree because parameters are currently not saved to file -> fix this!
            var imageSamples = SemanticTextons[0].Root.Decider.SamplingProvider.getDataPoints(img);

            var result = new Textonization();
            result.initializeEmpty(numNodes);

            var basicNodes = new List<TextonNode>();

            STFAlgo.treeCounter = 0;

            foreach(var tree in SemanticTextons)    //for each tree, get a textonization of the data set and sum up the result
            {
                STFAlgo.treeCounter++;

                tree.initializeEmpty(basicNodes);

                var curTex = tree.getClassDecision(imageSamples, parameters);

                result.addNodes(curTex);

            }

            result.addNodes(basicNodes);    //we can add all empty nodes after calculation because it simply increments all nodes by 0 (no change) while initializing unrepresented nodes

            return result;
        }

        #region DELETE THIS - unit test
        public Textonization getTESTtextonization()     //get 10 in each histogram bin
        {
            if (numNodes <= -1)
            {
                numNodes = SemanticTextons.Sum(x => x.NumNodes);
            }

            var result = new Textonization();
            result.initializeEmpty(numNodes);

            var basicNodes = new List<TextonNode>();

            STFAlgo.treeCounter = 0;

            foreach (var tree in SemanticTextons)    //for each tree, get a textonization of the data set and sum up the result
            {
                STFAlgo.treeCounter++;

                tree.initializeEmpty(basicNodes);

                var curTex = tree.getTESTDecision();

                result.addNodes(curTex);

            }

            result.addNodes(basicNodes);    //we can add all empty nodes after calculation because it simply increments all nodes by 0 (no change) while initializing unrepresented nodes

            return result;
        }

        #endregion
    }
    #endregion

    #region Class Labels and Distributions

    //one class label
    public class ClassLabel
    {
        //index in the global label list
        public int Index;
        //string identifier
        public string Name;

        public ClassLabel()
        {
            Index = -1;
            Name = "";
        }
    }

    //a class distribution, containing a histogram over all classes and their respective values.
    public class ClassDistribution
    {
        //the histogram value for each label
        public double[] ClassValues;
        
        //number of labels (if variable - this is specified in the paper but not used)
        public int Length;

        //dont use this constructor, JSON only
        public ClassDistribution()
        {

        }

        //adds two class distributions, requires them to use the same global class label list
        public static ClassDistribution operator+(ClassDistribution a, ClassDistribution b)
        {
            ClassDistribution result = new ClassDistribution(GlobalParams.labels);

            foreach (var cl in GlobalParams.labels)
            {
                result.addClNum(cl, a.ClassValues[cl.Index] + b.ClassValues[cl.Index]);
            }

            return result;
        }

        //multiply histogram values with a constant
        public static ClassDistribution operator*(ClassDistribution a, double b)
        {
            ClassDistribution result = new ClassDistribution(GlobalParams.labels);

            foreach (var cl in GlobalParams.labels)
            {
                result.addClNum(cl, a.ClassValues[cl.Index] * b);
            }

            return result;
        }

        //initializes all classes with a count of 0
        public ClassDistribution(ClassLabel[] allLabels)
        {
            ClassValues = new double[allLabels.Length]; ;
            for(int i=0;i<allLabels.Length;i++) //allLabels must have a sequence of indices [0-n]
            {
                ClassValues[i] = 0;
            }
            Length = allLabels.Length;
        }

        //initialize classes and add the data points
        public ClassDistribution(ClassLabel[] allLabels, DataPointSet dps)
            : this(allLabels)
        {
            addDatapoints(dps);
        }

        //add one data point to histogram
        public void addDP(DataPoint dp)
        {
            if(dp.label == -2)
            {
                return;
            }

            double incrementValue = 1.0d;

            addClNum(GlobalParams.labels.Where(x => x.Index == dp.label).First() , incrementValue);
        }

        //add one histogram entry
        public void addClNum(ClassLabel cl, double num)
        {
            ClassValues[cl.Index] = ClassValues[cl.Index] + num;
        }

        //add all data points to histogram
        public void addDatapoints(DataPointSet dps)
        {
            foreach (var dp in dps.DPSet)
            {
                this.addDP(dp);
            }
        }

        //returns the proportion of the elements of this class to the number of all elements in this distribution
        public double getClassProbability(ClassLabel label)  
        {
            var sum = ClassValues.Sum();

            if(sum == 0)
            {
                return 0;
            }

            var prob = ClassValues[label.Index] / sum;

            return prob;
        }

        //returns sum of histogram values
        public double getClassDistSum()
        {
            return ClassValues.Sum();
        }

        //normalize histogram
        public void normalize()
        {
            var sum = ClassValues.Sum();

            if (sum == 0)
            {
                return;
            }

            for(int i=0; i<ClassValues.Length;i++)
            {
                ClassValues[i] = ClassValues[i] / sum;
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

        public void initializeEmpty(int numNodes)
        {
            Length = numNodes;
            Nodes = new TextonNode[numNodes];

            for (int i = 0; i < numNodes; i++)
            {
                Nodes[i] = new TextonNode() { Index = i, Level = 0, Value = 0 };
            }
        }

        public void addValues(double[] featureValues)
        {
            this.Values = featureValues;
            this.Length = featureValues.Length;
        }

        public void setNodes(TextonNode[] featureNodes)
        {
            this.Nodes = featureNodes;
            this.Length = featureNodes.Length;
        }

        public void addNodes(List<TextonNode> featureNodes)
        {
            foreach(var node in featureNodes)
            {
                var localNode = this.Nodes[node.Index];

                localNode.Level = node.Level;
                localNode.TreeIndex = node.TreeIndex;
                localNode.Value += node.Value;
            }
        }

        public static Textonization operator+(Textonization current, Textonization other)     //adds two textonizations. must have same length and same node indices (=be from the same forest)
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

    //wrapper class for PixImage
    public class STImage
    {
        public string ImagePath;

        //the image will be loaded into memory on first use
        private PixImage<byte> pImage;
        private bool isLoaded = false;

        //don't use, JSON only
        public STImage()
        {

        }

        //Creates a new image without loading it into memory
        public STImage(string filePath)
        {
            ImagePath = filePath;
        }

        [JsonIgnore]
        public PixImage<byte> PixImage
        {
            get
            {
                if (!isLoaded) Load();
                return pImage;
            }
        }

        //loads all images from a directory
        public static STImage[] GetImagesFromDirectory(string directoryPath)
        {
            return null;
            //todo
        }
        
        private void Load()
        {
            pImage = new PixImage<byte>(ImagePath);
            isLoaded = true;
        }
    }

    //STImage with added class label, used for training and testing
    public class STLabelledImage : STImage
    {
        //this image's class label
        public ClassLabel ClassLabel;

        //this value can be changed if needed different image bias during training
        public double TrainingBias = 1.0f;   

        //don't use, JSON only
        public STLabelledImage()
        {
               
        }

        //creates a new image from filename
        public STLabelledImage(string fileName) : base(fileName) 
        {
            ClassLabel = new ClassLabel();
        }

        //THIS METHOD WORKS ONLY FOR THE TEST PROBLEM - NEEDS TO BE GENERALIZED
        //reads all images from a directory and their labels from filename
        public static STLabelledImage[] getLabelledImagesFromDirectory(string directoryPath, ClassLabel[] labels)
        {
            string[] picFiles = Directory.GetFiles(directoryPath);
            
            var result = new STLabelledImage[picFiles.Length];

            for (int i = 0; i < picFiles.Length; i++ )
            {
                var s = picFiles[i];
                string currentFilename = Path.GetFileNameWithoutExtension(s);
                string[] filenameSplit = currentFilename.Split('_');
                int fileLabel = Convert.ToInt32(filenameSplit[0]);
                ClassLabel currentLabel = labels.First(x => x.Index == fileLabel-1);
                result[i] = new STLabelledImage(s) { ClassLabel = currentLabel };
            }


            return result;
            
        }

        public static STLabelledImage[] getTDatasetFromDirectory(string directoryPath, ClassLabel[] labels)
        {
            string nokpath = Path.Combine(directoryPath, "NOK");
            string okpath = Path.Combine(directoryPath, "OK");
            string[] nokFiles = Directory.GetFiles(nokpath);
            string[] okFiles = Directory.GetFiles(okpath);

            var result = new STLabelledImage[okFiles.Length + nokFiles.Length];

            for (int i = 0; i < nokFiles.Length; i++)
            {
                var s = nokFiles[i];

                result[i] = new STLabelledImage(s) { ClassLabel = labels[0] };
            }

            for (int i = 0; i < okFiles.Length; i++)
            {
                var s = okFiles[i];

                result[nokFiles.Length + i] = new STLabelledImage(s) { ClassLabel = labels[1] };
            }

            return result;
        }
    }

    //STLabelledImage with added Textonization
    public class STTextonizedLabelledImage : STLabelledImage
    {
        public Textonization Textonization;

        //don't use, JSON only
        public STTextonizedLabelledImage()
        {

        }

        public STTextonizedLabelledImage(string fileName) : base(fileName)
        {

        }

        //copy constructor
        public STTextonizedLabelledImage(STLabelledImage parent, Textonization textonization) : base(parent.ImagePath)
        {
            this.ClassLabel = parent.ClassLabel;
            this.TrainingBias = parent.TrainingBias;
            this.Textonization = textonization;
        }

    }
#endregion

    #region Parameter Classes

    public class TrainingParams
    {
        public string forestName;       //identifier of the forest, has no usage except for readability
        public int classesCount;        //how many classes
        public int treesCount;          //how many trees should the forest have
        public int maxTreeDepth;        //maximum depth of one tree
        public int imageSubsetCount;    //how many images should be randomly selected from the training set for each tree's training
        public int samplingWindow;      //side length of the square window around a pixel to be sampled; half of this size is effectively the border around the image
        public int maxSampleCount;      //limit the maximum number of samples for one image (selected randomly from all samples) -> set this to 99999999 for all samples
        public FeatureType featureType; //the type of feature that should be extracted using the feature providers
        public SamplingType samplingType;//mode of sampling
        public int randomSamplingCount;  //if sampling = random sampling, how many points?
        public FeatureProviderFactory featureProviderFactory;       //creates a new feature provider for each decision node in the trees to apply to a sample point (window); currently value of a random pixel, sum of two random pixels, absolute difference of two random pixels
        public SamplingProviderFactory samplingProviderFactory;     //creates a new sample point provider which is currently applied to all pictures; currently sample a regular grid with stride, sample a number of random points
        public int thresholdCandidateNumber;    //how many random thresholds should be tested in a tree node to find the best one
        public double thresholdInformationGainMinimum;    //break the tree node splitting if no threshold has a score better than this
        public ClassificationMode classificationMode;    //what feature representation method to use; currently: standard representation by leaves only, semantic texton representation using the entire tree
        public bool forcePassthrough;   //during forest generation, force each datapoint to reach a leaf
        public bool enableGridSearch;         //the SVM tries out many values to find the optimal C (can take a long time)

        //todo: definitely parse this from a text file or so
    }


    public static class GlobalParams
    {
        //experimental/temporary helper params
        public static bool EnableSampleNumberCountUnbias = false;    //remove bias for number of samples per image, enable if images vary in size for regular grid sampling
        public static bool NormalizeDistributions = false;           //normalize class distributions to [0-1]

        //required params
        public static ClassLabel[] labels;     //list of class labels that is used globally
    }

    public class FilePaths
    {
        public string forestFilePath;
        public string testsetpath1;
        public string testsetpath2;
        public string semantictestsetpath1;
        public string semantictestsetpath2;
        public string trainingsetpath;
        public string kernelsetpath;
        public string trainingTextonsFilePath;
        public string testTextonsFilePath;
    }

#endregion
}
