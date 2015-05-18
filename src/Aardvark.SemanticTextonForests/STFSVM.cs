using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Aardvark.Base;
using LibSvm;

namespace Aardvark.SemanticTextonForests
{
    public class NodeHierarchy
    {
        /// <summary>
        /// Creates a new hierarchy object an already existing zero-intialized object, and fills it with the values.
        /// </summary>
        /// <param name="reference">Zero-valued data structure.</param>
        /// <param name="values">Values to fill into the data structure</param>
        public NodeHierarchy(Dictionary<int, double>[][] reference, HistogramNode[] values )
        {
            Nodes = reference;

            foreach(var v in values)
            {
                Nodes[v.TreeIndex][v.Level][v.Index] = v.Value;
            }
        }

        public Dictionary<int, double>[][] Nodes;
    }


    /// <summary>
    /// Classifier object which wraps the functionality of using a Semantic Texton Forest for image classification.
    /// </summary>
    public class Classifier
    {
        /// <summary>
        /// The model of this classifier after training.
        /// </summary>
        public Model ClassifierModel;
        /// <summary>
        /// A folder path for writing temporary files.
        /// </summary>
        public string TempFileFolderPath;

        private TextonizedLabeledImage[] TrainingSet;
        private Problem TrainingProb;
        private string TempTrainingKernelPath;
        private string TempTestProblemPath;
        private bool IsTrained = false;

        /// <summary>
        /// Hierarchical structure from which histogram node values can be retrieved quickly.
        /// The outer array accesses elements by the index of a tree, the inner array by depth level, and the Dictionary
        /// associates node indices with their histogram value.
        /// </summary>
        private Dictionary<int,double>[][] BaseHierarchy;

        /// <summary>
        /// Creates a new, untrained Classifier. 
        /// </summary>
        /// <param name="tempFileFolderPath">A folder in which temporary files are written.</param>
        public Classifier(string tempFileFolderPath)
        {
            if (!Directory.Exists(tempFileFolderPath)) Directory.CreateDirectory(tempFileFolderPath);
            TempFileFolderPath = tempFileFolderPath;
            TempTrainingKernelPath = Path.Combine(tempFileFolderPath, "SemanticKernel.ds");
            TempTestProblemPath = Path.Combine(tempFileFolderPath, "SemanticTestSet.ds");
        }

        /// <summary>
        /// The previously trained classifier creates a new semantic problem from an (unlabeled) textonization set.
        /// </summary>
        /// <param name="images">Textonized image set.</param>
        /// <param name="filename">Output filename.</param>
        private void NewSemanticProblem(TextonizedLabeledImage[] images, string filename)
        {
            if (!IsTrained) return;
            CreateSemanticKernelAndWriteToFile(images, this.TrainingSet, filename);
        }

        /// <summary>
        /// Creates a new training kernel from a labeled training textonization set.
        /// </summary>
        /// <param name="images">Labeled training textonized image set.</param>
        /// <param name="filename">Output filename.</param>
        private void NewKernel(TextonizedLabeledImage[] images, string filename)
        {
            CreateSemanticKernelAndWriteToFile(images, images, filename);
        }

        /// <summary>
        /// Trains this classifier on a data set of labeled textonizations and stores the resulting training kernel in the temp. folder.
        /// </summary>
        /// <param name="images">Textonized labeled training image set.</param>
        /// <param name="parameters">Training parameters.</param>
        public void Train(TextonizedLabeledImage[] images, TrainingParams parameters)
        {
            //since all textonizations must have the same hierarchy, we extract it from the first training example.
            InitNodeHierarchy(images[0].Textonization);

            string filename = TempTrainingKernelPath;
            NewKernel(images, filename);
            this.TrainingSet = images;
            TrainFromFile(filename, parameters);

            IsTrained = true;
        }

        /// <summary>
        /// Initialize the data structure from which textonization nodes can be efficiently retrieved. 
        /// This data structure is formed by the corresponding forest. One forest must be used for all data in this Classifier.
        /// </summary>
        /// <param name="example"></param>
        private void InitNodeHierarchy(Textonization example)
        {
            var numTrees = example.Histogram.Max(x => x.TreeIndex + 1);

            BaseHierarchy = new Dictionary<int, double>[numTrees][];

            for(var i=0; i<numTrees; i++)
            {
                var curTreeNodes = example.Histogram.Where(x => x.TreeIndex == i);
                var numDepth = curTreeNodes.Max(x => x.Level + 1);
                BaseHierarchy[i] = new Dictionary<int, double>[numDepth];
                for(var j =0; j<numDepth; j++)
                {
                    var curDepthNodes = curTreeNodes.Where(x => x.Level == j);
                    BaseHierarchy[i][j] = new Dictionary<int, double>();
                    foreach (var node in curDepthNodes)
                    {
                        BaseHierarchy[i][j].Add(node.Index, 0.0);
                    }
                }
            }
        }

        /// <summary>
        /// Trains this classifier on the training kernel given in a file.
        /// </summary>
        /// <param name="kernelFilePath">Path of the file containing the previously calculated training kernel.</param>
        /// <param name="parameters"></param>
        private void TrainFromFile(string kernelFilePath, TrainingParams parameters)
        {
            var prob = ReadSVMProblemFromFile(kernelFilePath);
            LearnProblem(prob, parameters);
        }

        /// <summary>
        /// Trains the classifier on a given problem.
        /// </summary>
        /// <param name="prob">Input problem.</param>
        /// <param name="parameters">Parameters.</param>
        private void LearnProblem(Problem prob, TrainingParams parameters)
        {
            TrainingProb = prob;

            //empirical value
            double C = 17.8;

            Report.BeginTimed("Training and CrossValidation");

            if (parameters.EnableGridSearch)
            {
                int Ccount = 20;
                double Cstart = 1.78;
                double bestScore = double.MinValue;

                for (int i = 0; i < Ccount; i++)
                {
                    double cC = Cstart * (Math.Pow(10, (double)i - (double)(Ccount / 2))) / 1.0;
                    double currentScore = prob.GetCrossValidationAccuracy(Extensions.CreateParamCHelper(cC), 5);

                    if (currentScore > bestScore)
                    {
                        bestScore = currentScore;
                        C = cC;
                    }
                }
            }

            ClassifierModel = Svm.Train(prob, Extensions.CreateParamCHelper(C));
            Report.End();
        }

        /// <summary>
        /// The trained classifier predicts the class label of one unlabeled textonization.
        /// </summary>
        /// <param name="image">Textonized image (label can be set to any value).</param>
        /// <param name="parameters">Parameters.</param>
        /// <returns></returns>
        public Label PredictLabel(TextonizedLabeledImage image, TrainingParams parameters)
        {
            return this.PredictLabels(new TextonizedLabeledImage[] { image }, parameters)[0];
        }

        /// <summary>
        /// The trained classifier predicts the class label of a set of unlabeled textonizations.
        /// </summary>
        /// <param name="images">Textonized images (labels can be set to any value).</param>
        /// <param name="parameters">Parameters.</param>
        /// <returns></returns>
        public Label[] PredictLabels(TextonizedLabeledImage[] images, TrainingParams parameters)
        {
            if (!IsTrained) return null;
            var result = new Label[images.Length];

            var tr = this.Test(images, parameters, "predict " + images.Length + " imgs");

            for(var i =0;i<images.Length;i++)
            {
                result[i] = parameters.Labels[tr.PredictedClassLabelIndices[i]];
            }
            return result;
        }

        /// <summary>
        /// Tests the precision of this classifier by operating on a training set with known labels.
        /// </summary>
        /// <param name="images">Labeled textonized image test set.</param>
        /// <param name="parameters">Test set.</param>
        /// <param name="name">Test run name.</param>
        /// <returns></returns>
        public ClassifierTestResult Test(TextonizedLabeledImage[] images, TrainingParams parameters, string name)
        {
            if (!IsTrained) return null;
            string filename = "";
            filename = TempTestProblemPath;
            NewSemanticProblem(images, filename);

            var prob = ReadSVMProblemFromFile(filename);

            return ClassifyProblem(prob, parameters, name);
        }

        /// <summary>
        /// Tests the recall of this classifier by attempting to classify its own training set.
        /// </summary>
        /// <param name="parameters">Parameters object.</param>
        /// <param name="name">Test run name.</param>
        /// <returns></returns>
        public ClassifierTestResult TestRecall(TrainingParams parameters, string name)
        {
            if (!IsTrained) return null;
            return ClassifyProblem(TrainingProb, parameters, name);
        }

        /// <summary>
        /// Performs testing classification on a semantic problem.
        /// </summary>
        /// <param name="prob">Input problem.</param>
        /// <param name="parameters">Input parameters.</param>
        /// <param name="name">Classification run name.</param>
        /// <returns></returns>
        private ClassifierTestResult ClassifyProblem(Problem prob, TrainingParams parameters, string name = "")
        {
            if (!IsTrained) return null;
            int correct = 0;
            int wrong = 0;

            int[] predictedLabels = new int[prob.Count];
            int[,] confusionMatrix = new int[parameters.ClassesCount, parameters.ClassesCount];
            for (int i = 0; i < prob.Count; i++)
            {
                var curFeature = prob.x[i];
                var curLabel = prob.y[i];

                var estimatedLabel = 0.0d;

                estimatedLabel = Svm.Predict(ClassifierModel, curFeature);

                predictedLabels[i] = (int)estimatedLabel;
                confusionMatrix[(int)curLabel, (int)estimatedLabel]++;
                if (curLabel == estimatedLabel)
                {
                    correct++;
                }
                else
                {
                    wrong++;
                }
            }

            var prec = ((double)correct / (double)(wrong + correct));

            string s = "\n\nClassification name: " + name;
            s += ("\nCorrect: " + correct + "(" + prec + ")" + "\nWrong: " + wrong);
            s += ("\nConfusion matrix:\n");

            // build formatted confusion matrix
            s += (String.Format(CultureInfo.InvariantCulture, "{0,6}", ""));
            for (int i = 0; i < parameters.ClassesCount; i++)
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + parameters.Labels[i].Index + ")"));
            s += Environment.NewLine;
            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + parameters.Labels[i].Index + ")"));
                for (int j = 0; j < confusionMatrix.GetLength(1); j++)
                {
                    if (i == j)
                    {
                        s += (String.Format(CultureInfo.InvariantCulture, "{0,4}*", confusionMatrix[i, j]));
                    }
                    else
                    {
                        s += (String.Format("{0,5}", confusionMatrix[i, j]));
                    }
                }
                s += Environment.NewLine;
            }

            var result = new ClassifierTestResult()
            {
                OutputString = s,
                Precision = prec,
                NumCorrect = correct,
                NumWrong = wrong,
                PredictedClassLabelIndices = predictedLabels
            };

            return result;
        }

        /// <summary>
        /// Writes to disk a classification problem string to be used with the default classifiers (linear, RBF, ...).
        /// </summary>
        /// <param name="images">Data set to be formatted.</param>
        /// <param name="path">Output filename.</param>
        private void CreateSVMProblemAndWriteToFile(TextonizedLabeledImage[] images, string path)
        {
            var problemVector = new StringBuilder();
            Report.BeginTimed(2, "Formatting SVM Data.");
            int reportCounter = 0;
            foreach (var curImg in images)
            {
                Report.Progress(2, (double)(reportCounter++) / (double)images.Length);

                var curFeatures = curImg.Textonization.Histogram;
                var curLabel = curImg.Label;

                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0}  ", (double)curLabel.Index));

                for (int j = 0; j < curFeatures.Length; j++)
                {
                    var curValue = curFeatures[j];
                    problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0}:{1} ", (double)(j + 1.0), curValue.Value));
                }
                problemVector.Append(Environment.NewLine);
            }
            Report.End(2);
            Report.Line(1, "Writing svm_problem to file.");
            File.WriteAllText(path, problemVector.ToString());
        }

        /// <summary>
        /// Creates a semantic kernel or semantic problem which stores the distances between all elements of the example array to all 
        /// elements of the references array.
        /// The training kernel is computed by calling this method with the training images in both parameters.
        /// A classification problem is obtained by calling this method with the training images and the test images respectively.
        /// </summary>
        /// <param name="examples">The data which is tested against the reference set.</param>
        /// <param name="references">The reference set which is used to train the classifier.</param>
        /// <param name="path">Output filename.</param>
        private void CreateSemanticKernelAndWriteToFile(TextonizedLabeledImage[] examples, TextonizedLabeledImage[] references, string path)
        {
            var problemVector = new StringBuilder();
            Report.BeginTimed(2, "Creating SVM kernel.");
            int reportCounter = 0;
            for (int i = 0; i < examples.Length; i++)
            {
                var curImg = examples[i];

                Report.Progress(2, (double)(reportCounter++) / (double)examples.Length);

                var Ph = new NodeHierarchy(BaseHierarchy.Copy(x => x.Copy(y => y.Copy())), curImg.Textonization.Histogram);
                var PIndex = (double)(i + 1);

                //for each image, calculate the semantic distance to all other images and store it in the kernel matrix
                //the format for the output is given in the libsvm documentation: https://github.com/encog/libsvm-java
                //the actual calculation of the values is given in the Semantic Textons (Cipolla) paper

                problemVector.Append(string.Format(CultureInfo.InvariantCulture, "{0} ", (double)curImg.Label.Index));
                problemVector.Append(string.Format(CultureInfo.InvariantCulture, "0:{0} ", PIndex));

                //for each other image
                for (int j = 0; j < references.Length; j++)
                {
                    //get the required indices
                    var otherImg = references[j];
                    var Qh = new NodeHierarchy(BaseHierarchy.Copy(x => x.Copy(y => y.Copy())), otherImg.Textonization.Histogram);
                    var QIndex = (double)(j + 1);

                    //calculate the distance value K(P,Q)
                    var KPQ = 0.0;
                    //count the non-contributing trees for correct scaling
                    var zeroTrees = 0;

                    //for each tree
                    var numtrees = Ph.Nodes.Length;
                    for (int ti = 0; ti < numtrees; ti++)
                    {
                        //evaluate the K~(P,Q) function for these two images P and Q for the current tree
                        double KTildePQ = Ktilde(Ph, Qh, ti);

                        //multiply the result by the Z scaling factor
                        //Z = K~(P,P) * K~(Q,Q)

                        double Z = Ktilde(Ph, Ph, ti) * Ktilde(Qh, Qh, ti);
                        double Zroot = 1.0 / Math.Sqrt(Z);

                        if (Z <= 0.0)  //this tree contributes only with its root or no nodes at all -> no information
                        {
                            Zroot = 0.0;
                            zeroTrees++;
                        }

                        KTildePQ = KTildePQ * Zroot;

                        //overall result is the normalized sum of tree results
                        KPQ += KTildePQ;
                    }

                    //scaling of the overall sum
                    KPQ = KPQ * (1.0 / (double)(numtrees - zeroTrees));

                    if (numtrees == zeroTrees)   //catch exception case
                    {
                        KPQ = 0.0;
                    }

                    //for each other image, append the distance to the current image to the result string
                    problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0}:{1} ", (double)QIndex, KPQ));
                }

                //begin a new line for each current image
                problemVector.Append(Environment.NewLine);
            }

            //the result is a formatted matrix storing the semantic distance of each image to all other images
            Report.End(2);

            Report.Line(1, "Writing svm_problem (kernel) to file.");

            File.WriteAllText(path, problemVector.ToString());
        }

        /// <summary>
        /// Evaluates K~(P,Q) for the semantic histograms P and Q in one tree (treeIndex)
        /// </summary>
        /// <param name="P">Semantic histogram P</param>
        /// <param name="Q">Semantic histogram Q</param>
        /// <param name="treeIndex">Index of the tree to evaluate the expression for</param>
        /// <returns></returns>
        private double Ktilde(NodeHierarchy P, NodeHierarchy Q, int treeIndex)
        {
            //consider all the nodes that belong to this tree
            var currentTreeNodes = P.Nodes[treeIndex];

            if (currentTreeNodes.Length <= 0)  //some of the trees have only the root node as leaf
            {
                Report.Line("Encountered zero-tree.");
                return 0.0;
            }

            //for each depth level
            var dSum = 0.0;
            var Idplusone = 0.0;
            var D = currentTreeNodes.Length - 1;
            for (int d = D; d >= 0; d--)  //shifted index (by -1) compared to paper
            {
                //normalization for depth
                var normFactor = 1.0 / (Math.Pow(2.0, (double)D - (double)d + 1.0));

                var currentNodes = currentTreeNodes[d];

                // for each individual node
                //   consider nodes in this tree in the current depth
                var Id = 0.0;
                foreach(var cn in currentNodes)
                {

                    //get the comparison node
                    var otherNodeValue = Q.Nodes[treeIndex][d][cn.Key];

                    //add the minimum of the two values to the result
                    Id += Math.Min(cn.Value, otherNodeValue);
                }

                //inner result is the difference between these two, multiplied with the normalization factor
                var innerResult = (Id - Idplusone) * normFactor;

                //store the current match sum for the next iteration
                Idplusone = Id;

                //tree result is the sum of inner results
                dSum += innerResult;
            }
            return dSum;
        }

        /// <summary>
        /// Reads a problem string in the LibSVM format from a file.
        /// </summary>
        /// <param name="path">Path of the LibSVM-formated file.</param>
        /// <returns></returns>
        private Problem ReadSVMProblemFromFile(string path)
        {
            Report.Line(1, "Reading svm_problem from file.");
            return Extensions.ReadProblem(path);
        }

        /// <summary>
        /// Builds a linear (dot product) kernel from an example data set to a training data set and writes it to file.
        /// </summary>
        /// <param name="example"></param>
        /// <param name="reference"></param>
        /// <param name="outputFilename"></param>
        private void BuildLinear(Problem example, Problem reference, string outputFilename)
        {
            var problemVector = new StringBuilder();

            for (int i = 0; i < example.Count; i++)
            {

                var curFeatures = example.x[i];
                var curLabel = example.y[i];

                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0} ", (double)curLabel));
                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "0:{0} ", (double)(i + 1)));

                for (int j = 0; j < reference.Count; j++)
                {
                    var refFeatures = reference.x[j];
                    var refLabel = reference.y[j];

                    //value = dot(xi,xj);

                    var diff = 0.0;

                    for (int k = 0; k < curFeatures.Length; k++)
                    {
                        var curDiff = curFeatures[k].Value * refFeatures[k].Value;
                        diff += curDiff;
                    }

                    var curValue = diff;

                    problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0}:{1} ", (j + 1), curValue));
                }

                problemVector.Append(Environment.NewLine);
            }

            Report.Line(1, "Writing svm_problem to file.");

            File.WriteAllText(outputFilename, problemVector.ToString());
        }

        /// <summary>
        /// Builds a radial basis function kernel from an example data set to a training data set and writes it to file.
        /// </summary>
        /// <param name="example"></param>
        /// <param name="reference"></param>
        /// <param name="gamma">Gamma parameter of the RBF.</param>
        /// <param name="outputFilename"></param>
        private void BuildRBF(Problem example, Problem reference, double gamma, string outputFilename)
        {

            var problemVector = new StringBuilder();

            for (int i = 0; i < example.Count; i++)
            {

                var curFeatures = example.x[i];
                var curLabel = example.y[i];

                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0} ", (double)curLabel));
                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "0:{0} ", (double)(i + 1)));

                for (int j = 0; j < reference.Count; j++)
                {
                    var refFeatures = reference.x[j];
                    var refLabel = reference.y[j];

                    //value = e^(-gamma* length(xi - xj)^2)

                    var diff = new List<double>();

                    for (int k = 0; k < curFeatures.Length; k++)
                    {
                        var curDiff = curFeatures[k].Value - refFeatures[k].Value;
                        diff.Add(curDiff);
                    }

                    var diffFeatures = diff.ToArray();

                    var diffLengthSquared = 0.0d;

                    for (int k = 0; k < diffFeatures.Length; k++)
                    {
                        var val = diffFeatures[k];
                        diffLengthSquared += val * val;
                    }

                    var curValue = Math.Pow(Math.E, ((-1.0d) * gamma * diffLengthSquared));

                    problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0}:{1} ", (j + 1), curValue));
                }

                problemVector.Append(Environment.NewLine);
            }

            Report.Line(1, "Writing svm_problem to file.");

            File.WriteAllText(outputFilename, problemVector.ToString());
        }
    }

    /// <summary>
    /// The result of a Classifier test storing statistics of the test run
    /// </summary>
    public class ClassifierTestResult
    {
        public int[] PredictedClassLabelIndices;
        public string OutputString;
        public double Precision;
        public int NumCorrect;
        public int NumWrong;

    }

    // TO BE DELETED

    //    public Dictionary<int, double> PredictProbabilities(svm_node[] x)
    //    {
    //        if (this.model == null)
    //            throw new Exception("No trained svm model");

    //        var probabilities = new Dictionary<int, double>();
    //        int nr_class = model.nr_class;

    //        double[] prob_estimates = new double[nr_class];
    //        int[] labels = new int[nr_class];
    //        svm.svm_get_labels(model, labels);

    //        svm.svm_predict_probability(this.model, x, prob_estimates);
    //        for (int i = 0; i < nr_class; i++)
    //            probabilities.Add(labels[i], prob_estimates[i]);

    //        return probabilities;
    //    }

    //    public double GetCrossValidationAccuracy(int nr_fold)
    //    {
    //        int i;
    //        int total_correct = 0;
    //        double[] target = new double[prob.l];

    //        svm.svm_cross_validation(prob, param, nr_fold, target);

    //        for (i = 0; i < prob.l; i++)
    //            if (Math.Abs(target[i] - prob.y[i]) < double.Epsilon)
    //                ++total_correct;
    //        var CVA = total_correct / (double)prob.l;
    //        //Debug.WriteLine("Cross Validation Accuracy = {0:P} ({1}/{2})", CVA, total_correct, prob.l);
    //        return CVA;
    //    }
    //}
}
