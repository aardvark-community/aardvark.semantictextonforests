using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibSvm;
using Aardvark.Base;
using System.IO;
using System.Globalization;

namespace ScratchAttila
{
    class Classifier
    {
        //public SVMParameter Parameter;
        public Model SemanticSVM;
        TextonizedLabelledImage[] trainingSet;
        //public bool createNewFiles = true;
        private Problem trainingProb;
        public string TempFileFolderPath;
        private string _tempTrainingKernelPath;
        private string _tempTestProblemPath;
        private bool _isTrained = false;

        public Classifier(string tempFileFolderPath)
        {
            this.TempFileFolderPath = tempFileFolderPath;
            this._tempTrainingKernelPath = Path.Combine(tempFileFolderPath, "SemanticKernel.ds");
            this._tempTestProblemPath = Path.Combine(tempFileFolderPath, "SemanticTestSet.ds");
        }

        void newProblem(TextonizedLabelledImage[] images, string filename)
        {
            createSVMProblemAndWriteToFile(images, filename);
        }

        void newSemanticProblem(TextonizedLabelledImage[] images, string filename)
        {
            createSemanticKernelAndWriteToFile(images, this.trainingSet, filename);
        }

        void newKernel(TextonizedLabelledImage[] images, string filename)
        {
            createSemanticKernelAndWriteToFile(images, images, filename);
        }

        //trains this SVM on a given labelled training set (stores the problem string in filename)
        public void train(TextonizedLabelledImage[] images, TrainingParams parameters)
        {
            string filename = "";

            if (parameters.ClassificationMode == ClassificationMode.LeafOnly)
            {
                filename = _tempTrainingKernelPath + ".l";
                newProblem(images, filename);
            }
            else if (parameters.ClassificationMode == ClassificationMode.Semantic)
            {
                filename = _tempTrainingKernelPath;
                newKernel(images, filename);
            }

            this.trainingSet = images;

            var prob = readSVMProblemFromFile(filename);

            learnProblem(prob, parameters);
        }

        public void trainFromFile(string kernelFilePath, TrainingParams parameters)
        {
            var prob = readSVMProblemFromFile(kernelFilePath);

            learnProblem(prob, parameters);
        }

        private void learnProblem(Problem prob, TrainingParams parameters)
        {
            trainingProb = prob;

            //values empirically found by cross validation
            double C = 1780;
            double gamma = 0.000005;

            //combined CV grid search for both C and gamma
            if (parameters.EnableGridSearch)
            {
                int Ccount = 20;
                double Cstart = 1.78;
                int Gcount = 1;
                double Gstart = 0.0005;
                double bestScore = double.MinValue;

                for (int i = 0; i < Ccount; i++)
                {
                    double cC = Cstart * (Math.Pow(10, (double)i - (double)(Ccount / 2)));
                    for (int j = 0; j < Gcount; j++)
                    {
                        double cgamma = Gstart * (Math.Pow(10, (double)j - (double)(Gcount / 2)));

                        //TrainedSvm = new C_SVC(prob, KernelHelper.RadialBasisFunctionKernel(cgamma), cC);
                        //TrainedSvm = new C_SVC(prob, KernelHelper.LinearKernel(), cC);

                        double currentScore = 0.0; // TrainedSvm.GetCrossValidationAccuracy(5);

                        if (currentScore > bestScore)
                        {
                            bestScore = currentScore;
                            C = cC;
                            gamma = cgamma;
                        }
                    }
                }
            }

            if (parameters.ClassificationMode == ClassificationMode.LeafOnly)
            {
                //TrainedSvm = new C_SVC(prob, KernelHelper.RadialBasisFunctionKernel(gamma), C);
                //TrainedSvm = new C_SVC(prob, KernelHelper.LinearKernel(), C);
            }
            else if (parameters.ClassificationMode == ClassificationMode.Semantic)
            {
                //TrainedSvm = new C_SVC(prob, KernelHelper.LinearKernel(), C);
                //gridsearch
                C = 17.8;

                Report.BeginTimed("Training and CrossValidation");

                if (parameters.EnableGridSearch)
                {
                    int Ccount = 20;
                    double Cstart = 1.78;
                    int Gcount = 1;
                    //double Gstart = 0.0005;
                    double bestScore = double.MinValue;

                    for (int i = 0; i < Ccount; i++)
                    {
                        double cC = Cstart * (Math.Pow(10, (double)i - (double)(Ccount / 2))) / 1.0;
                        for (int j = 0; j < Gcount; j++)
                        {
                            //double cgamma = Gstart * (Math.Pow(10, (double)j - (double)(Gcount / 2)));

                            //SemanticSVM = new MySVM(prob, cC);
                            //SemanticSVM = Tuple.Create(prob, );

                            double currentScore = prob.GetCrossValidationAccuracy(Sketches.CreateParamCHelper(cC), 5);

                            if (currentScore > bestScore)
                            {
                                bestScore = currentScore;
                                C = cC;
                                //gamma = cgamma;
                            }
                        }
                    }
                }
                
                SemanticSVM = Svm.Train(prob, Sketches.CreateParamCHelper(C));
                Report.End();
            }
            _isTrained = true;
        }

        public ClassLabel predictLabel(TextonizedLabelledImage image, TrainingParams parameters)
        {
            return this.predictLabels(new TextonizedLabelledImage[] { image }, parameters)[0];
        }

        //predict the class labels of a set of images with this trained classifier
        public ClassLabel[] predictLabels(TextonizedLabelledImage[] images, TrainingParams parameters)
        {
            if (!_isTrained)
            {
                return null;
            }

            var result = new ClassLabel[images.Length];

            var tr = this.test(images, parameters, "predict " + images.Length + " imgs");

            for(var i =0;i<images.Length;i++)
            {
                result[i] = GlobalParams.Labels.Where(l => l.Index == tr.PredictedClassLabelIndices[i]).First();
            }

            return result;
        }

        //performs a test classification of images using this SVM (stores the LibSVM-formatted problem string in filename).
        //returns output string with name as identifier.
        public SVMTestResult test(TextonizedLabelledImage[] images, TrainingParams parameters, string name)
        {
            string filename = "";
            if (parameters.ClassificationMode == ClassificationMode.LeafOnly)
            {
                filename = _tempTestProblemPath+".l";
                newProblem(images, filename);
            }
            else if (parameters.ClassificationMode == ClassificationMode.Semantic)
            {
                filename = _tempTestProblemPath;
                newSemanticProblem(images, filename);
            }

            var prob = readSVMProblemFromFile(filename);

            return classifyProblem(prob, parameters, name);
        }

        public void testFromFile(string testProblemFilePath, TrainingParams parameters, string name)
        {
            var prob = readSVMProblemFromFile(testProblemFilePath);

            classifyProblem(prob, parameters, name);
        }

        public SVMTestResult testRecall(TrainingParams parameters, string name)
        {
            return classifyProblem(trainingProb, parameters, name);
        }

        //performs classification on one svm_problem
        public SVMTestResult classifyProblem(Problem prob, TrainingParams parameters, string name)
        {
            int correct = 0;
            int wrong = 0;

            int[] predictedLabels = new int[prob.Count];
            int[,] confusionMatrix = new int[parameters.ClassesCount, parameters.ClassesCount];
            for (int i = 0; i < prob.Count; i++)
            {
                var curFeature = prob.x[i];
                var curLabel = prob.y[i];

                var estimatedLabel = 0.0d;

                if (parameters.ClassificationMode == ClassificationMode.LeafOnly)
                {
                    //estimatedLabel = TrainedSvm.Predict(curFeature);
                }
                else if (parameters.ClassificationMode == ClassificationMode.Semantic)
                {
                    estimatedLabel = Svm.Predict(SemanticSVM, curFeature);
                }

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
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + GlobalParams.Labels[i].Index + ")"));
            s += Environment.NewLine;
            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + GlobalParams.Labels[i].Index + ")"));
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

            var result = new SVMTestResult()
            {
                OutputString = s,
                Precision = prec,
                NumCorrect = correct,
                NumWrong = wrong,
                PredictedClassLabelIndices = predictedLabels
            };

            return result;
        }

        //creates a classification problem string in the LibSVM format and writes it to file
        void createSVMProblemAndWriteToFile(TextonizedLabelledImage[] images, string path)
        {
            //new format
            var problemVector = new StringBuilder();
            Report.BeginTimed(2, "Formatting SVM Data.");
            int reportCounter = 0;
            foreach (var curImg in images)
            {
                Report.Progress(2, (double)(reportCounter++) / (double)images.Length);

                var curFeatures = curImg.Textonization.Nodes;
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

        //creates a SVM kernel which stores the distances between all elements of the example array to all elements of the references array
        //the training kernel is computed by calling this method with the training images in both parameters
        //the testing kernel is obtained by calling this method with the training images and the test images respectively
        void createSemanticKernelAndWriteToFile(TextonizedLabelledImage[] examples, TextonizedLabelledImage[] references, string path)
        {
            var problemVector = new StringBuilder();
            Report.BeginTimed(2, "Creating SVM kernel.");
            int reportCounter = 0;
            for (int i = 0; i < examples.Length; i++)
            {
                var curImg = examples[i];

                Report.Progress(2, (double)(reportCounter++) / (double)examples.Length);

                var P = curImg.Textonization.Nodes;
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
                    var Q = otherImg.Textonization.Nodes;
                    var QIndex = (double)(j + 1);

                    //calculate the distance value K(P,Q)
                    var KPQ = 0.0;
                    //count the non-contributing trees for correct scaling
                    var zeroTrees = 0;

                    //for each tree
                    var numtrees = P.Max(t => t.TreeIndex) + 1;
                    for (int ti = 0; ti < numtrees; ti++)
                    {
                        //evaluate the K~(P,Q) function for these two images P and Q for the current tree
                        double KTildePQ = Ktilde(P, Q, ti);

                        //multiply the result by the Z scaling factor
                        //Z = K~(P,P) * K~(Q,Q)

                        double Z = Ktilde(P, P, ti) * Ktilde(Q, Q, ti);
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

        //Evaluates K~(P,Q) for the semantic histograms P and Q in one tree (treeIndex)
        private double Ktilde(TextonNode[] P, TextonNode[] Q, int treeIndex)
        {
            //consider all the nodes that belong to this tree
            var currentTreeNodes = P.Where(n => n.TreeIndex == treeIndex).ToArray();

            if (currentTreeNodes.Length <= 0)  //TODO: This shouldn't happen, but does - some of the trees have only the root node as leaf, which is ignored in the semantic histogram. fix this!
            {
                Report.Line("Encountered zero-tree.");
                return 0.0;
            }

            //for each depth level
            var dSum = 0.0;
            var Idplusone = 0.0;
            var D = currentTreeNodes.Max(t => t.Level);
            for (int d = D; d >= 0; d--)  //shifted index (by -1) compared to paper
            {
                //consider nodes in this tree in the current depth
                var currentDepthNodes = currentTreeNodes.Where(n => n.Level == d).ToArray();

                //normalization for depth
                var normFactor = 1.0 / (Math.Pow(2.0, (double)D - (double)d + 1.0));

                //for each individual node
                var Id = 0.0;
                foreach (var node in currentDepthNodes)
                {
                    //get the comparison node
                    var otherNode = Q[node.Index];

                    //add the minimum of the two values to the result
                    Id += Math.Min(node.Value, otherNode.Value);
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

        //reads a LibSVM problem string from file
        Problem readSVMProblemFromFile(string path)
        {
            Report.Line(1, "Reading svm_problem from file.");
            return Sketches.ReadProblem(path);
        }

        private void buildLinear(Problem example, Problem reference, string outputFilename)
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

        private void buildRBF(Problem example, Problem reference, double gamma, string outputFilename)
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

    public class SVMTestResult
    {
        public int[] PredictedClassLabelIndices;
        public string OutputString;
        public double Precision;
        public int NumCorrect;
        public int NumWrong;

    }

    //public class MySVM : Parameter
    //{
    //    //this class represents an SVM with precomputed kernel
    //    //this is a slightly modified class copied from the LIBSVM.NET library, because they forgot to include some functionality

    //    public MySVM(Problem kernelProb, double C, double cache_size = 100, bool probability = false)
    //        {
    //            //almost all parameters are copied from libsvm.net implementation, see https://github.com/nicolaspanel/libsvm.net/blob/master/LIBSVM/SVC/SVC.cs
    //            svm_type = 0,                     //libsvm type - always C_SVC (=0)      
    //            kernel_type = 4,                  //4 means precomputed kernel, see https://github.com/encog/libsvm-java
    //            degree = 0,                       //polynom kernel degree - not used
    //            C = C,                            //C
    //            gamma = 0,                        //RBF gamma - not used
    //            coef0 = 0,                        //polynom exponent - not used
    //            nu = 0.0,                         //regression parameter - not used
    //            cache_size = cache_size,          //libsvm parameter
    //            eps = 1e-3,                       //training parameter
    //            p = 0.1,                          //training parameter
    //            shrinking = 1,                    //training optimization
    //            probability = probability ? 1 : 0,//output
    //            nr_weight = 0,                    //output
    //            weight_label = new int[0],        //label weightings - not used
    //            weight = new double[0],
    //        })

    //    {

    //    }

    //    public override double Predict(svm_node[] x)
    //    {
    //        if (model == null)
    //            throw new Exception("No trained svm model");

    //        return svm.svm_predict(model, x);
    //    }

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
