using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Aardvark.Base;

namespace Aardvark.SemanticTextonForests
{
    class STFSVM
    {
        //public SVMParameter Parameter;
        public C_SVC TrainedSvm;
        //public MySVM SemanticSVM;
        TextWriter oldWriter;
        STTextonizedLabelledImage[] trainingSet;
        public bool createNewFiles = true;
        private LibSvm.Problem trainingProb;




        //the libraries spam A LOT of text
        void disableOutput()
        {
            oldWriter = Console.Out;
            Console.SetOut(TextWriter.Null);
        }

        void enableOutput()
        {
            Console.SetOut(oldWriter);
        }

        public STFSVM(bool createNewFiles = true)
        {
            this.createNewFiles = createNewFiles;
        }

        void newProblem(STTextonizedLabelledImage[] images, string filename)
        {
            if (createNewFiles)
            {
                createSVMProblemAndWriteToFile(images, filename);
            }
        }

        void newSemanticProblem(STTextonizedLabelledImage[] images, string filename)
        {
            if (createNewFiles)
            {
                createSemanticKernelAndWriteToFile(images, this.trainingSet, filename);
            }
        }

        void newKernel(STTextonizedLabelledImage[] images, string filename)
        {
            if (createNewFiles)
            {
                createSemanticKernelAndWriteToFile(images, images, filename);
            }
        }

        //trains this SVM on a given labelled training set (stores the problem string in filename)
        public void train(STTextonizedLabelledImage[] images, TrainingParams parameters, string filenameDefault, string filenameKernel)
        {
            //LIBSVM.NET STUFF
            ////no descriptions, reconstructed this from a tutorial
            ////classes are identified by index in the sequence [0-n]

            ////formulate problem
            ////create either default problem matrix or precomputed kernel problem matrix
            
            string filename = "";

            if (parameters.classificationMode == ClassificationMode.LeafOnly)
            {
                filename = filenameDefault;
                newProblem(images, filename);
            }
            else if (parameters.classificationMode == ClassificationMode.Semantic)
            {
                filename = filenameKernel;
                newKernel(images, filename);
            }

            this.trainingSet = images;

            

            LibSvm.Problem prob = readSVMProblemFromFile(filename);
            trainingProb = prob;

            //values empirically found by cross validation
            //double C = 62;    //two class problem
            //double C = 178;     //twenty class problem
            //double gamma = 0.005;  //from absdiff tree

            //double C = 17.8;     //rbf params from grid search
            //double gamma = 0.0005;

            double C = 1780;     //new format
            double gamma = 0.000005;  



            //double gamma = 0.00000005;    //found best after new method

            //combined CV grid search for both C and gamma
            disableOutput();

            if (parameters.enableGridSearch)
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
                        TrainedSvm = new C_SVC(prob, KernelHelper.LinearKernel(), cC);

                        double currentScore = TrainedSvm.GetCrossValidationAccuracy(5);

                        if (currentScore > bestScore)
                        {
                            bestScore = currentScore;
                            C = cC;
                            gamma = cgamma;
                        }
                    }
                }
            }
            enableOutput();

            if (parameters.classificationMode == ClassificationMode.LeafOnly)
            {
                TrainedSvm = new C_SVC(prob, KernelHelper.RadialBasisFunctionKernel(gamma), C);
                TrainedSvm = new C_SVC(prob, KernelHelper.LinearKernel(), C);
            }
            else if (parameters.classificationMode == ClassificationMode.Semantic)
            {
                //TrainedSvm = new C_SVC(prob, KernelHelper.LinearKernel(), C);
                //gridsearch
                C = 17.8;

                if (parameters.enableGridSearch)
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
                            double currentScore = SemanticSVM.GetCrossValidationAccuracy(5);

                            if (currentScore > bestScore)
                            {
                                bestScore = currentScore;
                                C = cC;
                                //gamma = cgamma;
                            }
                        }
                    }
                }


                //SemanticSVM = new MySVM(prob, C);
            }



            //LIBSVMNET STUFF - LIBRARY CURRENTLY REMOVED FROM SOLUTION, CODE TO BE REMOVED
            ////trainingProblem = trainingProblem.Normalize(SVMNormType.L2);

            ////create parameter set (copied)
            //Parameter = new SVMParameter();
            //Parameter.Type = SVMType.C_SVC;
            //Parameter.Kernel = SVMKernelType.RBF;
            //Parameter.C = 1;
            //Parameter.Gamma = 1;

            //Report.BeginTimed(1, "Training the SVM");

            ////train the svm

            //double[] crossValidationResults; // output labels
            //int nFold = 5;
            //trainingProblem.CrossValidation(Parameter, nFold, out crossValidationResults);

            //double crossValidationAccuracy = trainingProblem.EvaluateClassificationProblem(crossValidationResults);
            
            //SVMModel model = trainingProblem.Train(Parameter);

            //Report.End(1);

            //Report.BeginTimed(1, "Recall:");

            //// Predict the instances in the test set
            //double[] trainingResults = trainingProblem.Predict(model);

            //// Evaluate the test results
            //int[,] confusionMatrix;
            //double testAccuracy = trainingProblem.EvaluateClassificationProblem(trainingResults, model.Labels, out confusionMatrix);

            //string s = "";

            //// Print the resutls
            //s += ("\n\nCross validation accuracy: " + crossValidationAccuracy);
            //s+="\nTest accuracy: " + testAccuracy;
            //s+=("\nConfusion matrix:\n");

            //// Print formatted confusion matrix
            //s+=(String.Format("{0,6}", ""));
            //for (int i = 0; i < model.Labels.Length; i++)
            //    s+=(String.Format("{0,5}", "(" + model.Labels[i] + ")"));
            //s+=Environment.NewLine;
            //for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            //{
            //    s+=(String.Format("{0,5}", "(" + model.Labels[i] + ")"));
            //    for (int j = 0; j < confusionMatrix.GetLength(1); j++)
            //        s+=(String.Format("{0,5}", confusionMatrix[i, j]));
            //    s += Environment.NewLine;
            //}

            //File.WriteAllText(Path.Combine(Program.workDir, "test.txt"),s);

            //Report.End(1);
        }

        //public ClassLabel predict(STTextonizedLabelledImage image, TrainingParams parameters)
        //{

        //}

        //performs a test classification of images using this SVM (stores the LibSVM-formatted problem string in filename).
        //returns output string with name as identifier.
        public SVMTestResult test(STTextonizedLabelledImage[] images, TrainingParams parameters, string defaultFilename, string semanticFilename, string name)
        {
            string filename = "";
            if (parameters.classificationMode == ClassificationMode.LeafOnly)
            {
                filename = defaultFilename;
                newProblem(images, filename);
            }
            else if (parameters.classificationMode == ClassificationMode.Semantic)
            {
                filename = semanticFilename;
                newSemanticProblem(images, filename);
            }
            
            var prob = readSVMProblemFromFile(filename);

            return classifyProblem(prob, parameters, name);
        }

        public SVMTestResult testWithTrainingset(TrainingParams parameters, string name)
        {
            return classifyProblem(trainingProb, parameters, name);
        }

        //performs classification on one svm_problem
        public SVMTestResult classifyProblem(LibSvm.Problem prob, TrainingParams parameters, string name)
        {
            int correct = 0;
            int wrong = 0;

            int[,] confusionMatrix = new int[parameters.classesCount, parameters.classesCount];

            disableOutput();

            for (int i = 0; i < prob.l; i++)
            {
                var curFeature = prob.x[i];
                var curLabel = prob.y[i];

                var estimatedLabel = 0.0d;

                if (parameters.classificationMode == ClassificationMode.LeafOnly)
                {
                    estimatedLabel = TrainedSvm.Predict(curFeature);
                }
                else if (parameters.classificationMode == ClassificationMode.Semantic)
                {
                    estimatedLabel = SemanticSVM.Predict(curFeature);
                }

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

            enableOutput();

            var prec = ((double)correct / (double)(wrong + correct));

            string s = "\n\nClassification name: " + name;
            s += ("\nCorrect: " + correct + "(" + prec + ")" + "\nWrong: " + wrong);
            s += ("\nConfusion matrix:\n");

            // build formatted confusion matrix
            s += (String.Format(CultureInfo.InvariantCulture, "{0,6}", ""));
            for (int i = 0; i < parameters.classesCount; i++)
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + GlobalParams.labels[i].Index + ")"));
            s += Environment.NewLine;
            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + GlobalParams.labels[i].Index + ")"));
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
                outputString = s,
                precision = prec,
                numCorrect = correct,
                numWrong = wrong
            };

            return result;
        }

        

        //creates a classification problem string in the LibSVM format and writes it to file
        void createSVMProblemAndWriteToFile(STTextonizedLabelledImage[] images, string path)
        {
            //old format
            //var problemVector = new StringBuilder();
            //Report.BeginTimed(2, "Formatting SVM Data.");
            //int reportCounter = 0;
            //foreach (var curImg in images)
            //{
            //    Report.Progress(2, (double)(reportCounter++) / (double)images.Length);

            //    var curFeatures = curImg.Textonization.Values;
            //    var curLabel = curImg.ClassLabel;

            //    problemVector.Append(String.Format("{0}  ",(double)curLabel.Index));

            //    for (int j = 0; j < curFeatures.Length; j++)
            //    {
            //        var curValue = curFeatures[j];

            //        problemVector.Append(String.Format("{0}:{1} ", (j + 1), curValue));
            //    }



            //    problemVector.Append(Environment.NewLine);
            //}
            //Report.End(2);

            //Report.Line(1, "Writing svm_problem to file.");

            //File.WriteAllText(path, problemVector.ToString());

            //new format
            var problemVector = new StringBuilder();
            Report.BeginTimed(2, "Formatting SVM Data.");
            int reportCounter = 0;
            foreach (var curImg in images)
            {
                Report.Progress(2, (double)(reportCounter++) / (double)images.Length);

                var curFeatures = curImg.Textonization.Nodes;
                var curLabel = curImg.ClassLabel;

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
        void createSemanticKernelAndWriteToFile(STTextonizedLabelledImage[] examples, STTextonizedLabelledImage[] references, string path)
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

                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0} ", (double)curImg.ClassLabel.Index));
                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "0:{0} ", PIndex));

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
                    for (int ti = 0; ti < numtrees; ti++ )
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

                    if(numtrees == zeroTrees)   //catch exception case
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

            if (currentTreeNodes.Length <= 0)  //TODO: This shouldn't happen, but does - some of the trees have only the root node as leaf, which is 
            //ignored in the semantic histogram. fix this!
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



        //private double Ktilde(TextonNode[] P, TextonNode[] Q, int treeIndex)
        //{
        //    //consider all the nodes that belong to this tree
        //    var currentTreeNodes = P.Where(n => n.TreeIndex == treeIndex).ToArray();

        //    if(currentTreeNodes.Length <= 0)  //TODO: This shouldn't happen, but does - some of the trees have only the root node as leaf, which is 
        //                                      //ignored in the semantic histogram. fix this!
        //    {
        //        return 0.0;
        //    }

        //    //for each depth level
        //    var dSum = 0.0;
        //    var D = currentTreeNodes.Max(t => t.Level);
        //    for (int d = 1; d <= D; d++)  //shifted index (by -1) compared to paper
        //    {
        //        //consider nodes in this tree in the current depth
        //        var currentDepthNodes = currentTreeNodes.Where(n => n.Level == d).ToArray();    

        //        //normalization for depth
        //        var normFactor = 1.0 / (Math.Pow(2.0, (double)D - (double)d + 1.0));

        //        //for each individual node
        //        var Id = 0.0;
        //        foreach (var node in currentDepthNodes)
        //        {
        //            //get the comparison node
        //            var otherNode = Q[node.Index];

        //            //add the minimum of the two values to the result
        //            Id += Math.Min(node.Value, otherNode.Value);
        //        }

        //        //do the same for each node in the next depth level. if it's the max level, take 0 instead
        //        var Idplusone = 0.0;
        //        if (d < D)  //if we are at max level (d=D) this code block does nothing, skip
        //        {
        //            var nextDepthNodes = currentTreeNodes.Where(n => n.Level == d + 1).ToArray();

        //            foreach (var node in nextDepthNodes)
        //            {
        //                //get the comparison node
        //                var otherNode = Q[node.Index];

        //                //add the minimum of the two values to the result
        //                Idplusone += Math.Min(node.Value, otherNode.Value);
        //            }
        //        }

        //        //inner result is the difference between these two, multiplied with the normalization factor
        //        var innerResult = (Id - Idplusone) * normFactor;

        //        //tree result is the sum of inner results
        //        dSum += innerResult;
        //    }


        //    return dSum;
        //}


        //reads a LibSVM problem string from file
        LibSvm.Problem readSVMProblemFromFile(string path)
        {
            Report.Line(1, "Reading svm_problem from file.");
            return ProblemHelper.ReadProblem(path);
        }

        #region delete this

        //testing only - adds precise classification information, to be removed
        private string classifyProblemTEST(LibSvm.Problem prob, TrainingParams parameters, string name)
        {
            int correct = 0;
            int wrong = 0;

            string s = "\n\nClassification name: " + name + "\n";

            int[,] confusionMatrix = new int[parameters.classesCount, parameters.classesCount];

            disableOutput();

            for (int i = 0; i < prob.l; i++)
            {
                var curFeature = prob.x[i];
                var curLabel = prob.y[i];

                var estimatedLabel = 0.0d;

                if (parameters.classificationMode == ClassificationMode.LeafOnly)
                {
                    estimatedLabel = TrainedSvm.Predict(curFeature);
                    s += String.Format(CultureInfo.InvariantCulture, "Sample {0}: [{1},{2}] class {3} -> categorized as {4}\n", i, curFeature[0].Value, curFeature[1].Value, curLabel, estimatedLabel);
                }
                else if (parameters.classificationMode == ClassificationMode.Semantic)
                {
                    estimatedLabel = SemanticSVM.Predict(curFeature);
                    s += String.Format(CultureInfo.InvariantCulture, "Sample {0}: class {1} -> categorized as {2}\n", i, curLabel, estimatedLabel);
                }

                

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

            enableOutput();

            
            s += ("\nCorrect: " + correct + "(" + ((double)correct / (double)(wrong + correct)) + ")" + "\nWrong: " + wrong);
            s += ("\nConfusion matrix:\n");

            // build formatted confusion matrix
            s += (String.Format(CultureInfo.InvariantCulture, "{0,6}", ""));
            for (int i = 0; i < parameters.classesCount; i++)
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + GlobalParams.labels[i].Index + ")"));
            s += Environment.NewLine;
            for (int i = 0; i < confusionMatrix.GetLength(0); i++)
            {
                s += (String.Format(CultureInfo.InvariantCulture, "{0,5}", "(" + GlobalParams.labels[i].Index + ")"));
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

            return s;
        }

#if false
        //special test case for debugging
        public void unitTest()
        {
            //the input to this test case are the simple data sets supplied in the filenames below. 
            //the "semantic" files are the pre-computed RBF kernels for the same data as in the "normal" files.
            //the output should be the (almost) same classification rate for both approaches.
            string result = "";

            double C = 100;
            double gamma = 1000;

            string testdir = "unitTest";
            string trainingFilename = Path.Combine(Program.workDir, testdir, "trainingdata.ds");
            string testcaseFilename = Path.Combine(Program.workDir, testdir, "testdata.ds");

            string semantictrainingFilename = Path.Combine(Program.workDir, testdir, "trainingkernel.ds");
            string semantictestFilename = Path.Combine(Program.workDir, testdir, "testkernel.ds");

            string testOutputFilename = Path.Combine(Program.workDir, testdir, "out.txt");

            svm_problem trainingprob = readSVMProblemFromFile(trainingFilename);
            svm_problem testprob = readSVMProblemFromFile(testcaseFilename);

            //calculate the RBF kernel
            buildRBF(trainingprob, trainingprob, gamma, semantictrainingFilename);
            buildRBF(testprob, trainingprob, gamma, semantictestFilename);

            buildLinear(trainingprob, trainingprob, semantictrainingFilename);
            buildLinear(testprob, trainingprob, semantictestFilename);

            svm_problem semantickernel = readSVMProblemFromFile(semantictrainingFilename);
            svm_problem semanticprob = readSVMProblemFromFile(semantictestFilename);


            //create both types of SVM, feed them the training and test data/kernels
            TrainedSvm = new C_SVC(trainingprob, KernelHelper.RadialBasisFunctionKernel(gamma), C);
            TrainedSvm = new C_SVC(trainingprob, KernelHelper.LinearKernel(), C);

            SemanticSVM = new MySVM(semantickernel, C);

            TrainingParams testParams = new TrainingParams()
            {
                classesCount = (int)trainingprob.y.Max(x => x) + 1,     //classes count given in the training problem
                classificationMode = ClassificationMode.LeafOnly
            };

            result += this.classifyProblemTEST(testprob, testParams, "default SVM");

            testParams.classificationMode = ClassificationMode.Semantic;

            result += this.classifyProblemTEST(semanticprob, testParams, "semantic kernel SVM");

            //the result should be same classification rate in both cases
            File.WriteAllText(testOutputFilename, result);
        }
#endif
        private void buildLinear(LibSvm.Problem example, LibSvm.Problem reference, string outputFilename)
        {
            var problemVector = new StringBuilder();

            for (int i = 0; i < example.l; i++)
            {

                var curFeatures = example.x[i];
                var curLabel = example.y[i];

                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0} ", (double)curLabel));
                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "0:{0} ", (double)(i + 1)));

                for (int j = 0; j < reference.l; j++)
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


        private void buildRBF(LibSvm.Problem example, LibSvm.Problem reference, double gamma, string outputFilename)
        {

            var problemVector = new StringBuilder();

            for (int i = 0; i < example.l; i++)
            {

                var curFeatures = example.x[i];
                var curLabel = example.y[i];

                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "{0} ", (double)curLabel));
                problemVector.Append(String.Format(CultureInfo.InvariantCulture, "0:{0} ", (double)(i + 1)));

                for (int j = 0; j < reference.l; j++)
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

#endregion
    }

    public class SVMTestResult
    {
        public string outputString;
        public double precision;
        public int numCorrect;
        public int numWrong;

    }

#if false
    public class MySVM : SVM
    {
        //this class represents an SVM with precomputed kernel
        //this is a slightly modified class copied from the LIBSVM.NET library, because they forgot to include some functionality

        public MySVM(svm_problem kernelProb, double C, double cache_size = 100, bool probability = false)
            : base(kernelProb, new svm_parameter()
            {
                //almost all parameters are copied from libsvm.net implementation, see https://github.com/nicolaspanel/libsvm.net/blob/master/LIBSVM/SVC/SVC.cs
                svm_type = 0,                     //libsvm type - always C_SVC (=0)      
                kernel_type = 4,                  //4 means precomputed kernel, see https://github.com/encog/libsvm-java
                degree = 0,                       //polynom kernel degree - not used
                C = C,                            //C
                gamma = 0,                        //RBF gamma - not used
                coef0 = 0,                        //polynom exponent - not used
                nu = 0.0,                         //regression parameter - not used
                cache_size = cache_size,          //libsvm parameter
                eps = 1e-3,                       //training parameter
                p = 0.1,                          //training parameter
                shrinking = 1,                    //training optimization
                probability = probability ? 1 : 0,//output
                nr_weight = 0,                    //output
                weight_label = new int[0],        //label weightings - not used
                weight = new double[0],
            })

        {
            
        }

        public override double Predict(svm_node[] x)
        {
            if (model == null)
                throw new Exception("No trained svm model");

            return svm.svm_predict(model, x);
        }

        public Dictionary<int, double> PredictProbabilities(svm_node[] x)
        {
            if (this.model == null)
                throw new Exception("No trained svm model");

            var probabilities = new Dictionary<int, double>();
            int nr_class = model.nr_class;

            double[] prob_estimates = new double[nr_class];
            int[] labels = new int[nr_class];
            svm.svm_get_labels(model, labels);

            svm.svm_predict_probability(this.model, x, prob_estimates);
            for (int i = 0; i < nr_class; i++)
                probabilities.Add(labels[i], prob_estimates[i]);

            return probabilities;
        }

        public double GetCrossValidationAccuracy(int nr_fold)
        {
            int i;
            int total_correct = 0;
            double[] target = new double[prob.l];

            svm.svm_cross_validation(prob, param, nr_fold, target);

            for (i = 0; i < prob.l; i++)
                if (Math.Abs(target[i] - prob.y[i]) < double.Epsilon)
                    ++total_correct;
            var CVA = total_correct / (double)prob.l;
            //Debug.WriteLine("Cross Validation Accuracy = {0:P} ({1}/{2})", CVA, total_correct, prob.l);
            return CVA;
        }
    }
#endif
}
