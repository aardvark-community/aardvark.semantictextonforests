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
    public static class Extensions
    {
        public static Parameter CreateParamCHelper(double c)
        {
            return new Parameter
            {
                SvmType = SvmType.C_SVC,
                KernelType = KernelType.PRECOMPUTED,                  //4 means precomputed kernel, see https://github.com/encog/libsvm-java
                Degree = 0,                       //polynom kernel degree - not used
                C = c,                            //C
                Gamma = 0,                        //RBF gamma - not used
                Coef0 = 0,                        //polynom exponent - not used
                Nu = 0.0,                         //regression parameter - not used
                CacheSize = 100,                  //libsvm parameter
                Eps = 1e-3,                       //training parameter
                p = 0.1,                          //training parameter
                Shrinking = 1,                    //training optimization
                Probability = false ? 1 : 0,      //output
                WeightLabel = new int[0],         //label weightings - not used
                Weight = new double[0],
            };
        }
        public static double GetCrossValidationAccuracy(this Problem prob, Parameter param, int nr_fold)
        {
            int i;
            int total_correct = 0;
            double[] target = Svm.CrossValidation(prob, param, nr_fold);

            for (i = 0; i < prob.Count; i++)
                if (Math.Abs(target[i] - prob.y[i]) < double.Epsilon)
                    ++total_correct;
            var CVA = total_correct / (double)prob.Count;
            //Debug.WriteLine("Cross Validation Accuracy = {0:P} ({1}/{2})", CVA, total_correct, prob.Count);
            return CVA;
        }

        public static T[] GetRandomSubset<T>(this T[] self, int count)
        {
            if (count >= self.Length) return self;

            var result = new HashSet<T>();
            var r = new Random();
            while (result.Count < count) result.Add(self[r.Next(self.Length-1)]);
            return result.ToArray();
        }
        
        public static List<T> GetRandomSubset<T>(this List<T> self, int count)
        {
            if (count >= self.Count) return self;

            var result = new HashSet<T>();
            var r = new Random();
            while (result.Count < count) result.Add(self[r.Next(count)]);
            return result.ToList();
        }

        public static IList<T> GetRandomSubset<T>(this IList<T> self, int count)
        {
            if (count >= self.Count) return self;
            
            var result = new HashSet<T>();
            var r = new Random();
            while (result.Count < count) result.Add(self[r.Next(count)]);
            return result.ToArray();
        }

        public static Tr[] Copy<T, Tr>(this IList<T> array, Func<T, Tr> element_fun)
        {
            var result = new Tr[array.Count];
            for (var i = 0; i < array.Count; i++) result[i] = element_fun(array[i]);
            return result;
        }

        public static Problem ReadProblem(string filename)
        {
            var ys = new List<double>();
            var xss = new List<LibSvm.Node[]>();
            var lines = File.ReadLines(filename);
            foreach (var line in lines)
            {
                var ts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var y = double.Parse(ts[0], CultureInfo.InvariantCulture);
                var xs = new LibSvm.Node[ts.Length - 1];
                for (var i = 1; i < ts.Length; i++)
                {
                    var ns = ts[i].Split(':');
                    var index = int.Parse(ns[0]);
                    var value = double.Parse(ns[1], CultureInfo.InvariantCulture);
                    var n = new LibSvm.Node(index, value);
                    xs[i - 1] = n;
                }

                ys.Add(y);
                xss.Add(xs);
            }

            return new Problem(xss.ToArray(), ys.ToArray());
        }
    }
}
