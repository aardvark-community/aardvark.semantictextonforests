using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibSvm;

namespace Test
{
    class Program
    {
        static Problem ReadProblem(string filename)
        {
            var ys = new List<double>();
            var xss = new List<Node[]>();
            var lines = File.ReadLines(filename);
            foreach (var line in lines)
            {
                var ts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                var y = double.Parse(ts[0], CultureInfo.InvariantCulture);
                var xs = new Node[ts.Length - 1];
                for (var i = 1; i < ts.Length; i++)
                {
                    var ns = ts[i].Split(':');
                    var index = int.Parse(ns[0]);
                    var value = double.Parse(ns[1], CultureInfo.InvariantCulture);
                    var n = new Node(index, value);
                    xs[i - 1] = n;
                }

                ys.Add(y);
                xss.Add(xs);
            }

            return new Problem(xss.ToArray(), ys.ToArray());
        }

        static void Main(string[] args)
        {
            var heart_scale = ReadProblem(@"C:\Data\Development\libsvm\heart_scale");

            //var problem = new Problem
            //{
            //    y = new double[] { 11.11, 22.22 },
            //    x = new[]
            //    {
            //        new [] {
            //            new Node { Index = 1, Value = 3.14 },
            //            new Node { Index = 2, Value = 3.15 }
            //        },
            //        new [] {
            //            new Node { Index = 3, Value = 4.15 },
            //            new Node { Index = 4, Value = 5.16 },
            //            new Node { Index = 5, Value = 5.17 }
            //        }
            //    }
            //};

            var parameter = new Parameter
            {
                SvmType = SvmType.C_SVC,
                KernelType = KernelType.LINEAR,
                Degree = 0,
                Gamma = 0,
                Coef0 = 0,
                CacheSize = 1000,
                Eps = 0.001,
                C = 2,
                Weight = new double[0],
                WeightLabel = new int[0],
                Nu = 0,
                p = 0.1,
                Shrinking = 1,
                Probability = 0
            };

            Console.WriteLine("check: '{0}'", Svm.CheckParameter(heart_scale, parameter));

            var foo = Svm.Train(heart_scale, parameter);

            var bar = Svm.CrossValidation(heart_scale, parameter, 2);
            foreach (var x in bar) Console.WriteLine("cross consolidation: {0}", x);
        }
    }
}
