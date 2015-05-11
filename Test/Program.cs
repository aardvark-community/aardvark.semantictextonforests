using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibSvm;
using Aardvark.Base;

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

        static IEnumerable<Parameter> Search()
        {
            for (var gamma = 0.0; gamma < 5.0; gamma += 0.001)
            {
                for (var C = 2.0; C <= 200.0; C += 0.1)
                {
                    yield return new Parameter
                    {
                        SvmType = SvmType.C_SVC,
                        KernelType = KernelType.RBF,
                        Degree = 0,
                        Gamma = gamma,
                        Coef0 = 0,
                        CacheSize = 1000,
                        Eps = 0.001,
                        C = C,
                        Weight = new double[0],
                        WeightLabel = new int[0],
                        Nu = 0,
                        p = 0.1,
                        Shrinking = 1,
                        Probability = 0
                    };
                }
            }
        }

        static void Main(string[] args)
        {
            var heart_scale = ReadProblem(@"C:\Data\Development\libsvm\heart_scale");

            var parameter = new Parameter
            {
                SvmType = SvmType.C_SVC,
                KernelType = KernelType.SIGMOID,
                Degree = 0,
                Gamma = 0.1,
                Coef0 = 0,
                CacheSize = 1000,
                Eps = 0.001,
                C = 100,
                Weight = new double[0],
                WeightLabel = new int[0],
                Nu = 0,
                p = 0.1,
                Shrinking = 1,
                Probability = 0
            };

            Console.WriteLine("check: '{0}'", Svm.CheckParameter(heart_scale, parameter));

            //var learn = new Problem(
            //    heart_scale.x.TakePeriodic(2).ToArray(),
            //    heart_scale.y.TakePeriodic(2).ToArray()
            //    );
            //var model = Svm.Train(learn, parameter);

            var bestNok = heart_scale.Count + 1;
            foreach (var p in Search())
            {
                var model = Svm.Train(heart_scale, p);
                
                //var validation = Svm.CrossValidation(heart_scale, parameter, 10);

                
                var ok = 0;
                var nok = 0;
                for (var i = 0; i < heart_scale.x.Length; i++)
                {
                    var prediction = Svm.Predict(model, heart_scale.x[i]);
                    if (prediction == heart_scale.y[i]) ok++; else nok++;
                    //Console.WriteLine($"{heart_scale.y[i],3}    {prediction,-3}");
                }
                if (nok < bestNok)
                {
                    bestNok = nok;
                    Console.WriteLine($"ok: {ok}, nok: {nok}");
                    Console.WriteLine($"gamma = {p.Gamma}, C = {p.C}");
                }
                
            }
        }
    }
}
