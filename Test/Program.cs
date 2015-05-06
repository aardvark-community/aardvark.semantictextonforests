using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibSvm;

namespace Test
{
    class Program
    {
        static void Main(string[] args)
        {
            var p = new Problem
            {
                y = new double[] { 11.11, 22.22 },
                x = new[]
                {
                    new [] {
                        new Node { Index = 1, Value = 3.14 }
                    },
                    new [] {
                        new Node { Index = 2, Value = 4.15 },
                        new Node { Index = 3, Value = 5.16 }
                    }
                }
            };

            var config = new Parameter
            {
                SvmType = SvmType.C_SVC,
                KernelType = KernelType.LINEAR,
                Degree = 0,
                Gamma = 0,
                Coef0 = 0,
                CacheSize = 1000,
                Eps = 0.01,
                C = 0,
                Weight = new double[0],
                WeightLabel = new int[0],
                Nu = 0,
                p = 0,
                Shrinking = 0,
                Probability = 0
            };

            var foo = Svm.Train(p, config);
        }
    }
}
