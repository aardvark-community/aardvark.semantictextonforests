using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using LibLinear;
using Xunit;

namespace Tests
{
    public class LibLinearTests
    {
        [Fact]
        public void CanCreateNode()
        {
            var n = new Node(0, 1.23);
            Assert.True(n.Index == 0);
            Assert.True(n.Value == 1.23);
        }

        [Fact]
        public void CanCreateProblem()
        {
            var trainingVectors = new[]
            {
                new [] { new Node(0, 1.0), new Node(1, 2.0), new Node(3, 4.0) },
                new [] { new Node(0, 3.0), new Node(2, 5.0), new Node(3, 6.0) },
                new [] { new Node(1, 8.0), new Node(2, 7.0) },
            };
            var targetValues = new[] { 2.1, 3.2, 4.3 };
            var problem = new Problem(trainingVectors, targetValues, -1.0);
            Assert.True(problem.Count == 3);
            Assert.True(problem.y.Length == 3);
            Assert.True(problem.x.Length == 3);
        }
    }
}
