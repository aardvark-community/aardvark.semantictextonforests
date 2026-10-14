using System;
using System.IO;
using Aardvark.SemanticTextonForests;
using Newtonsoft.Json;
using Xunit;

namespace Tests
{
    public class SerializationTests : IDisposable
    {
        private readonly string directory = Path.Combine(Path.GetTempPath(), "stf-json-" + Guid.NewGuid().ToString("N"));

        public SerializationTests()
        {
            Directory.CreateDirectory(directory);
        }

        [Fact]
        public void CanReadAndRoundTripLegacyForest()
        {
            var forest = HelperFunctions.ReadForestFromFile(Fixture("forest.json"));

            Assert.Equal(3, forest.NumNodes);
            Assert.Equal(1, forest.Trees.Length);
            Assert.Equal(3, forest.Trees[0].NumNodes);
            var root = forest.Trees[0].Root;
            Assert.Equal(new[] { 2.0, 1.0 }, root.ClassDistribution.Distribution);
            Assert.Equal(1, root.LeftChild.GlobalIndex);
            Assert.True(root.LeftChild.IsLeaf);
            Assert.Equal(2, root.RightChild.GlobalIndex);
            Assert.True(root.RightChild.IsLeaf);
            var provider = Assert.IsType<ValueOfPixelFeatureProvider>(root.Decider.FeatureProvider);
            Assert.Equal(0, provider.X);
            Assert.Equal(0, provider.Y);
            var path = Path.Combine(directory, "forest.json");

            forest.WriteToFile(path);

            var loaded = HelperFunctions.ReadForestFromFile(path);
            Assert.IsType<ValueOfPixelFeatureProvider>(loaded.Trees[0].Root.Decider.FeatureProvider);
            Assert.Equal(JsonConvert.SerializeObject(forest), JsonConvert.SerializeObject(loaded));
        }

        [Fact]
        public void CanRoundTripForestAtExampleDepth()
        {
            // PredictionTest in Examples/Program.cs uses a maximum tree depth of 25.
            var forest = new Forest("depth test", 1) { NumNodes = 49 };
            forest.Trees[0].NumNodes = 49;
            var node = forest.Trees[0].Root;
            for (var depth = 1; depth < 25; depth++)
            {
                node.LeftChild = new Node { DistanceFromRoot = depth };
                node.RightChild = new Node { DistanceFromRoot = depth, IsLeaf = true };
                node = node.LeftChild;
            }
            node.IsLeaf = true;
            var path = Path.Combine(directory, "forest.json");
            forest.WriteToFile(path);
            Assert.Contains(typeof(Forest).Assembly.FullName, File.ReadAllText(path));

            var loaded = HelperFunctions.ReadForestFromFile(path);

            Assert.Equal(49, loaded.NumNodes);
            Assert.Equal(49, loaded.Trees[0].NumNodes);
            node = loaded.Trees[0].Root;
            for (var depth = 0; depth < 24; depth++)
            {
                Assert.Equal(depth, node.DistanceFromRoot);
                Assert.True(node.RightChild.IsLeaf);
                node = node.LeftChild;
            }
            Assert.Equal(24, node.DistanceFromRoot);
            Assert.True(node.IsLeaf);
        }

        [Fact]
        public void CanReadAndRoundTripLegacyEmptyImages()
        {
            var images = HelperFunctions.ReadTextonizedImagesFromFile(Fixture("images-empty.json"));
            Assert.Empty(images);
            var path = Path.Combine(directory, "images.json");

            images.WriteToFile(path);

            Assert.Empty(HelperFunctions.ReadTextonizedImagesFromFile(path));
        }

        [Fact(Skip = "Non-empty imports already fail with Json.NET 6.0.8; see docs/ideas/json-round-trip-data-loss.md.")]
        public void CanReadLegacyNonEmptyImages()
        {
            var images = HelperFunctions.ReadTextonizedImagesFromFile(Fixture("images.json"));

            Assert.Equal(1, images.Length);
            Assert.Equal("fixture.png", images[0].Image.Image.ImagePath);
            Assert.Equal("fixture", images[0].Label.Name);
            Assert.Equal(2.5, images[0].Textonization.Histogram[0].Value);
        }

        [Fact]
        public void CanReadAndRoundTripLegacyResults()
        {
            var result = TestSeries.LoadFromJSON(Fixture("result.json"));
            Assert.Equal("legacy result", result.OutputString);
            Assert.Equal(0, result.IndexOfBestParams);
            Assert.Equal(1, result.TestCaseResults.Length);
            Assert.Empty(result.TestCaseTrainingparams);
            var testCase = result.TestCaseResults[0];
            Assert.Equal("legacy case", testCase.Name);
            Assert.Equal(0.5, testCase.TestSetResult.Precision);
            Assert.Equal(new[] { 0, 1 }, testCase.TestSetResult.PredictedClassLabelIndices);
            Assert.Equal(1, testCase.TestSetResult.NumCorrect);
            Assert.Equal(1, testCase.TestSetResult.NumWrong);
            var path = Path.Combine(directory, "result.json");

            File.WriteAllText(path, result.writeJSON());

            Assert.Equal(JsonConvert.SerializeObject(result), JsonConvert.SerializeObject(TestSeries.LoadFromJSON(path)));
        }

        [Theory]
        [InlineData("forest")]
        [InlineData("images")]
        [InlineData("result")]
        public void RejectsExcessiveNestingInEveryImportPath(string kind)
        {
            // Small, valid JSON beyond Json.NET 13's default maximum depth of 64.
            // An unknown property exercises the reader's skip path too.
            var json = "{\"Unknown\":" + new string('[', 65) + "0" + new string(']', 65) + "}";
            if (kind == "images") json = "[" + json + "]";
            var path = Path.Combine(directory, "nested.json");
            File.WriteAllText(path, json);

            Assert.Throws<JsonReaderException>(() =>
            {
                if (kind == "forest") HelperFunctions.ReadForestFromFile(path);
                else if (kind == "images") HelperFunctions.ReadTextonizedImagesFromFile(path);
                else TestSeries.LoadFromJSON(path);
            });
        }

        private static string Fixture(string filename)
        {
            return Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Fixtures", "Newtonsoft.Json.6.0.8", filename);
        }

        public void Dispose()
        {
            Directory.Delete(directory, true);
        }
    }
}
