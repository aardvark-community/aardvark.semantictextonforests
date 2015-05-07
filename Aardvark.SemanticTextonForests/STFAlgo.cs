using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Aardvark.Base;
using Newtonsoft.Json;

namespace Aardvark.SemanticTextonForests
{
    #region STF training

    public static class STFAlgo
    {
        public static Random rand = new Random();
        public static int treeCounter = 0;        //progress counter
        public static int nodeProgressCounter = 0;               //progress report

        public static int nodeIndexCounter = 0;          //the counter variable to determine a node's global index


        public static void train(this STForest forest, STLabelledImage[] trainingImages, TrainingParams parameters)
        {
            nodeIndexCounter = -1;

            Report.BeginTimed(0, "Training Forest of " + forest.SemanticTextons.Length + " trees with " + trainingImages.Length + " images.");

            treeCounter = 0;

            Parallel.ForEach(forest.SemanticTextons, tree =>
            //foreach (var tree in forest.SemanticTextons)
            {
                STLabelledImage[] currentSubset = trainingImages.getRandomSubset(parameters.imageSubsetCount);

                Report.BeginTimed(1, "Training tree " + (tree.Index + 1) + " of " + forest.SemanticTextons.Length + ".");

                tree.train(currentSubset, parameters);

                Report.Line(2, "Finished training tree with " + nodeProgressCounter + " nodes.");

                Report.End(1);
            }
            );

            forest.numNodes = forest.SemanticTextons.Sum(x => x.NumNodes);

            Report.End(0);
        }

        public static void train(this SemanticTexton tree, STLabelledImage[] trainingImages, TrainingParams parameters)
        {
            var nodeCounterObject = new NodeCountObject();
            var provider = parameters.samplingProviderFactory.getNewProvider();
            var baseDPS = provider.getDataPoints(trainingImages);
            var baseClassDist = new ClassDistribution(GlobalParams.labels, baseDPS);

            tree.Root.trainRecursive(null, baseDPS, parameters, 0, baseClassDist, nodeCounterObject);
            tree.NumNodes = nodeCounterObject.counter;

            nodeProgressCounter = nodeCounterObject.counter;
        }

        public static void trainRecursive(this STNode node, STNode parent, DataPointSet currentData, TrainingParams parameters, int depth, ClassDistribution currentClassDist, NodeCountObject currentNodeCounter)
        {
            currentNodeCounter.increment();

            node.GlobalIndex = Interlocked.Increment(ref nodeIndexCounter);

            //create a decider object and train it on the incoming data
            node.Decider = new Decider();

            //only one sampling rule per tree (currently only one exists, regular sampling)
            if(parent == null)
            {
                node.Decider.SamplingProvider = parameters.samplingProviderFactory.getNewProvider();
            }
            else
            {
                node.Decider.SamplingProvider = parent.Decider.SamplingProvider;
            }

            node.ClassDistribution = currentClassDist;

            //get a new feature provider for this node
            node.Decider.FeatureProvider = parameters.featureProviderFactory.getNewProvider();
            node.DistanceFromRoot = depth;
            int newdepth = depth + 1;

            DataPointSet leftRemaining;
            DataPointSet rightRemaining;
            ClassDistribution leftClassDist;
            ClassDistribution rightClassDist;

            //training step: the decider finds the best split threshold for the current data
            DeciderTrainingResult trainingResult = node.Decider.InitializeDecision(currentData, currentClassDist, parameters, out leftRemaining, out rightRemaining, out leftClassDist, out rightClassDist);

            bool passthroughDeactivated = (!parameters.forcePassthrough && trainingResult == DeciderTrainingResult.PassThrough);

            if (trainingResult == DeciderTrainingResult.Leaf //node is a leaf (empty)
                || depth >= parameters.maxTreeDepth - 1        //node is at max level
                || passthroughDeactivated)                   //this node should pass through (information gain too small) but passthrough mode is deactivated
            //-> leaf
            {
                Report.Line(3, "->LEAF remaining dp=" + currentData.DPSet.Length + "; depth=" + depth);
                node.isLeaf = true;
                return;
            }

            if (trainingResult == DeciderTrainingResult.PassThrough) //too small information gain and not at max level -> create pass through node (copy from parent)
            {
                if (parent == null)      //empty tree or empty training set
                {
                    node.isLeaf = true;
                    return;
                }

                Report.Line(3, "PASS THROUGH NODE dp#=" + currentData.DPSet.Length + "; depth=" + depth + " t=" + parent.Decider.DecisionThreshold);

                node.Decider.DecisionThreshold = parent.Decider.DecisionThreshold;
                node.ClassDistribution = parent.ClassDistribution;
                node.Decider.FeatureProvider = parent.Decider.FeatureProvider;
                node.Decider.SamplingProvider = parent.Decider.SamplingProvider;
            }

            var rightNode = new STNode();
            var leftNode = new STNode();

            trainRecursive(rightNode, node, rightRemaining, parameters, newdepth, rightClassDist, currentNodeCounter);
            trainRecursive(leftNode, node, leftRemaining, parameters, newdepth, leftClassDist, currentNodeCounter);

            node.RightChild = rightNode;
            node.LeftChild = leftNode;
        }

        public static STTextonizedLabelledImage[] textonize(this STLabelledImage[] images, STForest forest, TrainingParams parameters)
        {
            var result = new STTextonizedLabelledImage[images.Length];

            Report.BeginTimed(0, "Textonizing " + images.Length + " images.");

            int count = 0;
            Parallel.For(0, images.Length, i =>
            //for (int i = 0; i < images.Length; i++)
            {
                //Report.Progress(0, (double)i / (double)images.Length);
                var img = images[i];

                var dist = forest.getTextonRepresentation(img, parameters);

                result[i] = new STTextonizedLabelledImage(img, dist);

                Report.Line("{0} of {1} images textonized", Interlocked.Increment(ref count), images.Length);
            }
            );

            Report.End(0);

            return result;
        }

        public static STTextonizedLabelledImage[] normalizeInvDocFreq(this STTextonizedLabelledImage[] images)
        {
            //assumes each feature vector has the same length, which is always the case currently
            var result = images;

            //for each column
            for (int i = 0; i < images[0].Textonization.Values.Length; i++)
            {
                double nonzerocounter = 0;
                //for each row
                for (int datapoint = 0; datapoint < images.Length; datapoint++)
                {
                    if(images[datapoint].Textonization.Values[i] != 0)
                    {
                        //nonzerocounter += images[datapoint].Textonization.Values[i];
                        nonzerocounter += 1.0;
                    }
                }
                double normalizationFactor = Math.Log((double)nonzerocounter / (double)images.Length);
                for (int datapoint = 0; datapoint < images.Length; datapoint++)
                {
                    result[datapoint].Textonization.Values[i] *= normalizationFactor;
                }
            }

            return result;
        }

        public enum DeciderTrainingResult
        {
            Leaf,           //become a leaf
            PassThrough,    //copy the parent
            InnerNode       //continue training normally
        }
    }
    #endregion

    #region Temp. Helper Functions

    public class NodeCountObject
    {
        public int counter = 0;
        public void increment()
        {
            counter++;
        }
    }

    internal class RandomSystem : IRandomUniform
    {
        private Random m_r = new Random();

        public bool GeneratesFullDoubles
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public int RandomBits
        {
            get
            {
                throw new NotImplementedException();
            }
        }

        public void ReSeed(int seed)
        {
            m_r = new Random(seed);
        }

        public double UniformDouble()
        {
            return m_r.NextDouble();
        }

        public double UniformDoubleClosed()
        {
            throw new NotImplementedException();
        }

        public double UniformDoubleOpen()
        {
            throw new NotImplementedException();
        }

        public float UniformFloat()
        {
            throw new NotImplementedException();
        }

        public float UniformFloatClosed()
        {
            throw new NotImplementedException();
        }

        public float UniformFloatOpen()
        {
            throw new NotImplementedException();
        }

        public int UniformInt()
        {
            return m_r.Next();
        }

        public long UniformLong()
        {
            throw new NotImplementedException();
        }

        public uint UniformUInt()
        {
            throw new NotImplementedException();
        }

        public ulong UniformULong()
        {
            throw new NotImplementedException();
        }
    }

    internal static class Extensions
    {
        /// <summary>
         /// Enumerates elements in random order.
         /// </summary>
        public static IEnumerable<T> RandomOrder<T>(this IEnumerable<T> self)
        {
            var tmp = self.ToArray();
            var perm = new RandomSystem().CreatePermutationArray(tmp.Length);
            return perm.Select(index => tmp[index]);
        }

    }

    /// <summary>
    /// This interface enforces a common API for random number generators.
    /// </summary>
    public interface IRandomUniform
    {
        #region Info and Seeding

        /// <summary>
        /// Returns the number of random bits that the generator
        /// delivers. This many bits are actually random in the 
        /// doubles returned by <see cref="UniformDouble()"/>.
        /// </summary>
        int RandomBits { get; }

        /// <summary>
        /// Returns true if the doubles generated by this random
        /// generator contain 52 random mantissa bits.
        /// </summary>
        bool GeneratesFullDoubles { get; }

        /// <summary>
        /// Reinitializes the random generator with the specified seed.
        /// </summary>
        void ReSeed(int seed);

        #endregion

        #region Random Integers

        /// <summary>
        /// Returns a uniformly distributed integer in the interval
        /// [0, 2^31-1].
        /// </summary>
        int UniformInt();

        /// <summary>
        /// Returns a uniformly distributed integer in the interval
        /// [0, 2^32-1].
        /// </summary>
        uint UniformUInt();

        /// <summary>
        /// Returns a uniformly distributed integer in the interval
        /// [0, 2^63-1].
        /// </summary>
        long UniformLong();

        /// <summary>
        /// Returns a uniformly distributed integer in the interval
        /// [0, 2^64-1].
        /// </summary>
        ulong UniformULong();

        #endregion

        #region Random Floating Point Values

        /// <summary>
        /// Returns a uniformly distributed float in the half-open interval
        /// [0.0f, 1.0f).
        /// </summary>
        float UniformFloat();

        /// <summary>
        /// Returns a uniformly distributed float in the closed interval
        /// [0.0f, 1.0f].
        /// </summary>
        float UniformFloatClosed();

        /// <summary>
        /// Returns a uniformly distributed float in the open interval
        /// (0.0f, 1.0f).
        /// </summary>
        float UniformFloatOpen();

        /// <summary>
        /// Returns a uniformly distributed double in the half-open interval
        /// [0.0, 1.0). Note, that only RandomBits bits are guaranteed to be
        /// random.
        /// </summary>
        double UniformDouble();

        /// <summary>
        /// Returns a uniformly distributed double in the closed interval
        /// [0.0, 1.0]. Note, that only RandomBits bits are guaranteed to be
        /// random.
        /// </summary>
        double UniformDoubleClosed();

        /// <summary>
        /// Returns a uniformly distributed double in the open interval
        /// (0.0, 1.0). Note, that only RandomBits bits are guaranteed to be
        /// random.
        /// </summary>
        double UniformDoubleOpen();

        #endregion
    }

    public static class IRandomUniformExtensions
    {
        #region Random Bits

        /// <summary>
        /// Supply random bits one at a time. The currently unused bits are
        /// maintained in the supplied reference parameter. Before the first
        /// call randomBits must be 0.
        /// </summary>
        public static bool RandomBit(
               this IRandomUniform rnd, ref int randomBits)
        {
            if (randomBits <= 1)
            {
                randomBits = rnd.UniformInt();
                bool bit = (randomBits & 1) != 0;
                randomBits = 0x40000000 | (randomBits >> 1);
                return bit;
            }
            else
            {
                bool bit = (randomBits & 1) != 0;
                randomBits >>= 1;
                return bit;
            }
        }

        #endregion

        #region Random Integers

        /// <summary>
        /// Returns a uniformly distributed int in the interval [0, count-1].
        /// In order to avoid excessive aliasing, two random numbers are used
        /// when count is greater or equal 2^24 and the random generator
        /// delivers 32 random bits or less. The method thus works fairly
        /// decently for all integers.
        /// </summary>
        public static int UniformInt(this IRandomUniform rnd, int size)
        {
            if (rnd.GeneratesFullDoubles || size < 16777216)
                return (int)(rnd.UniformDouble() * size);
            else
                return (int)(rnd.UniformDoubleFull() * size);
        }

        /// <summary>
        /// Returns a uniformly distributed long in the interval [0, size-1].
        /// NOTE: If count has more than about 48 bits, aliasing leads to 
        /// noticeable (greater 5%) shifts in the probabilities (i.e. one
        /// long has a probability of x and the other a probability of
        /// x * (2^(52-b)-1)/(2^(52-b)), where b is log(size)/log(2)).
        /// </summary>
        public static long UniformLong(this IRandomUniform rnd, long size)
        {
            if (rnd.GeneratesFullDoubles || size < 16777216)
                return (long)(rnd.UniformDouble() * size);
            else
                return (long)(rnd.UniformDoubleFull() * size);
        }

        /// <summary>
        /// Returns a uniform int which is guaranteed not to be zero.
        /// </summary>
        public static int UniformIntNonZero(this IRandomUniform rnd)
        {
            int r;
            do { r = rnd.UniformInt(); } while (r == 0);
            return r;
        }

        /// <summary>
        /// Returns a uniform long which is guaranteed not to be zero.
        /// </summary>
        public static long UniformLongNonZero(this IRandomUniform rnd)
        {
            long r;
            do { r = rnd.UniformLong(); } while (r == 0);
            return r;
        }

        #endregion

        #region Random Floating Point Values

        /// <summary>
        /// Returns a uniformly distributed double in the half-open interval
        /// [0.0, 1.0). Note, that two random values are used to make all 53
        /// bits random. If you use this repeatedly, consider using a 64-bit
        /// random generator such as <see cref="RandomMT64"/>, which can
        /// provide such doubles directly using <see cref="UniformDouble()"/>.
        /// </summary>
        public static double UniformDoubleFull(this IRandomUniform rnd)
        {
            if (rnd.GeneratesFullDoubles) return rnd.UniformDouble();
            long r = ((~0xfL & (long)rnd.UniformInt()) << 22)
                      | ((long)rnd.UniformInt() >> 5);
            return r * (1.0 / 9007199254740992.0);
        }

        /// <summary>
        /// Returns a uniformly distributed double in the closed interval
        /// [0.0, 1.0]. Note, that two random values are used to make all 53
        /// bits random.
        /// </summary>
        public static double UniformDoubleFullClosed(this IRandomUniform rnd)
        {
            if (rnd.GeneratesFullDoubles) return rnd.UniformDoubleClosed();
            long r = ((~0xfL & (long)rnd.UniformInt()) << 22)
                      | ((long)rnd.UniformInt() >> 5);
            return r * (1.0 / 9007199254740991.0);
        }

        /// <summary>
        /// Returns a uniformly distributed double in the open interval
        /// (0.0, 1.0). Note, that two random values are used to make all 53
        /// bits random.
        /// </summary>
        public static double UniformDoubleFullOpen(this IRandomUniform rnd)
        {
            if (rnd.GeneratesFullDoubles) return rnd.UniformDoubleOpen();
            long r;
            do
            {
                r = ((~0xfL & (long)rnd.UniformInt()) << 22)
                    | ((long)rnd.UniformInt() >> 5);
            }
            while (r == 0);
            return r * (1.0 / 9007199254740992.0);
        }

        #endregion

        #region Creating Randomly Filled Arrays

        /// <summary>
        /// Create a random array of doubles in the half-open interval
        /// [0.0, 1.0) of the specified length.
        /// </summary>
        public static double[] CreateUniformDoubleArray(
                this IRandomUniform rnd, long length)
        {
            var array = new double[length];
            rnd.FillUniform(array);
            return array;
        }

        /// <summary>
        /// Create a random array of full doubles in the half-open interval
        /// [0.0, 1.0) of the specified length.
        /// </summary>
        public static double[] CreateUniformDoubleFullArray(
            this IRandomUniform rnd, long length)
        {
            var array = new double[length];
            rnd.FillUniformFull(array);
            return array;
        }

        /// <summary>
        /// Fills the specified array with random ints in the interval
        /// [0, 2^31-1].
        /// </summary>
        public static void FillUniform(this IRandomUniform rnd, int[] array)
        {
            long count = array.LongLength;
            for (long i = 0; i < count; i++)
                array[i] = rnd.UniformInt();
        }

        /// <summary>
        /// Fills the specified array with random floats in the half-open
        /// interval [0.0f, 1.0f).
        /// </summary>
        public static void FillUniform(this IRandomUniform rnd, float[] array)
        {
            long count = array.LongLength;
            for (long i = 0; i < count; i++)
                array[i] = rnd.UniformFloat();
        }

        /// <summary>
        /// Fills the specified array with random doubles in the half-open
        /// interval [0.0, 1.0).
        /// </summary>
        public static void FillUniform(
                this IRandomUniform rnd, double[] array)
        {
            long count = array.LongLength;
            for (long i = 0; i < count; i++)
                array[i] = rnd.UniformDouble();
        }

        /// <summary>
        /// Fills the specified array with fully random doubles (53 random
        /// bits) in the half-open interval [0.0, 1.0).
        /// </summary>
        public static void FillUniformFull(
                this IRandomUniform rnd, double[] array)
        {
            long count = array.LongLength;
            if (rnd.GeneratesFullDoubles)
            {
                for (long i = 0; i < count; i++)
                    array[i] = rnd.UniformDoubleFull();
            }
            else
            {
                for (long i = 0; i < count; i++)
                    array[i] = rnd.UniformDouble();
            }
        }

        /// <summary>
        /// Creates an array that contains a random permutation of the
        /// ints in the interval [0, count-1].
        /// </summary>
        public static int[] CreatePermutationArray(
                this IRandomUniform rnd, int count)
        {
            var p = new int[count].SetByIndex(i => i);
            rnd.Randomize(p);
            return p;
        }

        /// <summary>
        /// Creates an array that contains a random permutation of the
        /// numbers in the interval [0, count-1].
        /// </summary>
        public static long[] CreatePermutationArrayLong(
                this IRandomUniform rnd, long count)
        {
            var p = new long[count].SetByIndexLong(i => i);
            rnd.Randomize(p);
            return p;
        }

        #endregion

        #region Creationg a Random Subset (while maintaing order)

        /// <summary>
        /// Returns a random subset of an array with a supplied number of
        /// elements (subsetCount). The elements in the subset are in the
        /// same order as in the original array. O(count).
        /// NOTE: this method needs to generate one random number for each
        /// element of the original array. If subsetCount is signficantly
        /// smaller than count, it is more efficient to use
        /// <see cref="CreateSmallRandomSubsetIndexArray"/> or
        /// <see cref="CreateSmallRandomSubsetIndexArrayLong"/> or
        /// <see cref="CreateSmallRandomOrderedSubsetIndexArray"/> or
        /// <see cref="CreateSmallRandomOrderedSubsetIndexArrayLong"/>.
        /// </summary>
        public static T[] CreateRandomSubsetOfSize<T>(
                this T[] array, long subsetCount, IRandomUniform rnd)
        {
            long count = array.LongLength;
            Requires.That(subsetCount >= 0 && subsetCount <= count);
            var subset = new T[subsetCount];
            long si = 0;
            for (int ai = 0; ai < count && si < subsetCount; ai++)
            {
                var p = (double)(subsetCount - si) / (double)(count - ai);
                if (rnd.UniformDouble() <= p) subset[si++] = array[ai];
            }
            return subset;
        }

        /// <summary>
        /// Creates an unordered array of subsetCount long indices that
        /// constitute a subset of all longs in the range  [0, count-1].
        /// O(subsetCount) for subsetCount &lt;&lt; count.
        /// NOTE: It is assumed that subsetCount is significantly smaller
        /// than count. If this is not the case, use
        /// <see cref="CreateRandomSubsetOfSize"/> instead.
        /// WARNING: As subsetCount approaches count execution time
        /// increases significantly.
        /// </summary>
        public static long[] CreateSmallRandomSubsetIndexArrayLong(
                this IRandomUniform rnd, long subsetCount, long count)
        {
            Requires.That(subsetCount >= 0 && subsetCount <= count);
            var subsetIndices = new LongSet(subsetCount);
            for (int i = 0; i < subsetCount; i++)
            {
                long index;
                do { index = rnd.UniformLong(count); }
                while (!subsetIndices.TryAdd(index));
            }
            return subsetIndices.ToArray();
        }

        /// <summary>
        /// Creates an ordered array of subsetCount long indices that
        /// constitute a subset of all longs in the range [0, count-1].
        /// O(subsetCount * log(subsetCount)) for subsetCount &lt;&lt; count.
        /// NOTE: It is assumed that subsetCount is significantly smaller
        /// than count. If this is not the case, use
        /// <see cref="CreateRandomSubsetOfSize"/> instead.
        /// WARNING: As subsetCount approaches count execution time
        /// increases significantly.
        /// </summary>
        public static long[] CreateSmallRandomOrderedSubsetIndexArrayLong(
                this IRandomUniform rnd, long subsetCount, long count)
        {
            var subsetIndexArray = rnd.CreateSmallRandomSubsetIndexArrayLong(subsetCount, count);
            subsetIndexArray.QuickSortAscending();
            return subsetIndexArray;
        }

        /// <summary>
        /// Creates an unordered array of subsetCount int indices that
        /// constitute a subset of all ints in the range [0, count-1].
        /// O(subsetCount) for subsetCount &lt;&lt; count.
        /// NOTE: It is assumed that subsetCount is significantly smaller
        /// than count. If this is not the case, use
        /// <see cref="CreateRandomSubsetOfSize"/> instead.
        /// WARNING: As subsetCount approaches count execution time
        /// increases significantly.
        /// </summary>
        public static int[] CreateSmallRandomSubsetIndexArray(
                this IRandomUniform rnd, int subsetCount, int count)
        {
            Requires.That(subsetCount >= 0 && subsetCount <= count);
            var subsetIndices = new IntSet(subsetCount);
            for (int i = 0; i < subsetCount; i++)
            {
                int index;
                do { index = rnd.UniformInt(count); }
                while (!subsetIndices.TryAdd(index));
            }
            return subsetIndices.ToArray();
        }

        /// <summary>
        /// Creates an ordered array of subsetCount int indices that
        /// constitute a subset of all ints in the range [0, count-1].
        /// O(subsetCount * log(subsetCount)) for subsetCount &lt;&lt; count.
        /// NOTE: It is assumed that subsetCount is significantly smaller
        /// than count. If this is not the case, use
        /// <see cref="CreateRandomSubsetOfSize"/> instead.
        /// WARNING: As subsetCount approaches count execution time
        /// increases significantly.
        /// </summary>
        public static int[] CreateSmallRandomOrderedSubsetIndexArray(
                this IRandomUniform rnd, int subsetCount, int count)
        {
            var subsetIndexArray = rnd.CreateSmallRandomSubsetIndexArray(subsetCount, count);
            subsetIndexArray.QuickSortAscending();
            return subsetIndexArray;
        }

        #endregion

        #region Randomizing Existing Arrays

        /// <summary>
        /// Randomly permute the first count elements of the
        /// supplied array. This does work with counts of up
        /// to about 2^50. 
        /// </summary>
        public static void Randomize<T>(
                this IRandomUniform rnd, T[] array, long count)
        {
            if (count <= (long)int.MaxValue)
            {
                int intCount = (int)count;
                for (int i = 0; i < intCount; i++)
                    array.Swap(i, rnd.UniformInt(intCount));
            }
            else
            {
                for (long i = 0; i < count; i++)
                    array.Swap(i, rnd.UniformLong(count));
            }
        }

        /// <summary>
        /// Randomly permute the elements of the supplied array. This does
        /// work with arrays up to a length of about 2^50. 
        /// </summary>
        public static void Randomize<T>(
            this IRandomUniform rnd, T[] array)
        {
            rnd.Randomize(array, array.LongLength);
        }

        /// <summary>
        /// Randomly permute the elements of the supplied list.
        /// </summary>
        public static void Randomize<T>(
            this IRandomUniform rnd, List<T> list)
        {
            int count = list.Count;
            for (int i = 0; i < count; i++)
                list.Swap(i, rnd.UniformInt(count));
        }

        /// <summary>
        /// Randomly permute the specified number of elements in the supplied
        /// array starting at the specified index.
        /// </summary>
        public static void Randomize<T>(
            this IRandomUniform rnd, T[] array, int start, int count)
        {
            for (int i = start, e = start + count; i < e; i++)
                array.Swap(i, start + rnd.UniformInt(count));
        }

        /// <summary>
        /// Randomly permute the specified number of elements in the supplied
        /// array starting at the specified index.
        /// </summary>
        public static void Randomize<T>(
            this IRandomUniform rnd, T[] array, long start, long count)
        {
            for (long i = start, e = start + count; i < e; i++)
                array.Swap(i, start + rnd.UniformLong(count));
        }

        /// <summary>
        /// Randomly permute the specified number of elements in the supplied
        /// list starting at the specified index.
        /// </summary>
        public static void Randomize<T>(
            this IRandomUniform rnd, List<T> list, int start, int count)
        {
            for (int i = start; i < start + count; i++)
                list.Swap(i, start + rnd.UniformInt(count));
        }

        #endregion

        #region V2d

        public static V2d UniformV2d(this IRandomUniform rnd)
        {
            return new V2d(rnd.UniformDouble(),
                           rnd.UniformDouble());
        }

        public static V2d UniformV2dOpen(this IRandomUniform rnd)
        {
            return new V2d(rnd.UniformDoubleOpen(),
                           rnd.UniformDoubleOpen());
        }

        public static V2d UniformV2dClosed(this IRandomUniform rnd)
        {
            return new V2d(rnd.UniformDoubleClosed(),
                           rnd.UniformDoubleClosed());
        }

        public static V2d UniformV2dFull(this IRandomUniform rnd)
        {
            return new V2d(rnd.UniformDoubleFull(),
                           rnd.UniformDoubleFull());
        }

        public static V2d UniformV2dFullOpen(this IRandomUniform rnd)
        {
            return new V2d(rnd.UniformDoubleFullOpen(),
                           rnd.UniformDoubleFullOpen());
        }

        public static V2d UniformV2dFullClosed(this IRandomUniform rnd)
        {
            return new V2d(rnd.UniformDoubleFullClosed(),
                           rnd.UniformDoubleFullClosed());
        }

        public static V2d UniformV2d(this IRandomUniform rnd, Box2d box)
        {
            return new V2d(box.Min.X + rnd.UniformDouble() * (box.Max.X - box.Min.X),
                           box.Min.Y + rnd.UniformDouble() * (box.Max.Y - box.Min.Y));

        }

        public static V2d UniformV2dOpen(this IRandomUniform rnd, Box2d box)
        {
            return new V2d(box.Min.X + rnd.UniformDoubleOpen() * (box.Max.X - box.Min.X),
                           box.Min.Y + rnd.UniformDoubleOpen() * (box.Max.Y - box.Min.Y));

        }

        public static V2d UniformV2dClosed(this IRandomUniform rnd, Box2d box)
        {
            return new V2d(box.Min.X + rnd.UniformDoubleClosed() * (box.Max.X - box.Min.X),
                           box.Min.Y + rnd.UniformDoubleClosed() * (box.Max.Y - box.Min.Y));

        }

        public static V2d UniformV2dFull(this IRandomUniform rnd, Box2d box)
        {
            return new V2d(box.Min.X + rnd.UniformDoubleFull() * (box.Max.X - box.Min.X),
                           box.Min.Y + rnd.UniformDoubleFull() * (box.Max.Y - box.Min.Y));

        }

        public static V2d UniformV2dFullOpen(this IRandomUniform rnd, Box2d box)
        {
            return new V2d(box.Min.X + rnd.UniformDoubleFullOpen() * (box.Max.X - box.Min.X),
                           box.Min.Y + rnd.UniformDoubleFullOpen() * (box.Max.Y - box.Min.Y));

        }

        public static V2d UniformV2dFullClosed(this IRandomUniform rnd, Box2d box)
        {
            return new V2d(box.Min.X + rnd.UniformDoubleFullClosed() * (box.Max.X - box.Min.X),
                           box.Min.Y + rnd.UniformDoubleFullClosed() * (box.Max.Y - box.Min.Y));

        }

        public static V2d UniformV2dDirection(this IRandomUniform rnd)
        {
            double phi = rnd.UniformDouble() * Constant.PiTimesTwo;
            return new V2d(System.Math.Cos(phi),
                           System.Math.Sin(phi));
        }

        #endregion

        #region C3f

        public static C3f UniformC3f(this IRandomUniform rnd)
        {
            return new C3f(rnd.UniformFloat(),
                           rnd.UniformFloat(),
                           rnd.UniformFloat());
        }

        public static C3f UniformC3fOpen(this IRandomUniform rnd)
        {
            return new C3f(rnd.UniformFloatOpen(),
                           rnd.UniformFloatOpen(),
                           rnd.UniformFloatOpen());
        }

        public static C3f UniformC3fClosed(this IRandomUniform rnd)
        {
            return new C3f(rnd.UniformFloatClosed(),
                           rnd.UniformFloatClosed(),
                           rnd.UniformFloatClosed());
        }

        #endregion

        #region V3d

        public static V3d UniformV3d(this IRandomUniform rnd)
        {
            return new V3d(rnd.UniformDouble(),
                           rnd.UniformDouble(),
                           rnd.UniformDouble());
        }

        public static V3d UniformV3dOpen(this IRandomUniform rnd)
        {
            return new V3d(rnd.UniformDoubleOpen(),
                           rnd.UniformDoubleOpen(),
                           rnd.UniformDoubleOpen());
        }

        public static V3d UniformV3dClosed(this IRandomUniform rnd)
        {
            return new V3d(rnd.UniformDoubleClosed(),
                           rnd.UniformDoubleClosed(),
                           rnd.UniformDoubleClosed());
        }

        public static V3d UniformV3dFull(this IRandomUniform rnd)
        {
            return new V3d(rnd.UniformDoubleFull(),
                           rnd.UniformDoubleFull(),
                           rnd.UniformDoubleFull());
        }

        public static V3d UniformV3dFullOpen(this IRandomUniform rnd)
        {
            return new V3d(rnd.UniformDoubleFullOpen(),
                           rnd.UniformDoubleFullOpen(),
                           rnd.UniformDoubleFullOpen());
        }

        public static V3d UniformV3dFullClosed(this IRandomUniform rnd)
        {
            return new V3d(rnd.UniformDoubleFullClosed(),
                           rnd.UniformDoubleFullClosed(),
                           rnd.UniformDoubleFullClosed());
        }

        public static V3d UniformV3d(this IRandomUniform rnd, Box3d box)
        {
            return box.Lerp(rnd.UniformDouble(),
                            rnd.UniformDouble(),
                            rnd.UniformDouble());
        }

        public static V3d UniformV3dOpen(this IRandomUniform rnd, Box3d box)
        {
            return box.Lerp(rnd.UniformDoubleOpen(),
                            rnd.UniformDoubleOpen(),
                            rnd.UniformDoubleOpen());
        }

        public static V3d UniformV3dClosed(this IRandomUniform rnd, Box3d box)
        {
            return box.Lerp(rnd.UniformDoubleClosed(),
                            rnd.UniformDoubleClosed(),
                            rnd.UniformDoubleClosed());
        }

        public static V3d UniformV3dFull(this IRandomUniform rnd, Box3d box)
        {
            return box.Lerp(rnd.UniformDoubleFull(),
                            rnd.UniformDoubleFull(),
                            rnd.UniformDoubleFull());
        }

        public static V3d UniformV3dFullOpen(this IRandomUniform rnd, Box3d box)
        {
            return box.Lerp(rnd.UniformDoubleFullOpen(),
                            rnd.UniformDoubleFullOpen(),
                            rnd.UniformDoubleFullOpen());
        }

        public static V3d UniformV3dFullClosed(this IRandomUniform rnd, Box3d box)
        {
            return box.Lerp(rnd.UniformDoubleFullClosed(),
                            rnd.UniformDoubleFullClosed(),
                            rnd.UniformDoubleFullClosed());
        }

        /// <summary>
        /// Returns a uniformly distributed vecctor (corresponds to a
        /// uniformly distributed point on the unit sphere). Note however,
        /// that the returned vector will never be equal to [0, 0, -1].
        /// </summary>
        public static V3d UniformV3dDirection(this IRandomUniform rnd)
        {
            double phi = rnd.UniformDouble() * Constant.PiTimesTwo;
            double z = 1.0 - rnd.UniformDouble() * 2.0;
            double s = System.Math.Sqrt(1.0 - z * z);
            return new V3d(System.Math.Cos(phi) * s, System.Math.Sin(phi) * s, z);
        }

        public static V3d UniformV3dFullDirection(this IRandomUniform rnd)
        {
            double phi = rnd.UniformDoubleFull() * Constant.PiTimesTwo;
            double z = 1.0 - rnd.UniformDoubleFull() * 2.0;
            double s = System.Math.Sqrt(1.0 - z * z);
            return new V3d(System.Math.Cos(phi) * s, System.Math.Sin(phi) * s, z);
        }

        #endregion
    }

    public static class HelperFunctions     //temporary helper functions
    {
        public static void splitIntoSets(this STLabelledImage[] images, out STLabelledImage[] training, out STLabelledImage[] test)
        {
            ////50/50 split
            var tro = new List<STLabelledImage>();
            var teo = new List<STLabelledImage>();

            foreach (var img in images)
            {
                var rn = STFAlgo.rand.NextDouble();
                if (rn >= 0.5)
                {
                    tro.Add(img);
                }
                else
                {
                    teo.Add(img);
                }
            }

            training = tro.ToArray();
            test = teo.ToArray();
        }

        public static STLabelledImage[] getRandomSubset(this STLabelledImage[] images, int size)
        {
            return images.RandomOrder().Take(size).ToArray();
        }

        public static double toDouble(this byte par)
        {
            return ((double)par) * (1d / 255d);
        }

        public static void writeToFile(this STForest forest, string filename)
        {
            Report.Line(2, "Writing forest to file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full

            };


            var s = JsonConvert.SerializeObject(forest, Formatting.Indented, settings);

            File.WriteAllText(filename, s);

        }

        public static STForest readForestFromFile(string filename)
        {
            Report.Line(2, "Reading forest from file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
            };

            var parsed = JsonConvert.DeserializeObject<STForest>(File.ReadAllText(filename), settings);
            return parsed;
        }

        public static void writeToFile(this STTextonizedLabelledImage[] images, string filename)
        {
            Report.Line(2, "Writing textonized image set to file " + filename);

            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
                TypeNameAssemblyFormat = System.Runtime.Serialization.Formatters.FormatterAssemblyStyle.Full

            };


            var s = JsonConvert.SerializeObject(images, Formatting.Indented, settings);

            File.WriteAllText(filename, s);
        }

        public static STTextonizedLabelledImage[] readTextonizedImagesFromFile(string filename)
        {
            Report.Line(2, "Reading textonized image set from file " + filename);
            var settings = new JsonSerializerSettings
            {
                TypeNameHandling = TypeNameHandling.All,
            };

            var parsed = JsonConvert.DeserializeObject<STTextonizedLabelledImage[]>(File.ReadAllText(filename), settings);
            return parsed;
        }

        //creates a forest and saves it to file
        public static void createNewForestAndSaveToFile(string filename, STLabelledImage[] trainingSet, TrainingParams parameters)
        {
            STForest forest = new STForest(parameters.forestName);
            Report.Line(2, "Creating new forest " + forest.name + ".");
            forest.InitializeEmptyForest(parameters.treesCount);
            forest.train(trainingSet, parameters);

            Report.Line(2, "Saving forest " + forest.name + " to file.");
            forest.writeToFile(filename);
        }

        public static STForest createNewForest(STLabelledImage[] trainingSet, TrainingParams parameters)
        {
            STForest forest = new STForest(parameters.forestName);
            Report.Line(2, "Creating new forest " + forest.name + ".");
            forest.InitializeEmptyForest(parameters.treesCount);
            forest.train(trainingSet, parameters);

            return forest;
        }

        //textonizes images and saves the array to file
        public static void createTextonizationAndSaveToFile(string filename, STForest forest, STLabelledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.textonize(forest, parameters);


            Report.Line(2, "Saving textonization to file.");
            texImgs.writeToFile(filename);
        }

        public static STTextonizedLabelledImage[] createTextonization(STForest forest, STLabelledImage[] imageSet, TrainingParams parameters)
        {
            var texImgs = imageSet.textonize(forest, parameters);

            return texImgs;
        }

        //string formatting -> add spaces until the total length of the string = number
        public static string spaces(string prevValue, int number)
        {
            int numchar = prevValue.Length;
            var result = new StringBuilder();
            result.Append(prevValue);

            for (int i = prevValue.Length; i < number; i++)
            {
                result.Append(" ");
            }

            return result.ToString();
        }
    }

    #endregion

    #region Providers

    public enum ClassificationMode
    {
        LeafOnly,
        Semantic
    }

    public enum FeatureType
    {
        RandomPixelValue,
        RandomTwoPixelSum,
        RandomTwoPixelDifference,
        RandomTwoPixelAbsDiff,
        SelectRandom
    };

    public enum SamplingType
    {
        RegularGrid,
        RandomPoints
    };

    public class FeatureProviderFactory
    {
        IFeatureProvider currentProvider;
        int pixWinSize;
        FeatureType currentChoice;

        public void selectProvider(FeatureType featureType, int pixelWindowSize)
        {
            currentChoice = featureType;
            pixWinSize = pixelWindowSize;
        }

        public IFeatureProvider getNewProvider()
        {
            switch (currentChoice)
            {
                case FeatureType.RandomPixelValue:
                    currentProvider = new ValueOfPixelFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.RandomTwoPixelSum:
                    currentProvider = new PixelSumFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.RandomTwoPixelAbsDiff:
                    currentProvider = new AbsDiffOfPixelFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.RandomTwoPixelDifference:
                    currentProvider = new PixelDifferenceFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;
                case FeatureType.SelectRandom:      //select one of the three providers at random - equal chance
                    var choice = STFAlgo.rand.Next(4);
                    switch(choice)
                    {
                        case 0:
                            currentProvider = new ValueOfPixelFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        case 1:
                            currentProvider = new PixelSumFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        case 2:
                            currentProvider = new AbsDiffOfPixelFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        case 3:
                            currentProvider = new PixelDifferenceFeatureProvider();
                            currentProvider.Init(pixWinSize);
                            break;
                        default:
                            return null;
                    }
                    break;
                default:
                    currentProvider = new ValueOfPixelFeatureProvider();
                    currentProvider.Init(pixWinSize);
                    break;

            }
            return currentProvider;
        }
    }

    public class PixelSumFeatureProvider : IFeatureProvider
    {
        public int fX;
        public int fY;
        public int sX;
        public int sY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(fX, fY); }
            set { fX = value.X; fY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(sX, sY); }
            set { sX = value.X; sY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int firstY = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondY = STFAlgo.rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().toDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().toDouble();

            var op = (sample1 + sample2) / 2.0; //divide by two for normalization

            result.Value = op;

            return result;
        }
    }

    public class PixelDifferenceFeatureProvider : IFeatureProvider
    {
        public int fX;
        public int fY;
        public int sX;
        public int sY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(fX, fY); }
            set { fX = value.X; fY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(sX, sY); }
            set { sX = value.X; sY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int firstY = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondY = STFAlgo.rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().toDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().toDouble();

            var op = ((sample1 - sample2)+1.0) / 2.0; //normalize to [0,1]

            result.Value = op;

            return result;
        }
    }

    public class ValueOfPixelFeatureProvider : IFeatureProvider
    {
        public int X;
        public int Y;

        [JsonIgnore]
        public V2i pixelOffset
        {
            get { return new V2i(X, Y); }
            set { X = value.X; Y = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {

            int half = (int)(pixelWindowSize / 2);
            int x = STFAlgo.rand.Next(pixelWindowSize) - half;
            int y = STFAlgo.rand.Next(pixelWindowSize) - half;

            pixelOffset = new V2i(x, y);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample = pi[point.PixelCoords + pixelOffset].ToGrayByte().toDouble();

            result.Value = sample;

            return result;
        }
    }

    public class AbsDiffOfPixelFeatureProvider : IFeatureProvider
    {
        public int fX;
        public int fY;
        public int sX;
        public int sY;

        [JsonIgnore]
        public V2i FirstPixelOffset
        {
            get { return new V2i(fX, fY); }
            set { fX = value.X; fY = value.Y; }
        }

        [JsonIgnore]
        public V2i SecondPixelOffset
        {
            get { return new V2i(sX, sY); }
            set { sX = value.X; sY = value.Y; }
        }

        public override void Init(int pixelWindowSize)
        {
            //note: could be the same pixel.

            int half = (int)(pixelWindowSize / 2);
            int firstX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int firstY = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondX = STFAlgo.rand.Next(pixelWindowSize) - half;
            int secondY = STFAlgo.rand.Next(pixelWindowSize) - half;

            FirstPixelOffset = new V2i(firstX, firstY);
            SecondPixelOffset = new V2i(secondX, secondY);
        }

        public override STFeature getFeature(DataPoint point)
        {
            STFeature result = new STFeature();

            var pi = point.Image.PixImage.GetMatrix<C3b>();

            var sample1 = pi[point.PixelCoords + FirstPixelOffset].ToGrayByte().toDouble();
            var sample2 = pi[point.PixelCoords + SecondPixelOffset].ToGrayByte().toDouble();

            var op = Math.Abs(sample2 - sample1);

            result.Value = op;

            return result;
        }
    }


    public class SamplingProviderFactory
    {
        ISamplingProvider currentProvider;
        int pixelWindowSize;
        SamplingType samplingType;
        int randomSampleCount = 0;

        public void selectProvider(SamplingType samplingType, int pixelWindowSize)
        {
            this.pixelWindowSize = pixelWindowSize;
            this.samplingType = samplingType;
        }

        public void selectProvider(SamplingType samplingType, int pixelWindowSize, int randomSampleCount)
        {
            this.pixelWindowSize = pixelWindowSize;
            this.samplingType = samplingType;
            this.randomSampleCount = randomSampleCount;
        }

        public ISamplingProvider getNewProvider()
        {

            switch (samplingType)
            {
                case SamplingType.RegularGrid:
                    currentProvider = new RegularGridSamplingProvider();
                    currentProvider.init(pixelWindowSize);
                    break;
                case SamplingType.RandomPoints:
                    var result = new RandomPointSamplingProvider();
                    result.init(pixelWindowSize);
                    result.SampleCount = this.randomSampleCount;
                    currentProvider = result;
                    break;
                default:
                    currentProvider = new RegularGridSamplingProvider();
                    currentProvider.init(pixelWindowSize);
                    break;
            }

            return currentProvider;
        }
    }

    public class RegularGridSamplingProvider : ISamplingProvider
    {
        public int pixWinSize;

        public override void init(int pixWindowSize)
        {
            pixWinSize = pixWindowSize;
        }

        public override DataPointSet getDataPoints(STImage image)
        {
            //currently, this gets a regular grid starting from the top left and continuing as long as there are pixels left.
            var pi = image.PixImage.GetMatrix<C3b>();

            List<DataPoint> result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling((double)pixWinSize / 2.0f); //ceiling cuts away too much in most cases

            int pointCounter = 0;

            for (int x = borderOffset; x < pi.SX - borderOffset; x = x + pixWinSize)
            {
                for (int y = borderOffset; y < pi.SY - borderOffset; y = y + pixWinSize)
                {
                    var newDP = new DataPoint()
                    {
                        PixelCoords = new V2i(x, y),
                        Image = image
                    };
                    result.Add(newDP);
                    pointCounter = pointCounter + 1;
                }
            }

            var bias = (double)1 / (double)pointCounter;     //weigh the sample points by the image's size (larger image = lower weight)

            var resDPS = new DataPointSet();
            resDPS.DPSet = result.ToArray();
            resDPS.SetWeight = bias;

            return resDPS;
        }

        public override DataPointSet getDataPoints(STLabelledImage[] images)
        {
            var result = new DataPointSet();
            
            foreach(var img in images)
            {
                var currentDPS = getDataPoints(img);
                foreach(var dp in currentDPS.DPSet)
                {
                    dp.label = img.ClassLabel.Index;
                }
                result = result + currentDPS;
            }
            

            return result;
        }

    }

    public class RandomPointSamplingProvider : ISamplingProvider
    {
        public int pixWinSize;
        public int SampleCount;

        public override void init(int pixWindowSize)
        {
            pixWinSize = pixWindowSize;
        }

        public override DataPointSet getDataPoints(STImage image)
        {
            //Gets random points within the usable area of the image (= image with a border respecting the feature window)
            var pi = image.PixImage.GetMatrix<C3b>();

            List<DataPoint> result = new List<DataPoint>();

            var borderOffset = (int)Math.Ceiling((double)pixWinSize / 2.0d);

            for (int i = 0; i < SampleCount; i++)
            {
                var x = STFAlgo.rand.Next(borderOffset, (int)pi.SX - borderOffset);
                var y = STFAlgo.rand.Next(borderOffset, (int)pi.SY - borderOffset);

                var newDP = new DataPoint()
                {
                    PixelCoords = new V2i(x, y),
                    Image = image
                };
                result.Add(newDP);

            }


            var resDPS = new DataPointSet();
            resDPS.DPSet = result.ToArray();
            resDPS.SetWeight = 1.0d;

            return resDPS;
        }

        public override DataPointSet getDataPoints(STLabelledImage[] labelledImages)
        {
            throw new NotImplementedException();
        }

    }

    #endregion
}
