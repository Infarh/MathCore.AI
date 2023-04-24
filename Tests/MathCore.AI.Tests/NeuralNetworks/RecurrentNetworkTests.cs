namespace MathCore.AI.Tests.NeuralNetworks;

[TestClass]
public class RecurrentNetworkTests
{
    private static (double[][,] Weights, double[][,] Feedbacks) GetTestNetworkStructure()
    {
        double[][,] weights =
        {
            new [,]
            {
                { -2.386, 0, 0 },
                { 0, -0.702735, 0 },
                {  0,  0,  -0.19817829729727853934 },
                {  0.3862943611198906188 , 0,  0 }
            },
            new [,]
            {
                {  1,  0,  -1.32422481981972603,  0 },
                { 0, 3.5, 0, -4.7328679513998632735 }
            },
            new [,]
            {
                { 2, -17.931471805599453094 },
                { -1, -2 },
                { 1.643823935199817698, -3 }
            }
        };

        double[][,] feedbacks =
        {
            new [,]
            {
                { 0.01, 0.02, 0.03, 0.04 },
                { 0.05, 0.06, 0.07, 0.08 },
                { -.05, -.06, -.07, -.07 },
                { -.04, -.03, -.02, -.01 }
            },
            new [,]
            {
                { 0.01, 0.02 },
                { -.02, -.01 }
            },
            new [,]
            {
                { 0.01, 0.02, 0.03 },
                { -.06, -.05, 0.04 },
                { -.01, -.02, -.03 }
            }
        };
        return (weights, feedbacks);
    }

    [TestMethod]
    public void CreationTest()
    {
        var (weights, feedbacks) = GetTestNetworkStructure();
        var network = new RecurrentNetwork(weights, feedbacks);

        Assert.That.Value(network.LayersCount).IsEqual(weights.Length);
        Assert.That.Value(network.InputsCount).IsEqual(weights[0].GetLength(1));
        Assert.That.Value(network.OutputsCount).IsEqual(weights[^1].GetLength(0));
    }

    [TestMethod]
    public void SingleProcessTest()
    {
        var (weights, feedback) = GetTestNetworkStructure();
        var network = new RecurrentNetwork(weights, feedback);

        var input  = Enumerable.Range(1, network.InputsCount).Select(v => (double)v).ToArray();
        var output = new double[network.OutputsCount];

        network.Process(input, output);

        CollectionAssert.That.Collection(output).IsEqualTo(new[] { 0.2, 0.5, 0.8 }, 5.505e-006);
    }
}