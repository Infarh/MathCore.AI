namespace MathCore.AI.Tests.NeuralNetworks;

using MathCore.AI.NeuralNetworks;

[TestClass]
public class CascadeNetworkTests
{
    private static MultilayerPerceptron CreateNet(int InputsCount, int OutputsCount, double weight = 0.5)
    {
        var layers = new double[][,]
        {
            new double[OutputsCount, InputsCount]
        };
        for (var i = 0; i < OutputsCount; i++)
            for (var j = 0; j < InputsCount; j++)
                layers[0][i, j] = weight;

        return new MultilayerPerceptron(layers);
    }

    [TestMethod]
    public void CascadeSequential_TwoNetworks()
    {
        // net1: 2→3, net2: 3→1
        var net1 = CreateNet(2, 3, 0.5);
        var net2 = CreateNet(3, 1, 0.7);

        var cascade = CascadeBuilder
            .From(net1)
            .Then(net2)
            .Build();

        Assert.AreEqual(2, cascade.InputsCount);
        Assert.AreEqual(1, cascade.OutputsCount);
        Assert.AreEqual(2, cascade.StepsCount);

        double[] input = [1.0, 2.0];
        var output = new double[1];
        cascade.Process(input, output);

        Assert.IsFalse(double.IsNaN(output[0]));
        Assert.IsFalse(double.IsInfinity(output[0]));
    }

    [TestMethod]
    public void CascadeSequential_ThreeNetworks()
    {
        // net1: 2→3, net2: 3→4, net3: 4→2
        var net1 = CreateNet(2, 3, 0.5);
        var net2 = CreateNet(3, 4, 0.5);
        var net3 = CreateNet(4, 2, 0.5);

        var cascade = CascadeBuilder
            .From(net1)
            .Then(net2)
            .Then(net3)
            .Build();

        Assert.AreEqual(2, cascade.InputsCount);
        Assert.AreEqual(2, cascade.OutputsCount);
        Assert.AreEqual(3, cascade.StepsCount);

        double[] input = [0.5, -0.5];
        var output = new double[2];
        cascade.Process(input, output);

        Assert.IsFalse(double.IsNaN(output[0]));
        Assert.IsFalse(double.IsNaN(output[1]));
    }

    [TestMethod]
    public void CascadeSingleNetwork_Identity()
    {
        var net = CreateNet(2, 3, 0.5);

        var cascade = CascadeBuilder
            .From(net)
            .Build();

        Assert.AreEqual(net.InputsCount, cascade.InputsCount);
        Assert.AreEqual(net.OutputsCount, cascade.OutputsCount);
        Assert.AreEqual(1, cascade.StepsCount);

        double[] input = [1.0, 2.0];
        var expected = new double[3];
        var actual = new double[3];

        net.Process(input, expected);
        cascade.Process(input, actual);

        CollectionAssert.AreEqual(expected, actual);
    }

    [TestMethod]
    public void CascadeWithTee_TwoBranches()
    {
        // net1: 2→4, tee: net2 (4→3) + net3 (4→5) параллельно, конкатенация = 8 выходов
        var net1 = CreateNet(2, 4, 0.5);
        var net2 = CreateNet(4, 3, 0.6);
        var net3 = CreateNet(4, 5, 0.7);

        var cascade = CascadeBuilder
            .From(net1)
            .Tee(net2, net3)
            .Build();

        Assert.AreEqual(2, cascade.InputsCount);
        Assert.AreEqual(8, cascade.OutputsCount); // 3 + 5
        Assert.AreEqual(2, cascade.StepsCount);

        double[] input = [1.0, 2.0];
        var output = new double[8];
        cascade.Process(input, output);

        Assert.IsFalse(double.IsNaN(output[0]));
        Assert.IsFalse(double.IsNaN(output[7]));
    }

    [TestMethod]
    public void CascadeWithJoin_ConcatSources()
    {
        // netA: 3→2, netB: 3→4 — параллельно от одного входа
        // затем конкатенация 2+4=6 → netC: 6→1
        var netA = CreateNet(3, 2, 0.5);
        var netB = CreateNet(3, 4, 0.5);
        var netC = CreateNet(6, 1, 0.5);

        var cascade = CascadeBuilder
            .Join(netA, netB)
            .Then(netC)
            .Build();

        Assert.AreEqual(3, cascade.InputsCount);
        Assert.AreEqual(1, cascade.OutputsCount);
        Assert.AreEqual(2, cascade.StepsCount);

        double[] input = [0.1, 0.2, 0.3];
        var output = new double[1];
        cascade.Process(input, output);

        Assert.IsFalse(double.IsNaN(output[0]));
    }

    [TestMethod]
    public void Cascade_NestedCascade()
    {
        // Внутренний каскад: net1(2→3) → net2(3→4)
        // Внешний каскад: inner(2→4) → net3(4→1)
        var net1 = CreateNet(2, 3, 0.5);
        var net2 = CreateNet(3, 4, 0.5);
        var net3 = CreateNet(4, 1, 0.5);

        var inner = CascadeBuilder
            .From(net1)
            .Then(net2)
            .Build();

        var outer = CascadeBuilder
            .From(inner)
            .Then(net3)
            .Build();

        Assert.AreEqual(2, outer.InputsCount);
        Assert.AreEqual(1, outer.OutputsCount);

        double[] input = [0.5, -0.3];
        var output = new double[1];
        outer.Process(input, output);

        Assert.IsFalse(double.IsNaN(output[0]));
    }

    [TestMethod]
    public void Cascade_SizeMismatch_Throws()
    {
        // Выход 2, вход следующей сети 3 — несоответствие
        var net1 = CreateNet(2, 2, 0.5);
        var net2 = CreateNet(3, 1, 0.5);

        Assert.Throws<InvalidOperationException>(() =>
            CascadeBuilder
                .From(net1)
                .Then(net2)
                .Build());
    }

    [TestMethod]
    public void Cascade_SlotNotNetwork()
    {
        var net1 = CreateNet(2, 3, 0.5);
        var net2 = CreateNet(2, 4, 0.5);

        var join = new CascadeBuilder.ConcatSources(net1, net2);
        var cascade = CascadeBuilder.From(join).Build();

        Assert.AreEqual(2, cascade.InputsCount);
        Assert.AreEqual(7, cascade.OutputsCount);

        double[] input = [1.0, 2.0];
        var output = new double[7];
        cascade.Process(input, output);

        Assert.IsFalse(double.IsNaN(output[0]));
        Assert.IsFalse(double.IsNaN(output[6]));
    }
}
