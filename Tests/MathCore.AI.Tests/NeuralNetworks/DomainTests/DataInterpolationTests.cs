namespace MathCore.AI.Tests.NeuralNetworks.DomainTests;

[TestClass]
public class DataInterpolationTests
{
    private static string __DataFilePath = @"NeuralNetworks/DomainTests/Data/InterpolatorNDData.zip";

    private static FileInfo DataFile => new(__DataFilePath);

    [TestMethod]
    public void MultidimensionalInterpolationTest()
    {
        var file = DataFile;
        file.ThrowIfNotFound();
    }
}
