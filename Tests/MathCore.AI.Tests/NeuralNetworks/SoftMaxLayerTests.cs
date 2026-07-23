namespace MathCore.AI.Tests.NeuralNetworks;

using MathCore.AI.NeuralNetworks;

[TestClass]
public class SoftMaxLayerTests
{
    /* --------------------------------------------------------------------------------------------- */

    [TestMethod]
    public void SoftMax_Basic_SumToOne()
    {
        // Простейшая проверка: сумма выходов должна быть равна 1
        var softmax = new SoftMaxLayer(3);
        double[] input = [1.0, 2.0, 3.0];
        var output = new double[3];

        softmax.Process(input, output);

        var sum = output.Sum();
        Assert.AreEqual(1.0, sum, 1e-12, "Сумма вероятностей SoftMax должна быть равна 1");
    }

    [TestMethod]
    public void SoftMax_AllValuesBetweenZeroAndOne()
    {
        // Каждый выход должен быть в диапазоне (0, 1)
        var softmax = new SoftMaxLayer(4);
        double[] input = [-1.0, 0.0, 2.0, 5.0];
        var output = new double[4];

        softmax.Process(input, output);

        for (var i = 0; i < output.Length; i++)
        {
            Assert.IsTrue(output[i] > 0, $"Выход [{i}] должен быть больше 0");
            Assert.IsTrue(output[i] < 1, $"Выход [{i}] должен быть меньше 1");
        }
    }

    [TestMethod]
    public void SoftMax_LargestInput_GetsLargestProbability()
    {
        // Наибольший вход должен получить наибольшую вероятность
        var softmax = new SoftMaxLayer(3);
        double[] input = [0.5, 2.0, 1.0];
        var output = new double[3];

        softmax.Process(input, output);

        Assert.IsTrue(output[1] > output[0], "Вероятность для x=2.0 должна быть больше чем для x=0.5");
        Assert.IsTrue(output[1] > output[2], "Вероятность для x=2.0 должна быть больше чем для x=1.0");
    }

    [TestMethod]
    public void SoftMax_EqualInputs_EqualProbabilities()
    {
        // Одинаковые входы дают одинаковые вероятности
        var softmax = new SoftMaxLayer(4);
        double[] input = [1.0, 1.0, 1.0, 1.0];
        var output = new double[4];

        softmax.Process(input, output);

        for (var i = 0; i < output.Length; i++)
            Assert.AreEqual(0.25, output[i], 1e-12, $"Вероятность [{i}] должна быть 0.25");
    }

    [TestMethod]
    public void SoftMax_NegativeInputs_ValidProbabilities()
    {
        // Отрицательные входы всё равно дают корректное распределение
        var softmax = new SoftMaxLayer(3);
        double[] input = [-10.0, -20.0, -30.0];
        var output = new double[3];

        softmax.Process(input, output);

        var sum = output.Sum();
        Assert.AreEqual(1.0, sum, 1e-12);
        Assert.IsTrue(output[0] > output[1], "Больший (менее отрицательный) вход должен давать большую вероятность");
        Assert.IsTrue(output[1] > output[2], "Больший (менее отрицательный) вход должен давать большую вероятность");
    }

    [TestMethod]
    public void SoftMax_OrderInvariant()
    {
        // Порядок элементов сохраняется: SoftMax — монотонное преобразование
        var softmax = new SoftMaxLayer(5);
        double[] input = [3.0, 0.5, -2.0, 4.0, 1.0];
        var output = new double[5];

        softmax.Process(input, output);

        // Индекс максимального элемента должен совпадать
        var max_input_index = 0;
        var max_output_index = 0;
        for (var i = 1; i < input.Length; i++)
        {
            if (input[i] > input[max_input_index]) max_input_index = i;
            if (output[i] > output[max_output_index]) max_output_index = i;
        }

        Assert.AreEqual(max_input_index, max_output_index,
            "Индекс максимального элемента должен сохраняться после SoftMax");
    }

    /* --------------------------------------------------------------------------------------------- */

    [TestMethod]
    public void SoftMaxInCascade_AfterMultilayerPerceptron()
    {
        // Создаём MLP (4 входа → 3 выхода) и каскадно присоединяем SoftMax
        var mlp = new MultilayerPerceptron(
            (double[,])new double[,]
            {
                { 0.1, 0.2, 0.3, 0.4 },
                { 0.5, 0.6, 0.7, 0.8 },
                { 0.9, 1.0, 1.1, 1.2 }
            });

        var softmax = new SoftMaxLayer(3);

        var cascade = CascadeBuilder
            .From(mlp)
            .Then(softmax)
            .Build();

        Assert.AreEqual(4, cascade.InputsCount);
        Assert.AreEqual(3, cascade.OutputsCount);

        double[] input = [0.5, 1.0, 1.5, 2.0];
        var output = new double[3];
        cascade.Process(input, output);

        // Сумма вероятностей должна быть 1
        var sum = output.Sum();
        Assert.AreEqual(1.0, sum, 1e-10, "Сумма вероятностей после каскада MLP+SoftMax должна быть 1");

        // Все значения — корректные вероятности
        for (var i = 0; i < output.Length; i++)
        {
            Assert.IsTrue(output[i] >= 0, $"Выход [{i}] не может быть отрицательным");
            Assert.IsTrue(output[i] <= 1, $"Выход [{i}] не может быть больше 1");
        }
    }

    [TestMethod]
    public void SoftMaxInCascade_WithTee()
    {
        // Два MLP работают параллельно от одного входа, каждый со своим SoftMax
        var mlp1 = new MultilayerPerceptron(
            (double[,])new double[,] { { 0.1, 0.5 }, { 0.3, 0.7 } }); // 2→2
        var mlp2 = new MultilayerPerceptron(
            (double[,])new double[,] { { 0.2, 0.6 }, { 0.4, 0.8 }, { 0.9, 0.1 } }); // 2→3

        var softmax1 = new SoftMaxLayer(2);
        var softmax2 = new SoftMaxLayer(3);

        // Каждая ветка: MLP → SoftMax
        var branch1 = CascadeBuilder.From(mlp1).Then(softmax1).Build();
        var branch2 = CascadeBuilder.From(mlp2).Then(softmax2).Build();

        // Общий cascade с Tee: вход → [branch1, branch2] → конкатенация 5 выходов
        var cascade = CascadeBuilder
            .From(new CascadeBuilder.ConcatSources(branch1, branch2))
            .Build();

        Assert.AreEqual(2, cascade.InputsCount);
        Assert.AreEqual(5, cascade.OutputsCount);

        double[] input = [0.3, 0.7];
        var output = new double[5];
        cascade.Process(input, output);

        // Первые два выхода — вероятности от branch1 (должны суммироваться в 1)
        var sum1 = output[0] + output[1];
        Assert.AreEqual(1.0, sum1, 1e-10, "SoftMax первой ветки должен давать сумму 1");

        // Последние три — вероятности от branch2 (сумма 1)
        var sum2 = output[2] + output[3] + output[4];
        Assert.AreEqual(1.0, sum2, 1e-10, "SoftMax второй ветки должен давать сумму 1");
    }

    [TestMethod]
    public void SoftMax_CascadeAlreadyBuilt_CascadeBuilder()
    {
        // Каскад MLP → SoftMax через CascadeBuilder
        var mlp = new MultilayerPerceptron(
            (double[,])new double[,] { { 0.2, 0.8 }, { 0.5, 0.3 }, { 0.7, 0.1 } }); // 2→3
        var softmax = new SoftMaxLayer(3);

        var cascade = CascadeBuilder
            .From(mlp)
            .Then(softmax)
            .Build();

        // Проверяем через Process массивы
        double[] input = [2.0, 1.0];
        var output = new double[3];
        cascade.Process(input, output);

        // Проверяем свойства распределения
        var sum = output.Sum();
        Assert.AreEqual(1.0, sum, 1e-10);
    }

    [TestMethod]
    public void SoftMax_SingleElement()
    {
        // SoftMax от одного элемента всегда даёт 1.0
        var softmax = new SoftMaxLayer(1);

        double[] input = [42.0];
        var output = new double[1];
        softmax.Process(input, output);

        Assert.AreEqual(1.0, output[0], 1e-15, "SoftMax от одного элемента должен быть 1.0");
    }

    /* --------------------------------------------------------------------------------------------- */
}
