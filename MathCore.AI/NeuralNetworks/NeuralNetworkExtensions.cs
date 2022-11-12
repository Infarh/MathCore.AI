// ReSharper disable UnusedMember.Global
namespace MathCore.AI.NeuralNetworks;

public static class NeuralNetworkExtensions
{
    public static Epoch TeachRandom(
        this INetworkTeacher Teacher,
        Example[] Examples,
        Random? rnd = null) =>
        Teacher.NotNull().Teach(Examples.NotNull().AsRandomEnumerable(rnd));

    /// <summary>Выполнить обучение нейронной сети (одну эпоху) по заданному набору обучающих примеров</summary>
    /// <param name="Teacher">Обучаемая нейронная сеть</param>
    /// <param name="Examples">Набор обучающих примеров</param>
    /// <returns>Результат выполнения обучения за эпоху</returns>
    public static Epoch Teach(this INetworkTeacher Teacher, params Example[] Examples)
    {
        Examples.NotNull();
        if (Teacher.NotNull().Network.InputsCount == 0) throw new InvalidOperationException("Сеть не имеет входов");
        if (Teacher.Network.OutputsCount == 0) throw new InvalidOperationException("Сеть не имеет выходов");

        var inputs_count  = Teacher.Network.InputsCount;
        var outputs_count = Teacher.Network.OutputsCount;
        for (var i = 0; i < Examples.Length; i++)
        {
            var example = Examples[i];
            // ReSharper disable once ConditionIsAlwaysTrueOrFalse
            if (example is null)
                // ReSharper disable once HeuristicUnreachableCode
                throw new InvalidOperationException($"Обучающий пример с индексом {i} отсутствует");
            if (example.Input.Length != inputs_count)
                throw new InvalidOperationException($"Длина входного вектора примера №{i} ({example.Input.Length}) не равна количеству входов сети ({inputs_count})");
            if (example.ExpectedOutput.Length != outputs_count)
                throw new InvalidOperationException($"Длина вектора ожидаемого результата №{i} ({example.ExpectedOutput.Length}) не равна количеству выходов сети ({outputs_count})");
        }

        return Teach(Teacher, (IEnumerable<Example>)Examples);
    }

    /// <summary>Обучение нейронной сети на заданном перечислении примеров эпохи</summary>
    /// <param name="Teacher">Учитель нейронной сети, по средствам которого осуществляется процесс обучения</param>
    /// <param name="Examples">Перечисление примеров для одной эпохи обучения</param>
    /// <returns>Результат обучения нейронной сети за эпоху</returns>
    private static Epoch Teach(this INetworkTeacher Teacher, IEnumerable<Example> Examples)
    {
        var outputs_count = Teacher.Network.OutputsCount;
        var examples      = Examples as Example[] ?? Examples.ToArray();
        var results       = new TeachResult[examples.Length];
        var max_error     = 0d;
        var avg_error     = 0d;
        var output        = new double[outputs_count];
        for (var i = 0; i < examples.Length; i++)
        {
            var example                      = examples[i];
            var example_input                = example.Input;
            var example_expected_output      = example.ExpectedOutput;
            var error                        = Teacher.Teach(example_input, output, example_expected_output);
            if (error > max_error) max_error = error;
            avg_error += error;
            if (double.IsNaN(error)) throw new InvalidOperationException("Ошибка сети является \"не числом\" - возможно сеть нестабвльна.");
            if (double.IsInfinity(error)) throw new InvalidOperationException("Ошибка сети является \"бесконечностью\" - возможно сеть нестабильна.");
            results[i] = new TeachResult(example, output, error);
        }

        return new Epoch(results, max_error, avg_error / examples.Length);
    }

    /// <summary>Рассчитать отклик сети для входного воздействия</summary>
    /// <param name="Network">Нейронная сеть, осуществляющая обработку данных</param>
    /// <param name="Input">Вектор входного воздействия для сети</param>
    /// <returns>Вновь созданный вектор отклика сети</returns>
    public static double[] Process(this INeuralNetwork Network, double[] Input)
    {
        Input.NotNull();
        var result = new double[Network.NotNull().OutputsCount];
        Network.Process(Input, result);
        return result;
    }

    public static void Process(
        this INeuralNetwork Network,
        double[] Input,
        double[] Output,
        double[] ExpectedOutput,
        double[] Error)
    {
        Network.Process(Input, Output);

        for (var i = 0; i < Output.Length; i++)
        {
            var delta = ExpectedOutput[i] - Output[i];
            Error[i] = 0.5 * delta * delta;
        }
    }

    public static double[] Process(
        this INeuralNetwork Network,
        double[] Input,
        double[] Output,
        double[] ExpectedOutput
    )
    {
        var error = new double[Output.Length];
        Network.Process(Input, Output, ExpectedOutput, error);
        return error;
    }
}