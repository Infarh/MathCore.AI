// ReSharper disable UnusedMember.Global
namespace MathCore.AI.NeuralNetworks;

/// <summary>Образец для обучения нейронной сети</summary>
public class Example
{
    private static double[] IntToDouble(IEnumerable<int> v) => v.NotNull().Select(i => (double)i).ToArray();

    /// <summary>Входное воздействие</summary>
    public double[] Input { get; }

    /// <summary>Ожидаемый результат</summary>
    public double[] ExpectedOutput { get; }

    public Example(double[] Input, int[] ExpectedOutput) : this(Input, IntToDouble(ExpectedOutput)) { }

    public Example(int[] Input, double[] ExpectedOutput) : this(IntToDouble(Input), ExpectedOutput) { }

    public Example(int[] Input, int[] ExpectedOutput) : this(IntToDouble(Input), IntToDouble(ExpectedOutput)) { }

    public Example(double[] Input, double[] ExpectedOutput)
    {
        this.Input          = Input.NotNull();
        this.ExpectedOutput = ExpectedOutput.NotNull();
    }

    #region Overrides of Object

    public override string ToString()
    {
        var inputs  = Input.Select(v => v.RoundAdaptive(3));
        var outputs = ExpectedOutput.Select(v => v.RoundAdaptive(3));
        return $"in:{string.Join(",", inputs)} out:{string.Join(",", outputs)}";
    }

    #endregion
}

public class Example<TInput, TOutput>
{
    public TInput Input { get; }

    public TOutput ExpectedOutput { get; }

    public Example(TInput Input, TOutput ExpectedOutput)
    {
        this.Input          = Input;
        this.ExpectedOutput = ExpectedOutput;
    }
}