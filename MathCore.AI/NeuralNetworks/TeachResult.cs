

// ReSharper disable UnusedMember.Global
// ReSharper disable ClassNeverInstantiated.Global
namespace MathCore.AI.NeuralNetworks;

/// <summary>Результат обучения для одного обучающего образца</summary>
public class TeachResult
{
    /// <summary>Образец, на котором проводилось обучение</summary>
    public Example Example { get; }

    /// <summary>Входное воздействие</summary>
    public double[] Input => Example.Input;

    /// <summary>Отклик сети</summary>
    public double[] Output { get; }

    /// <summary>Желаемый результат</summary>
    public double[] ExpectedOutput => Example.ExpectedOutput;

    /// <summary>Ошибка отклика</summary>
    public double Error { get; }

    public TeachResult(Example Example, double[] Output, double Error)
    {
        this.Example = Example.NotNull();
        this.Output  = Output.NotNull();
        this.Error   = Error;
    }

    public override string ToString() => $"err - {Error.RoundAdaptive(3)}";
}

/// <summary>Результат обучения для одного обучающего образца</summary>
public class TeachResult<TInput, TOutput>
{
    /// <summary>Образец, на котором проводилось обучение</summary>
    public Example<TInput, TOutput> Example { get; }

    /// <summary>Входное воздействие</summary>
    public TInput Input => Example.Input;

    /// <summary>Отклик контроллера</summary>
    public TOutput Output { get; }

    /// <summary>Желаемый результат</summary>
    public TOutput ExpectedOutput => Example.ExpectedOutput;

    /// <summary>Ошибка отклика</summary>
    public double Error { get; }

    public TeachResult(Example<TInput, TOutput> Example, TOutput Output, double Error)
    {
        this.Example = Example.NotNull();
        this.Output  = Output;
        this.Error   = Error;
    }

    public override string ToString() => $"err - {Error.RoundAdaptive(3)}";
}