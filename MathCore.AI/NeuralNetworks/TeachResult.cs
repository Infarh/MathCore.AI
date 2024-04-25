

// ReSharper disable UnusedMember.Global
// ReSharper disable ClassNeverInstantiated.Global
namespace MathCore.AI.NeuralNetworks;

/// <summary>Результат обучения для одного обучающего образца</summary>
public class TeachResult(Example Example, double[] Output, double Error)
{
    /// <summary>Образец, на котором проводилось обучение</summary>
    public Example Example { get; } = Example.NotNull();

    /// <summary>Входное воздействие</summary>
    public double[] Input => Example.Input;

    /// <summary>Отклик сети</summary>
    public double[] Output { get; } = Output.NotNull();

    /// <summary>Желаемый результат</summary>
    public double[] ExpectedOutput => Example.ExpectedOutput;

    /// <summary>Ошибка отклика</summary>
    public double Error { get; } = Error;

    public override string ToString() => $"err - {Error.RoundAdaptive(3)}";
}

/// <summary>Результат обучения для одного обучающего образца</summary>
public class TeachResult<TInput, TOutput>(Example<TInput, TOutput> Example, TOutput Output, double Error)
{
    /// <summary>Образец, на котором проводилось обучение</summary>
    public Example<TInput, TOutput> Example { get; } = Example.NotNull();

    /// <summary>Входное воздействие</summary>
    public TInput Input => Example.Input;

    /// <summary>Отклик контроллера</summary>
    public TOutput Output { get; } = Output;

    /// <summary>Желаемый результат</summary>
    public TOutput ExpectedOutput => Example.ExpectedOutput;

    /// <summary>Ошибка отклика</summary>
    public double Error { get; } = Error;

    public override string ToString() => $"err - {Error.RoundAdaptive(3)}";
}