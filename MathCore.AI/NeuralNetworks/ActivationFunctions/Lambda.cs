namespace MathCore.AI.NeuralNetworks.ActivationFunctions;

/// <summary>Лямбда</summary>
public class Lambda(
    Func<double, double> Activation,
    Func<double, double> dActivation) : ActivationFunction
{
    private readonly Func<double, double> _Activation = Activation.NotNull();
    private readonly Func<double, double> _DiffActivation = dActivation.NotNull();

    public override double Value(double x) => _Activation(x);

    public override double DiffValue(double x) => _DiffActivation(x);
}