namespace MathCore.AI.NeuralNetworks.ActivationFunctions;

/// <summary>Лямбда</summary>
public class Lambda : ActivationFunction
{
    private readonly Func<double, double> _Activation;
    private readonly Func<double, double> _DiffActivation;

    public Lambda(
        Func<double, double> Activation, 
        Func<double, double> dActivation)
    {
        _Activation     = Activation.NotNull();
        _DiffActivation = dActivation.NotNull();
    }

    public override double Value(double x) => _Activation(x);

    public override double DiffValue(double x) => _DiffActivation(x);
}