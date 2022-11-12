namespace MathCore.AI.NeuralNetworks.ActivationFunctions;

/// <summary>Отсечка</summary>
public class ReLU : ActivationFunction
{
    private readonly double _K = 1;

    /// <summary>Смещение</summary>
    private readonly double _B;

    /// <summary>Тангенс угла наклона</summary>
    public double K => _K;

    /// <summary>Смещение</summary>
    public double B => _B;

    public ReLU() { }

    public ReLU(double K, double B = 0)
    {
        _K = K;
        _B = B;
    }

    public override double Value(double x) => x > _B ? _K * x : 0;

    public override double DiffValue(double x) => x > _B ? _K : 0;
}