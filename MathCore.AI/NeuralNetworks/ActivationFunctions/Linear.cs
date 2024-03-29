﻿namespace MathCore.AI.NeuralNetworks.ActivationFunctions;

/// <summary>Линейная</summary>
public class Linear : ActivationFunction
{
    private readonly double _K = 1;

    /// <summary>Смещение</summary>
    private readonly double _B;

    /// <summary>Тангенс угла наклона</summary>
    public double K => _K;

    /// <summary>Смещение</summary>
    public double B => _B;

    public Linear() { }

    public Linear(double K, double B = 0)
    {
        _K = K;
        _B = B;
    }

    public override double Value(double x) => _K * x + _B;

    public override double DiffValue(double x) => _K;
}