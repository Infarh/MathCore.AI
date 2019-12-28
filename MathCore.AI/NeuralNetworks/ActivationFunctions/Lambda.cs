using System;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    /// <summary>Лямбда</summary>
    public class Lambda : ActivationFunction
    {
        [NotNull] private readonly Func<double, double> _Activation;
        [NotNull] private readonly Func<double, double> _DiffActivation;

        public Lambda(
            [NotNull] Func<double, double> Activation, 
            [NotNull] Func<double, double> dActivation)
        {
            _Activation = Activation ?? throw new ArgumentNullException(nameof(Activation));
            _DiffActivation = dActivation ?? throw new ArgumentNullException(nameof(dActivation));
        }

        public override double Value(double x) => _Activation(x);

        public override double DiffValue(double x) => _DiffActivation(x);
    }
}