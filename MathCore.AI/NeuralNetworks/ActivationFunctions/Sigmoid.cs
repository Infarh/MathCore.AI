using System;

namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    /// <summary>Логистическая функция (Сигмоид)</summary>
    public class Sigmoid : DiffSiplifiedActivationFunction
    {
        public static double Activation(double x) => 1 / (1 + Math.Exp(-x));

        public static double DiffActivation(double u) => u * (1 - u);

        public override double Value(double x) => Activation(x);

        public override double DiffValue(double x) => DiffFunc(Value(x));

        public override double DiffFunc(double u) => DiffActivation(u);

        public double Inverse(double u) => -Math.Log(1 / u - 1);
    }
}
