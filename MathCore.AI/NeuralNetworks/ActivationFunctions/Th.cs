using System;

namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    /// <summary>Гиперболический тангенс</summary>
    public class Th : DiffSiplifiedActivationFunction
    {
        public override double Value(double x)
        {
            var e = Math.Exp(x);
            var inv_e = 1 / e;
            return (e - inv_e) / (e + inv_e);
        }

        public override double DiffValue(double x) => DiffFunc(Value(x));

        public override double DiffFunc(double u) => 1 - u * u;

        public double Inverse(double u) => Math.Log((1 + u) / (1 - u)) / 2;
    }
}