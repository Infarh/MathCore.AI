namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    public class ComplexExponent : ComplexActivationFunction
    {
        public static Complex Activation(Complex z) => Complex.Exp(z);

        public static Complex dActivation(Complex u) => u * (1 - u);

        public override Complex Value(Complex z) => Activation(z);

        public override Complex dValue(Complex z) => throw new System.NotImplementedException();
    }
}