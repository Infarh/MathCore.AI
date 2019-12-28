using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    public abstract class ComplexActivationFunction
    {
        [NotNull] public ComplexActivationFunction Exponent => new ComplexExponenrt();

        /// <summary>Значение функции активации</summary>
        /// <param name="x">Взвешенная сумма входов нейронов</param>
        /// <returns>Значение выхода нейрона</returns>
        public abstract Complex Value(Complex x);

        /// <summary>Производная функции активации</summary>
        /// <param name="u">Значение на выходе нейрона</param>
        /// <returns>Значение производной функции активации</returns>
        public abstract Complex dValue(Complex z); 
    }
}