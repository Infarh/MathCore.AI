namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    /// <summary>Активационная функция с упрощённой производной</summary>
    public abstract class DiffSiplifiedActivationFunction : ActivationFunction
    {
        /// <summary>Производная функции, выраженая через значение самой функции</summary>
        /// <param name="u">Значение функции по которому расчитывается значение производной</param>
        /// <returns>Значение производной функции</returns>
        public abstract double DiffFunc(double u);
    }
}