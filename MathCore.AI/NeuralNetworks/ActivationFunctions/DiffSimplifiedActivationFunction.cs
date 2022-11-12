namespace MathCore.AI.NeuralNetworks.ActivationFunctions;

/// <summary>Активационная функция с упрощённой производной</summary>
public abstract class DiffSimplifiedActivationFunction : ActivationFunction
{
    /// <summary>Производная функции, выраженная через значение самой функции</summary>
    /// <param name="u">Значение функции по которому рассчитывается значение производной</param>
    /// <returns>Значение производной функции</returns>
    public abstract double DiffFunc(double u);
}