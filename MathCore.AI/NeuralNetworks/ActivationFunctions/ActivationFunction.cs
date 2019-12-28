using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    /// <summary>Функция активации</summary>
    public abstract class ActivationFunction
    {
        /// <summary>Сигмоид (логистическая)</summary>
        [NotNull] public static Sigmoid Sigmoid => new Sigmoid();

        /// <summary>Линейная функция</summary>
        [NotNull] public static Linear Linear => new Linear();

        /// <summary>Гиперболический тангенс</summary>
        [NotNull] public static Th Th => new Th();

        /// <summary>Значение функции активации</summary>
        /// <param name="x">Взвешенная сумма входов нейронов</param>
        /// <returns>Значение выхода нейрона</returns>
        public abstract double Value(double x);

        /// <summary>Производная функции активации</summary>
        /// <param name="x">Значение на выходе нейрона</param>
        /// <returns>Значение производной функции активации</returns>
        public abstract double DiffValue(double x);
    }
}
