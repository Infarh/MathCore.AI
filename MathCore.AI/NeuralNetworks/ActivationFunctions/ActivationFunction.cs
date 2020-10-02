using MathCore.Annotations;
// ReSharper disable UnusedMember.Global

namespace MathCore.AI.NeuralNetworks.ActivationFunctions
{
    /// <summary>Функция активации</summary>
    public abstract class ActivationFunction
    {
        /// <summary>Сигмоид (логистическая)</summary>
        [NotNull] public static Sigmoid Sigmoid => new Sigmoid();

        /// <summary>Линейная функция</summary>
        [NotNull] public static Linear Linear => new Linear();

        /// <summary>Отсечка</summary>
        [NotNull] public static ReLU ReLU => new ReLU();

        /// <summary>Линейная функция с параметрами</summary>
        /// <param name="K">Производная</param>
        /// <param name="B">Смещение</param>
        [NotNull] public static Linear GetLinear(double K, double B = 0) => new Linear(K, B);

        /// <summary>Функция отсечки (линейная ломанная x>0?k*x:0)</summary>
        /// <param name="K">Производная</param>
        /// <param name="B">Смещение</param>
        [NotNull] public static ReLU GetReLU(double K, double B = 0) => new ReLU(K, B);

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
