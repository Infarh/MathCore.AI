using MathCore.Annotations;
// ReSharper disable UnusedMember.Global

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Учитель нейронной сети</summary>
    public interface INetworkTeacher
    {
        /// <summary>Обучаемая сеть</summary>
        INeuralNetwork Network { get; }

        /// <summary>Обучение сети методом обратного распространения ошибки</summary>
        /// <param name="Input">Массив входа</param>
        /// <param name="Output">Массив выхода</param>
        /// <param name="Expected">Ожидаемое значение на выходе сети</param>
        /// <returns>Среднеквадратичная ошибка обучения</returns>
        double Teach([NotNull] double[] Input, [NotNull] double[] Output, [NotNull] double[] Expected);
    }
}