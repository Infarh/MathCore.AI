using System;
using MathCore.Annotations;

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

    /// <summary>Учитель нейронного процессора</summary>
    /// <typeparam name="TInput">Тип входного значения</typeparam>
    /// <typeparam name="TOutput">Тип выходного значения</typeparam>
    public interface INeuralProcessorTeacher<in TInput, TOutput>
    {
        /// <summary>Обучаемая сеть</summary>
        [NotNull] INeuralNetwork Network { get; }

        /// <summary>Обучить процессор</summary>
        /// <param name="Input">Входное значение</param>
        /// <param name="Expected">Ожидаемое значение</param>
        /// <returns>Ошибка обучения</returns>
        double Teach(TInput Input, TOutput Expected);

        /// <summary>Обучить процессор</summary>
        /// <param name="Input">Входное значение</param>
        /// <param name="Expected">Ожидаемое значение</param>
        /// <param name="Output">Результат обработки</param>
        /// <returns>Ошибка обучения</returns>
        double Teach(TInput Input, TOutput Expected, out TOutput Output);
    }
}