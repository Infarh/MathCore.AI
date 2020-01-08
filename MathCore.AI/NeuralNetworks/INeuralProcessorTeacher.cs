using MathCore.Annotations;
// ReSharper disable UnusedMember.Global

namespace MathCore.AI.NeuralNetworks
{
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