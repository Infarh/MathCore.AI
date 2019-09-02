using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Комплексная нейронная сеть</summary>
    public interface IComplexNeuralNetwork
    {
        /// <summary>Число входов сети</summary>
        int InputsCount { get; }

        /// <summary>Число выходов сети</summary>
        int OutputsCount { get; }

        /// <summary>Обработать образ</summary>
        /// <param name="Input">Входной образ</param>
        /// <param name="Output">Отклик сети</param>
        void Process([NotNull] Complex[] Input, [NotNull] Complex[] Output);
    }
}