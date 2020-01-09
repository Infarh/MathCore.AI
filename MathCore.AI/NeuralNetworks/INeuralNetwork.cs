using System;

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Нейронная сеть</summary>
    public interface INeuralNetwork
    {
        /// <summary>Число входов сети</summary>
        int InputsCount { get; }

        /// <summary>Число выходов сети</summary>
        int OutputsCount { get; }

        /// <summary>Обработать образ</summary>
        /// <param name="Input">Входной образ</param>
        /// <param name="Output">Отклик сети</param>
        void Process(Span<double> Input, Span<double> Output);
    }
}
