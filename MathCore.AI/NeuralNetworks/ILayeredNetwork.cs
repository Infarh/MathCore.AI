using System.Collections.Generic;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public interface ILayeredNetwork : INeuralNetwork
    {
        /// <summary>Количество слоёв сети</summary>
        int LayersCount { get; }

        /// <summary>Массив выходов скрытых слоёв</summary>
        [NotNull] IReadOnlyList<double[]> HiddenOutputs { get; }
    }
}