using System;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public interface INeuralNetwork
    {
        /// <summary>Число входов сети</summary>
        int InputsCount { get; }

        /// <summary>Число выходов сети</summary>
        int OutputsCount { get; }

        void Process([NotNull] double[] Input, [NotNull] double[] Output);
    }

    public interface IComplexNeuralNetwork
    {
        /// <summary>Число входов сети</summary>
        int InputsCount { get; }

        /// <summary>Число выходов сети</summary>
        int OutputsCount { get; }

        void Process([NotNull] Complex[] Input, [NotNull] Complex[] Output);
    }
}
