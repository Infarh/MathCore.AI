using System;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Результат обучения для одного обучающего образца</summary>
    public class TeachResult
    {
        /// <summary>Образец, на котором прводилось обучение</summary>
        [NotNull]
        public Example Example { get; }

        /// <summary>Входное воздействие</summary>
        [NotNull]
        public double[] Input => Example.Input;
        /// <summary>Отклик сети</summary>
        [NotNull]
        public double[] Output { get; }
        /// <summary>Желаемый результат</summary>
        [NotNull]
        public double[] ExpectedOutput => Example.ExpectedOutput;

        /// <summary>Ошибка отклика</summary>
        public double Error { get; }

        public TeachResult([NotNull] Example Example, [NotNull] double[] Output, double Error)
        {
            this.Example = Example ?? throw new ArgumentNullException(nameof(Example));
            this.Output = Output ?? throw new ArgumentNullException(nameof(Output));
            this.Error = Error;
        }

        public override string ToString() => $"err - {Error.RoundAdaptive(3)}";
    }
}