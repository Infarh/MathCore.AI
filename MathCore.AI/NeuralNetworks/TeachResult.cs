using System;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Результат обучения для одного обучающего образца</summary>
    public class TeachResult
    {
        /// <summary>Образец, на котором проводилось обучение</summary>
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

        [NotNull] public override string ToString() => $"err - {Error.RoundAdaptive(3)}";
    }

    /// <summary>Результат обучения для одного обучающего образца</summary>
    public class TeachResult<TInput, TOutput>
    {
        /// <summary>Образец, на котором проводилось обучение</summary>
        [NotNull] public Example<TInput, TOutput> Example { get; }

        /// <summary>Входное воздействие</summary>
        public TInput Input => Example.Input;

        /// <summary>Отклик контроллера</summary>
        public TOutput Output { get; }

        /// <summary>Желаемый результат</summary>
        public TOutput ExpectedOutput => Example.ExpectedOutput;

        /// <summary>Ошибка отклика</summary>
        public double Error { get; }

        public TeachResult([NotNull] Example<TInput, TOutput> Example, TOutput Output, double Error)
        {
            this.Example = Example;
            this.Output = Output;
            this.Error = Error;
        }

        [NotNull] public override string ToString() => $"err - {Error.RoundAdaptive(3)}";
    }
}