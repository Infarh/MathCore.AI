using System;
using System.Collections.Generic;
using System.Linq;
using MathCore.Annotations;
// ReSharper disable UnusedMember.Global

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Образец для обучения нейронной сети</summary>
    public class Example
    {
        [NotNull] private static double[] IntToDouble([NotNull] IEnumerable<int> v) => (v ?? throw new ArgumentNullException(nameof(v))).Select(i => (double)i).ToArray();

        /// <summary>Входное воздействие</summary>
        [NotNull] public double[] Input { get; }

        /// <summary>Ожидаемый результат</summary>
        [NotNull] public double[] ExpectedOutput { get; }

        public Example([NotNull] double[] Input, [NotNull] int[] ExpectedOutput) : this(Input, IntToDouble(ExpectedOutput)) { }
        public Example([NotNull] int[] Input, [NotNull] double[] ExpectedOutput) : this(IntToDouble(Input), ExpectedOutput) { }
        public Example([NotNull] int[] Input, [NotNull] int[] ExpectedOutput) : this(IntToDouble(Input), IntToDouble(ExpectedOutput)) { }
        public Example([NotNull] double[] Input, [NotNull] double[] ExpectedOutput)
        {
            this.Input = Input ?? throw new ArgumentNullException(nameof(Input));
            this.ExpectedOutput = ExpectedOutput ?? throw new ArgumentNullException(nameof(ExpectedOutput));
        }

        #region Overrides of Object

        public override string ToString()
        {
            var inputs = Input.Select(v => v.RoundAdaptive(3));
            var outputs = ExpectedOutput.Select(v => v.RoundAdaptive(3));
            return $"in:{string.Join(",", inputs)} out:{string.Join(",", outputs)}";
        }

        #endregion
    }

    public class Example<TInput, TOutput>
    {
        public TInput Input { get; }

        public TOutput ExpectedOutput { get; }

        public Example(TInput Input, TOutput ExpectedOutput)
        {
            this.Input = Input;
            this.ExpectedOutput = ExpectedOutput;
        }
    }
}