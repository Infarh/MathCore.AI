using System;
using System.Collections.Generic;
using System.Linq;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Эпоха обучения</summary>
    public class Epoch
    {
        /// <summary>Результаты обучения за эпоху</summary>
        [NotNull]
        public IReadOnlyCollection<TeachResult> Results { get; }

        /// <summary>Максимальное значение ошибки за эпоху</summary>
        public double ErrorMax { get; }
        /// <summary>Среднее значение ошибки за эпоху</summary>
        public double ErrorAverage { get; }

        internal Epoch([NotNull] IReadOnlyCollection<TeachResult> Results, double ErrorMax, double ErrorAverage)
        {
            this.Results = Results;
            this.ErrorMax = ErrorMax;
            this.ErrorAverage = ErrorAverage;
        }

        public override string ToString() => $"err - m:{ErrorMax.RoundAdaptive(3)}({ErrorAverage.RoundAdaptive(3)})";
    }
}