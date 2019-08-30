using System;
using MathCore.Annotations;

namespace MathCore.AI.ART1
{
    public sealed class ClassificationCriteria<T>
    {
        [NotNull] private readonly Func<T, double> _Criteria;
        [CanBeNull] public string Name { get; set; }

        public ClassificationCriteria([NotNull] Func<T, double> Criteria) => _Criteria = Criteria ?? throw new ArgumentNullException(nameof(Criteria));

        public ClassificationCriteria([CanBeNull] string Name, [NotNull] Func<T, double> Criteria) : this(Criteria) => this.Name = Name;

        public double GetFeatureValue(T Item) => _Criteria(Item);
    }
}