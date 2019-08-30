using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathCore.Annotations;
// ReSharper disable UnusedMethodReturnValue.Global

namespace MathCore.AI.ART1
{
    public sealed class ClassificationCriterias<T> : IEnumerable<ClassificationCriteria<T>>
    {
        [NotNull] private readonly List<ClassificationCriteria<T>> _Criterias = new List<ClassificationCriteria<T>>();

        public int Count => _Criterias.Count;

        public ClassificationCriteria<T> Add([CanBeNull] string Name, [NotNull] Func<T, double> Criteria)
        {
            var criteria = new ClassificationCriteria<T>(Name, Criteria ?? throw new ArgumentNullException(nameof(Criteria)));
            _Criterias.Add(criteria);
            return criteria;
        }

        [NotNull]
        public ClassificationCriteria<T> Add([NotNull] Func<T, double> Criteria)
        {
            var criteria = new ClassificationCriteria<T>(Criteria ?? throw new ArgumentNullException(nameof(Criteria)));
            _Criterias.Add(criteria);
            return criteria;
        }

        [NotNull]
        public double[] GetFeaturesVector(T Item)
        {
            var result = new double[Count];
            for (var i = 0; i < result.Length; i++)
                result[i] = _Criterias[i].GetFeatureValue(Item);
            return result;
        }

        [NotNull, ItemCanBeNull] public string[] GetFeatureNames() => _Criterias.Select(f => f.Name).ToArray();

        #region Implementation of IEnumerable

        public IEnumerator<ClassificationCriteria<T>> GetEnumerator() => _Criterias.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)_Criterias).GetEnumerator();

        #endregion
    }
}