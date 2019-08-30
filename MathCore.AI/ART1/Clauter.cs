using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;
using MathCore.Annotations;

namespace MathCore.AI.ART1
{
    [DebuggerDisplay("{ToDebuggerString()}")]
    public sealed class Clauter<T> : IEnumerable<T>
    {
        [NotNull] private readonly double[] _FeatureVector;
        [NotNull] private readonly Func<T, double[]> _Classificator;
        private readonly string[] _FeatureNames;

        [NotNull] private readonly HashSet<T> _Items = new HashSet<T>(new LambdaEqualityComparer<T>((v1, v2) => ReferenceEquals(v1, v2), v => ((object)v).GetHashCode()));

        public int ItemsCount => _Items.Count;

        internal Clauter([NotNull] double[] FeatureVector, T Item, [NotNull] Func<T, double[]> Classificator, [NotNull] string[] FeatureNames)
        {
            _FeatureVector = FeatureVector;
            _Classificator = Classificator;
            _FeatureNames = FeatureNames;
            _Items.Add(Item);
        }

        internal void Add([NotNull] double[] Prototype, T Item)
        {
            if (_Items.Contains(Item)) return;
            _Items.Add(Item);
            for (var i = 0; i < Prototype.Length; i++)
                _FeatureVector[i] *= Prototype[i];
        }

        internal void Clear()
        {
            _Items.Clear();
            _FeatureVector.Initialize(0);
        }

        public bool Contains(T Item) => _Items.Contains(Item);

        internal bool Remove(T Item)
        {
            var removed = _Items.Remove(Item);
            if (!removed) return false;

            var feature_vector = _FeatureVector;
            var i = 0;
            foreach (var item in _Items)
            {
                var item_features = _Classificator(item);
                if (i == 0) item_features.CopyTo(feature_vector, 0);
                else
                    for (var j = 0; j < feature_vector.Length; j++)
                        feature_vector[j] *= item_features[j];
                i++;
            }

            return true;
        }

        internal bool SimilarityAndCareTest([NotNull] double[] Prototype, double Betta, double Rho)
        {
            var correlation = 0d;
            var prototype_conditionaly = 0d;
            var features_conditionaly = 0d;
            var feature_vector = _FeatureVector;
            var length = feature_vector.Length;

            for (var i = 0; i < length; i++)
            {
                var p = Prototype[i];
                var e = feature_vector[i];
                correlation += p * e;
                prototype_conditionaly += p;
                features_conditionaly += e;
            }

            return correlation / (Betta + prototype_conditionaly) > features_conditionaly / (Betta + length)
                   && correlation / features_conditionaly < Rho;
        }

        #region Overrides of Object

        public override string ToString()
        {
            var result = new StringBuilder();
            for (var i = 0; i < _FeatureVector.Length; i++)
                if (_FeatureVector[i] != 0)
                    if (_FeatureVector[i] == 1d)
                        result.AppendFormat("{0}, ", _FeatureNames[i]);
                    else
                        result.AppendFormat("{0}({1}), ", _FeatureNames[i], _FeatureVector[i].RoundAdaptive(2));

            if (result.Length > 0) result.Length -= 2;
            return $"[ {result} ]";
        }

        private string ToDebuggerString() => $"{this}:{string.Join(", ", _Items)}";

        #endregion

        #region Implementation of IEnumerable

        public IEnumerator<T> GetEnumerator() => _Items.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)_Items).GetEnumerator();

        #endregion
    }
}