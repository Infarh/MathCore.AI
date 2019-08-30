using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathCore.Annotations;
// ReSharper disable ClassNeverInstantiated.Global

namespace MathCore.AI.ART1
{
    public class Classificator<T> : IEnumerable<Clauter<T>>
        where T : class
    {
        /// <summary>Параметр внимательности (0;1]</summary>
        /// <remarks>Определяет размер кластера</remarks>
        private double _Vigilance = 0.9;

        /// <summary>Бетта-параметр (разрушения связей) - чем больше, тем больше кластеров будет образовано</summary>
        private double _Beta = 1;

        private int _AddIterationsCount = 50;

        [NotNull] private readonly List<Clauter<T>> _Clusters = new List<Clauter<T>>();

        /// <summary>Параметр внимательности (0;1]</summary>
        /// <remarks>Определяет размер кластера</remarks>
        public double Vigilance
        {
            get => _Vigilance;
            set
            {
                if (value <= 0 || value > 1)
                    throw new ArgumentOutOfRangeException(
                        nameof(value), 
                        value, 
                        "Vigilance должно быть в пределах от 0 до 1 (включительно)");
                _Vigilance = value;
            }
        }

        /// <summary>Бетта-параметр (разрушения связей) - чем больше, тем больше кластеров будет образовано</summary>
        public double Beta
        {
            get => _Beta;
            set
            {
                if (value <= 0 || value > 1)
                    throw new ArgumentOutOfRangeException(
                        nameof(value), 
                        value, 
                        "Betta должно быть в пределах от 0 до 1 (включительно)");
                _Beta = value;
            }
        }

        /// <summary>Количество иметарций перераспределения образцов по кластерам для операции добавления</summary>
        public int AddOperationIterationCount
        {
            get => _AddIterationsCount;
            set => _AddIterationsCount = value;
        }

        [NotNull] public ClassificationCriterias<T> Criterias { get; } = new ClassificationCriterias<T>();

        [NotNull, ItemNotNull] public IReadOnlyCollection<Clauter<T>> Clusters => _Clusters;

        [NotNull]
        public Clauter<T> Add([NotNull] T Item)
        {
            if (Item is null) throw new ArgumentNullException(nameof(Item));
               

            if (_Clusters.Count == 0)
            {
                var first_cluster = new Clauter<T>(
                    Criterias.GetFeaturesVector(Item), 
                    Item, 
                    Criterias.GetFeaturesVector, 
                    Criterias.GetFeatureNames());
                _Clusters.Add(first_cluster);
                return first_cluster;
            }

            var items = _Clusters
                .SelectMany(c => c)
                .AppendFirst(Item)
                .Select(i => new KeyValuePair<double[], T>(Criterias.GetFeaturesVector(i), i))
                .ToArray();

            var rho = Vigilance;
            var betta = Beta;

            var cluster = _Clusters.FirstOrDefault(c => c.SimilarityAndCareTest(items[0].Key, betta, rho));
            if (cluster != null)
                cluster.Add(items[0].Key, Item);
            else
            {
                cluster = new Clauter<T>(
                    items[0].Key,
                    Item,
                    Criterias.GetFeaturesVector,
                    Criterias.GetFeatureNames());
                _Clusters.Add(cluster);
            }

            var iteration = _AddIterationsCount;
            bool repeat;
            do
            {
                repeat = false;
                foreach (var (prototype_vector, item) in items)
                {
                    var new_cluster = _Clusters.FirstOrDefault(c => c.SimilarityAndCareTest(prototype_vector, betta, rho));
                    var old_cluster = _Clusters.First(c => c.Contains(item));

                    if (new_cluster is null || ReferenceEquals(old_cluster, new_cluster)) continue;
                    old_cluster.Remove(item);
                    if (old_cluster.ItemsCount == 0)
                        _Clusters.Remove(old_cluster);
                    new_cluster.Add(prototype_vector, item);
                    repeat = iteration >= 0;
                }
                if (repeat) iteration--;
            } while (repeat);

            return _Clusters.First(c => c.Contains(Item));
        }

        [NotNull]
        public Dictionary<T, Clauter<T>> Add([NotNull] IEnumerable<T> Items)
        {
            var result = new Dictionary<T, Clauter<T>>(new LambdaEqualityComparer<T>(ReferenceEquals, x => x.GetHashCode()));
            foreach (var item in Items)
                result.Add(item, Add(item));
            return result;
        }


        #region Implementation of IEnumerable

        public IEnumerator<Clauter<T>> GetEnumerator() => _Clusters.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)_Clusters).GetEnumerator();

        #endregion
    }
}
