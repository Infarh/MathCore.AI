using System.Collections;

// ReSharper disable ClassNeverInstantiated.Global
namespace MathCore.AI.ART1;

/// <summary>Классификатор объектов по алгоритму ART-1</summary>
/// <typeparam name="T">Тип классифицируемых объектов</typeparam>
public class Classificator<T> : IEnumerable<Cluster<T>>
    where T : class
{
    /// <summary>Параметр внимательности (0;1]</summary>
    /// <remarks>Определяет размер кластера</remarks>
    private double _Vigilance = 0.9;

    /// <summary>Бета-параметр (разрушения связей) - чем больше, тем больше кластеров будет образовано</summary>
    private double _Beta = 1;

    /// <summary>Количество имитаций перераспределения образцов по кластерам для операции добавления</summary>
    private int _AddIterationsCount = 50;

    /// <summary>Образованные кластеры</summary>
    private readonly List<Cluster<T>> _Clusters = new();

    /// <summary>Параметр внимательности (0;1]</summary>
    /// <remarks>Определяет размер кластера</remarks>
    public double Vigilance
    {
        get => _Vigilance;
        set
        {
            if (value is <= 0 or > 1)
                throw new ArgumentOutOfRangeException(
                    nameof(value), 
                    value, 
                    "Vigilance должно быть в пределах от 0 до 1 (включительно)");
            _Vigilance = value;
        }
    }

    /// <summary>Бета-параметр (разрушения связей) - чем больше, тем больше кластеров будет образовано</summary>
    public double Beta
    {
        get => _Beta;
        set
        {
            if (value is <= 0 or > 1)
                throw new ArgumentOutOfRangeException(
                    nameof(value), 
                    value, 
                    "Betta должно быть в пределах от 0 до 1 (включительно)");
            _Beta = value;
        }
    }

    /// <summary>Количество имитаций перераспределения образцов по кластерам для операции добавления</summary>
    public int AddOperationIterationCount
    {
        get => _AddIterationsCount;
        set => _AddIterationsCount = value;
    }
        
    /// <summary>Критерии классификации</summary>
    public ClassificationCriterias<T> Criterias { get; } = new();

    /// <summary>Сформированные кластеры</summary>
    public IReadOnlyCollection<Cluster<T>> Clusters => _Clusters;

    /// <summary>Добавить новый объект в классификатор</summary>
    /// <param name="Item">Классифицируемый объект</param>
    /// <returns>Кластер, в который отнесён добавляемый объект</returns>
    public Cluster<T> Add(T Item)
    {
        if (Item is null) throw new ArgumentNullException(nameof(Item));
               

        if (_Clusters.Count == 0) return CreateFirstCluster(Item);

        static KeyValuePair<double[], T>[] GetAllItems(T item, IEnumerable<Cluster<T>> clusters, ClassificationCriterias<T> criterias) =>
            clusters
               .SelectMany(c => c)
               .AppendFirst(item)
               .Select(i => new KeyValuePair<double[], T>(criterias.GetFeaturesVector(i), i))
               .ToArray();

        var items = GetAllItems(Item, _Clusters, Criterias);

        var cluster = _Clusters.FirstOrDefault(c => c.SimilarityAndCareTest(items[0].Key, Beta, Vigilance));
        if (cluster != null)
            cluster.Add(items[0].Key, Item);
        else
        {
            cluster = new Cluster<T>(
                items[0].Key,
                Item,
                Criterias.GetFeaturesVector,
                Criterias.GetFeatureNames()!);
            _Clusters.Add(cluster);
        }

        CheckClusters(items, _AddIterationsCount);

        return _Clusters.First(c => c.Contains(Item));
    }

    private void CheckClusters(KeyValuePair<double[], T>[] Items, int MaxIterationCount)
    {
        var  iteration = MaxIterationCount;
        bool has_changes;
        do
        {
            has_changes = false;
            foreach (var (prototype_vector, item) in Items)
            {
                var new_cluster = _Clusters.FirstOrDefault(c => c.SimilarityAndCareTest(prototype_vector, Beta, Vigilance));
                var old_cluster = _Clusters.First(c => c.Contains(item));

                if (new_cluster is null || ReferenceEquals(old_cluster, new_cluster)) continue;
                old_cluster.Remove(item);
                if (old_cluster.ItemsCount == 0)
                    _Clusters.Remove(old_cluster);
                new_cluster.Add(prototype_vector, item);
                has_changes = true;
            }

            iteration--;
        } while (has_changes && iteration >= 0);
    }

    private Cluster<T> CreateFirstCluster(T Item)
    {
        var first_cluster = new Cluster<T>(
            Criterias.GetFeaturesVector(Item),
            Item,
            Criterias.GetFeaturesVector,
            Criterias.GetFeatureNames()!);
        _Clusters.Add(first_cluster);
        return first_cluster;
    }

    /// <summary>Добавить элементы в классификатор</summary>
    /// <param name="Items">Классифицируемые элементы</param>
    /// <returns>Словарь классов элементов</returns>
    public Dictionary<T, Cluster<T>> Add(IEnumerable<T> Items)
    {
        var result = new Dictionary<T, Cluster<T>>(new LambdaEqualityComparer<T>(ReferenceEquals, x => x.GetHashCode()));
        foreach (var item in Items)
            result.Add(item, Add(item));
        return result;
    }


    #region Implementation of IEnumerable

    public IEnumerator<Cluster<T>> GetEnumerator() => _Clusters.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)_Clusters).GetEnumerator();

    #endregion
}