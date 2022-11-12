using System.Collections;
using System.Diagnostics;
using System.Text;

namespace MathCore.AI.ART1;

/// <summary>Кластер элементов</summary>
/// <typeparam name="T">Тип классифицируемых элементов</typeparam>
[DebuggerDisplay("{" + nameof(ToDebuggerString) + "()}")]
public sealed class Cluster<T> : IEnumerable<T>
{
    /// <summary> Вектор признаков</summary>
    private readonly double[] _FeatureVector;

    /// <summary>Функция вычисления вектора признаков кластера для элемента</summary>
    private readonly Func<T, double[]> _Classificator;

    /// <summary>Имена критериев классификации</summary>
    private readonly string[] _FeatureNames;

    /// <summary>Элементы кластера</summary>
    private readonly HashSet<T> _Items = new(new LambdaEqualityComparer<T>((v1, v2) => ReferenceEquals(v1, v2), v => ((object)v).GetHashCode()));

    /// <summary>Число элементов в кластере</summary>
    public int ItemsCount => _Items.Count;

    /// <summary>Инициализация нового кластера</summary>
    /// <param name="FeatureVector">Вектор признаков</param>
    /// <param name="Item">Первый элемент кластера</param>
    /// <param name="Classificator">Функция вычисления вектора признаков кластера для элемента</param>
    /// <param name="FeatureNames">Имена классификаторов</param>
    internal Cluster(double[] FeatureVector, T Item, Func<T, double[]> Classificator, string[] FeatureNames)
    {
        _FeatureVector = FeatureVector;
        _Classificator = Classificator;
        _FeatureNames  = FeatureNames;
        _Items.Add(Item);
    }

    /// <summary>Добавление нового элемента в кластер</summary>
    /// <param name="Prototype">Вектор признаков прототипа</param>
    /// <param name="Item">Добавляемый элемент</param>
    internal void Add(double[] Prototype, T Item)
    {
        if (_Items.Contains(Item)) return;
        _Items.Add(Item);
        for (var i = 0; i < Prototype.Length; i++)
            _FeatureVector[i] *= Prototype[i];
    }

    /// <summary>Очистка кластера от элементов</summary>
    internal void Clear()
    {
        _Items.Clear();
        _FeatureVector.Initialize(0);
    }

    /// <summary>Проверка - состоит ли элемент в кластере</summary>
    /// <param name="Item">Проверяемый элемент</param>
    /// <returns>Истина, если элемент находится в кластере</returns>
    public bool Contains(T Item) => _Items.Contains(Item);

    /// <summary>Удаление элемента из кластера</summary>
    /// <param name="Item">Удаляемый элемент</param>
    /// <returns>Истина, если элемент был и удалён из кластера</returns>
    internal void Remove(T Item)
    {
        if (!_Items.Remove(Item)) return;

        var feature_vector = _FeatureVector;
        var i              = 0;
        foreach (var item_features in _Items.Select(item => _Classificator(item)))
        {
            if (i == 0) item_features.CopyTo(feature_vector, 0);
            else
                for (var j = 0; j < feature_vector.Length; j++)
                    feature_vector[j] *= item_features[j];
            i++;
        }
    }

    /// <summary>Проверка на принадлежность элемента кластеру</summary>
    /// <param name="Prototype">Вектор-прототип</param>
    /// <param name="Betta">Бета-параметр (разрушения связей) - чем больше, тем больше кластеров будет образовано</param>
    /// <param name="Vigilance">Параметр внимательности (0;1]</param>
    /// <returns></returns>
    internal bool SimilarityAndCareTest(double[] Prototype, double Betta, double Vigilance)
    {
        var correlation            = 0d;
        var prototype_conditionaly = 0d;
        var features_conditionaly  = 0d;
        var feature_vector         = _FeatureVector;
        var length                 = feature_vector.Length;

        for (var i = 0; i < length; i++)
        {
            var p = Prototype[i];
            var e = feature_vector[i];
            correlation            += p * e;
            prototype_conditionaly += p;
            features_conditionaly  += e;
        }

        return correlation / (Betta + prototype_conditionaly) > features_conditionaly / (Betta + length)
            && correlation / features_conditionaly < Vigilance;
    }

    #region Overrides of Object

    /// <inheritdoc />
    public override string ToString()
    {
        var result = new StringBuilder();
        for (var i = 0; i < _FeatureVector.Length; i++)
            if (_FeatureVector[i] != 0)
                if (_FeatureVector[i] == 1)
                    result.AppendFormat("{0}, ", _FeatureNames[i]);
                else
                    result.AppendFormat("{0}({1}), ", _FeatureNames[i], _FeatureVector[i].RoundAdaptive(2));

        if (result.Length > 0) 
            result.Length -= 2;
        
        return $"[ {result} ]";
    }

    private string ToDebuggerString() => $"{this}:{string.Join(", ", _Items)}";

    #endregion

    #region Implementation of IEnumerable

    /// <inheritdoc />
    public IEnumerator<T> GetEnumerator() => _Items.GetEnumerator();

    IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)_Items).GetEnumerator();

    #endregion
}