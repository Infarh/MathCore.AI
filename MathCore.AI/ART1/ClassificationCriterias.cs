using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using MathCore.Annotations;
// ReSharper disable UnusedMethodReturnValue.Global

namespace MathCore.AI.ART1
{
    /// <summary>Система критериев для классификации объектов</summary>
    /// <typeparam name="T">Тип объектов классификации</typeparam>
    public sealed class ClassificationCriterias<T> : IEnumerable<ClassificationCriteria<T>>
    {
        /// <summary>Список критериев</summary>
        [NotNull] private readonly List<ClassificationCriteria<T>> _Criterias = new List<ClassificationCriteria<T>>();

        /// <summary>Число критериев в классификации</summary>
        public int Count => _Criterias.Count;

        /// <summary>Добавить критерий классификации</summary>
        /// <param name="Name">Имя критерия</param>
        /// <param name="Criteria">Функция критерия</param>
        /// <exception cref="ArgumentException">Если критерий с указанным именем уже существует</exception>
        /// <exception cref="ArgumentException">Если критерий с указанной функцией уже существует</exception>
        /// <returns>Сформированный критерий классификации</returns>
        [NotNull]
        public ClassificationCriteria<T> Add([CanBeNull] string Name, [NotNull] Func<T, double> Criteria)
        {
            if (_Criterias.Any(c => c.Name == Name)) throw new ArgumentException($"Критерий с именем {Name} уже существует", nameof(Name));
            if (_Criterias.Any(c => c.Equals(Criteria))) throw new ArgumentException("Критерий с указанной функцией уже существует", nameof(Criteria));

            var criteria = new ClassificationCriteria<T>(Name, Criteria ?? throw new ArgumentNullException(nameof(Criteria)));
            _Criterias.Add(criteria);
            return criteria;
        }

        /// <summary>Добавить критерий классификации</summary>
        /// <param name="Criteria">Функция критерия</param>
        /// <exception cref="ArgumentException">Если критерий с указанной функцией уже существует</exception>
        /// <returns>Сформированный критерий классификации</returns>
        [NotNull]
        public ClassificationCriteria<T> Add([NotNull] Func<T, double> Criteria)
        {
            if (_Criterias.Any(c => c.Equals(Criteria))) throw new ArgumentException("Критерий с указанной функцией уже существует", nameof(Criteria));

            var criteria = new ClassificationCriteria<T>(Criteria ?? throw new ArgumentNullException(nameof(Criteria)));
            _Criterias.Add(criteria);
            return criteria;
        }

        /// <summary>Получить вектор оценок классификатора</summary>
        /// <param name="Item">Оцениваемый объект</param>
        /// <returns>Вектор числовых значений оценок классификатора</returns>
        [NotNull]
        public double[] GetFeaturesVector(T Item)
        {
            var result = new double[Count];
            for (var i = 0; i < result.Length; i++)
                result[i] = _Criterias[i].GetFeatureValue(Item);
            return result;
        }

        /// <summary>Получить вектор имён критериев классификации</summary>
        /// <returns>Массив имён классификаторов</returns>
        [NotNull, ItemCanBeNull] public string[] GetFeatureNames() => _Criterias.Select(f => f.Name).ToArray();

        #region Implementation of IEnumerable

        /// <inheritdoc />
        public IEnumerator<ClassificationCriteria<T>> GetEnumerator() => _Criterias.GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => ((IEnumerable)_Criterias).GetEnumerator();

        #endregion
    }
}