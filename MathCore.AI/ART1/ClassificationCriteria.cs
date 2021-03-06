﻿using System;
using MathCore.Annotations;

namespace MathCore.AI.ART1
{
    /// <summary>Критерий классификации</summary>
    /// <typeparam name="T">Тип объектов классификации</typeparam>
    public sealed class ClassificationCriteria<T>: IEquatable<ClassificationCriteria<T>>, IEquatable<Func<T, double>>
    {
        /// <summary>Функция оценки критерия</summary>
        [NotNull] private readonly Func<T, double> _Criteria;

        /// <summary>Название критерия</summary>
        [CanBeNull] public string Name { get; }

        /// <summary>Инициализация нового критерия классификации</summary>
        /// <param name="Criteria">Функция критерия, сопоставляющая объект с числовым значением</param>
        public ClassificationCriteria([NotNull] Func<T, double> Criteria) => _Criteria = Criteria ?? throw new ArgumentNullException(nameof(Criteria));

        /// <summary>Инициализация нового критерия классификации</summary>
        /// <param name="Name">Имя критерия</param>
        /// <param name="Criteria">Функция критерия, сопоставляющая объект с числовым значением</param>
        public ClassificationCriteria([CanBeNull] string Name, [NotNull] Func<T, double> Criteria) : this(Criteria) => this.Name = Name;

        /// <summary>Получить числовое значение критерия для оценки</summary>
        /// <param name="Item">Оцениваемый объект</param>
        /// <returns>Числовое значение оценки для данного критерия</returns>
        public double GetFeatureValue(T Item) => _Criteria(Item);

        /// <inheritdoc />
        public bool Equals(ClassificationCriteria<T> other) => other != null && (ReferenceEquals(this, other) || _Criteria.Equals(other._Criteria) && Name == other.Name);

        /// <inheritdoc />
        public bool Equals(Func<T, double> other) => Equals(_Criteria, other);

        /// <inheritdoc />
        public override bool Equals(object obj) => ReferenceEquals(this, obj) || obj is ClassificationCriteria<T> other && Equals(other);

        /// <inheritdoc />
        public override int GetHashCode() { unchecked { return (_Criteria.GetHashCode() * 397) ^ (Name != null ? Name.GetHashCode() : 0); } }

        /// <summary>Оператор равенства двух критериев классификации</summary>
        /// <param name="left">Левый сравниваемый критерий классификации</param>
        /// <param name="right">Правый сравниваемый критерий классификации</param>
        /// <returns>Истина, если критерии равны</returns>
        public static bool operator ==([CanBeNull] ClassificationCriteria<T> left, [CanBeNull] ClassificationCriteria<T> right) => Equals(left, right);

        /// <summary>Оператор неравенства двух критериев классификации</summary>
        /// <param name="left">Левый сравниваемый критерий классификации</param>
        /// <param name="right">Правый сравниваемый критерий классификации</param>
        /// <returns>Истина, если критерии неравны</returns>
        public static bool operator !=([CanBeNull] ClassificationCriteria<T> left, [CanBeNull] ClassificationCriteria<T> right) => !Equals(left, right);
    }
}