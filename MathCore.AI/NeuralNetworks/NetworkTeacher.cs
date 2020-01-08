using System;
using MathCore.Annotations;
// ReSharper disable UnusedMember.Global

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Учитель нейронной сети</summary>
    public abstract class NetworkTeacher : INetworkTeacher
    {
        public INeuralNetwork Network { get; }

        /// <summary>Инициализация нового учителя нейронной сети</summary>
        /// <param name="Network">Обучаемая нейронная сеть</param>
        protected NetworkTeacher(INeuralNetwork Network) => this.Network = Network;

        /// <inheritdoc />
        public abstract double Teach(double[] Input, double[] Output, double[] Expected);

        [NotNull]
        public TNetworkTeacher As<TNetworkTeacher>([CanBeNull] Action<TNetworkTeacher> Configurator = null) 
            where TNetworkTeacher : class, INetworkTeacher
        {
            var teacher = this as TNetworkTeacher ?? throw new InvalidOperationException($"Учитель не поддерживает интерфейс {typeof(TNetworkTeacher)}");
            Configurator?.Invoke(teacher);
            return teacher;
        }

        /// <summary>Выбрать лучший вариант</summary>
        public abstract void SetBestVariant();
    }
}