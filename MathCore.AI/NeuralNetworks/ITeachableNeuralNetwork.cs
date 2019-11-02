using System;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Обучаемая нейронная сеть</summary>
    public interface ITeachableNeuralNetwork : INeuralNetwork
    {
        /// <summary>Создать учителя для нейронной сети</summary>
        /// <returns>Учитель сети</returns>
        [NotNull] INetworkTeacher CreateTeacher();

        /// <summary>Создать учителя для нейронной сети</summary>
        /// <typeparam name="TNetworkTeacher">Тип учителя сети</typeparam>
        /// <param name="Configurator">Метод конфигурирования учителя</param>
        /// <returns>Учитель сети указанного типа</returns>
        [NotNull] TNetworkTeacher CreateTeacher<TNetworkTeacher>([CanBeNull] Action<TNetworkTeacher> Configurator = null)
            where TNetworkTeacher : class, INetworkTeacher;
    }
}