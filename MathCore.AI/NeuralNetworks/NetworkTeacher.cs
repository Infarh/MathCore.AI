﻿using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Учитель нейронной сети</summary>
    public abstract class NetworkTeacher : INetworkTeacher
    {
        /// <summary>Обучаемая нейронная сеть</summary>
        public INeuralNetwork Network { get; }

        /// <summary>Инициализация нового учителя нейронной сети</summary>
        /// <param name="Network">Обучаемая нейронная сеть</param>
        protected NetworkTeacher(INeuralNetwork Network) => this.Network = Network;

        /// <summary>Обучение сети метдом обратного распространения ошибки</summary>
        /// <param name="Input">Массив входа</param>
        /// <param name="Output">Массив выхода</param>
        /// <param name="Expected">Ожилаемое значение на выходе сети</param>
        /// <param name="Rho">Коэффициент скорости обучения (0..1)</param>
        /// <param name="InertialFactor">Инерционность процесса обцчения [0..1)</param>
        /// <returns>Среднеквадратическая ошибка обучения</returns>
        public abstract double Teach([NotNull] double[] Input, [NotNull] double[] Output, [NotNull] double[] Expected, double Rho = 0.2, double InertialFactor = 0);

        /// <summary>Выбрать бучший вариант</summary>
        public abstract void SetBestVariant();
    }
}