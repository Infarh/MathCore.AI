namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Учитель нейронной сети методом обратного распространения ошибки</summary>
    public interface IBackPropagationTeacher : INetworkTeacher
    {
        /// <summary>Коэффициент скорости обучения (0..1)</summary>
        double Rho { get; set; }
        /// <summary>Инерционность процесса обучения [0..1)</summary>
        double InertialFactor { get; set; }
    }
}
