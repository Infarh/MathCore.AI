namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Обучаемая нейронная сеть</summary>
    public interface ITeachableNeuralNetwork : INeuralNetwork
    {
        /// <summary>Создать учителя для нейронной сети</summary>
        /// <returns>Учитель нейронной сети</returns>
        INetworkTeacher CreateTeacher();
    }
}