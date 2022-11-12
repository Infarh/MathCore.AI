namespace MathCore.AI.NeuralNetworks;

/// <summary>Обучаемая нейронная сеть</summary>
public interface ITeachableNeuralNetwork : INeuralNetwork
{
    /// <summary>Создать учителя для нейронной сети</summary>
    /// <returns>Учитель сети</returns>
    INetworkTeacher CreateTeacher();

    /// <summary>Создать учителя для нейронной сети</summary>
    /// <typeparam name="TNetworkTeacher">Тип учителя сети</typeparam>
    /// <param name="Configurator">Метод конфигурирования учителя</param>
    /// <returns>Учитель сети указанного типа</returns>
    TNetworkTeacher CreateTeacher<TNetworkTeacher>(Action<TNetworkTeacher>? Configurator = null)
        where TNetworkTeacher : class, INetworkTeacher;
}