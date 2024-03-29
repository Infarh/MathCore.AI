﻿// ReSharper disable UnusedMember.Global
// ReSharper disable UnusedType.Global
// ReSharper disable ConvertToAutoProperty
// ReSharper disable MemberCanBePrivate.Global
// ReSharper disable ConvertToAutoPropertyWhenPossible
namespace MathCore.AI.NeuralNetworks;

/// <summary>Нейронный процессор</summary>
/// <typeparam name="TInput">Тип входных данных</typeparam>
/// <typeparam name="TOutput">Тип выходных данных</typeparam>
public class NeuralProcessor<TInput, TOutput>
{
    /// <summary>Метод преобразования типа входных данных в массив вещественных чисел - массив признаков</summary>
    /// <param name="InputValue">Входной объект</param>
    /// <param name="NetworkInput">Массив входных признаков, подаваемый на вход нейронной сети</param>
    /// <remarks>Цель метода определить - как входной объект отображается (проецируется) на массив входов сети</remarks>
    public delegate void InputFormatter(TInput InputValue, double[] NetworkInput);

    /// <summary>Метод преобразования массива вещественных чисел - массива выходных признаков нейронной сети в объект выхода</summary>
    /// <param name="NetworkOutput">Массив выходных признаков нейронной сети</param>
    /// <returns>Объект, сформированный на основе массива признаков, рассчитанных нейронной сетью</returns>
    public delegate TOutput OutputFormatter(double[] NetworkOutput);

    /// <summary>Метод преобразования выходного значения в массив значений выхода сети (используется в процессе обучения)</summary>
    /// <param name="Output">Выходное значение</param>
    /// <param name="NetworkOutput">Массив значений на выходе нейронной сети</param>
    public delegate void BackOutputFormatter(TOutput Output, double[] NetworkOutput);

    /// <summary>Нейронная сеть, осуществляющая преобразование входного набора признаков в выходной</summary>
    private readonly INeuralNetwork _Network;

    /// <summary>Метод формирования входного набора признаков на основе входного объекта</summary>
    private readonly InputFormatter _InputFormatter;

    /// <summary>Метод формирования выходного объекта на основе набора признаков, сформированного нейронной сетью</summary>
    private readonly OutputFormatter _OutputFormatter;

    /// <summary>Массив вещественных чисел - вектор входных признаков</summary>
    private readonly double[] _Input;

    /// <summary>Массив вещественных чисел - вектор выходных признаков</summary>
    private readonly double[] _Output;

    /// <summary>Очищать массив входа сети перед обработкой данных</summary>
    private bool _ClearInput = true;

    /// <summary>Очищать массив входа сети перед обработкой данных</summary>
    public bool ClearInput { get => _ClearInput; set => _ClearInput = value; }

    /// <summary>Создать новый нейронный процессор</summary>
    /// <param name="Network">Нейронная сеть</param>
    /// <param name="InputFormatter">Метод формирования вектора признаков входного воздействия</param>
    /// <param name="OutputFormatter">Метод формирования выходного значения на основе вектора признаков, формируемого сетью</param>
    public NeuralProcessor(
        INeuralNetwork Network,
        InputFormatter InputFormatter,
        OutputFormatter OutputFormatter)
    {
        _Network         = Network.NotNull();
        _InputFormatter  = InputFormatter.NotNull();
        _OutputFormatter = OutputFormatter.NotNull();
        _Input           = new double[_Network.InputsCount];
        _Output          = new double[_Network.OutputsCount];
    }

    /// <summary>Обработать значение</summary>
    /// <param name="Input">Входное значение</param>
    /// <returns>Выходное значение</returns>
    public TOutput Process(TInput Input)
    {
        if (_ClearInput)
            Array.Clear(_Input, 0, _Input.Length);
        _InputFormatter(Input, _Input);
        _Network.Process(_Input, _Output);
        return _OutputFormatter(_Output);
    }

    /// <summary>Создать учителя сети</summary>
    /// <param name="Configurator">Метод конфигурации учителя</param>
    /// <returns>Учитель нейронной сети</returns>
    public TNetworkTeacher CreateTeacher<TNetworkTeacher>(Action<TNetworkTeacher>? Configurator)
        where TNetworkTeacher : class, INetworkTeacher =>
        _Network is not ITeachableNeuralNetwork teachable_network
            ? throw new InvalidOperationException("Сеть не является обучаемой")
            : teachable_network.CreateTeacher(Configurator);

    /// <summary>Создать учителя сети</summary><returns>Учитель нейронной сети</returns>
    public INetworkTeacher CreateTeacher() =>
        _Network is not ITeachableNeuralNetwork teachable_network
            ? throw new InvalidOperationException("Сеть не является обучаемой")
            : teachable_network.CreateTeacher();

    /// <summary>Учитель нейронной сети, используемой в нейропроцессоре</summary>
    private class ProcessorTeacher : INeuralProcessorTeacher<TInput, TOutput>
    {
        /// <summary>Обучаемый нейронный процессор</summary>
        private readonly NeuralProcessor<TInput, TOutput> _NeuralProcessor;

        /// <summary>Метод преобразования значения выхода нейропроцессора в массив вещественных значений выхода нейронной сети</summary>
        private readonly BackOutputFormatter _BackOutputFormatter;

        /// <summary>Объект, осуществляющий обучение нейронной сети</summary>
        private readonly INetworkTeacher _Teacher;

        /// <summary>Значения входа нейронной сети </summary>
        private readonly double[] _Input;

        /// <summary>Текущие значения выхода сети в процессе обучения</summary>
        private readonly double[] _Output;

        /// <summary>Массив ожидаемых значений на выходе сети</summary>
        private readonly double[] _Expected;

        /// <summary>Очищать вектор входа сети перед каждой итерацией обучения</summary>
        private bool _ClearInput;

        /// <summary>Очищать вектор ожидаемого значения сети перед каждой итерацией обучения</summary>
        private bool _ClearExpected = true;

        /// <inheritdoc />
        public bool ClearInput { get => _ClearInput; set => _ClearInput = value; }

        /// <inheritdoc />
        public bool ClearExpected { get => _ClearExpected; set => _ClearExpected = value; }

        /// <summary>Инициализация нового экземпляра <see cref="ProcessorTeacher"/></summary>
        /// <param name="NeuralProcessor">Обучаемый нейронный процессор</param>
        /// <param name="BackOutputFormatter">Метод упаковки ожидаемого значения на выходе нейронной сети в массив вещественных чисел - значений выходов сети</param>
        /// <param name="Teacher">Учитель сети</param>
        public ProcessorTeacher(
            NeuralProcessor<TInput, TOutput> NeuralProcessor,
            BackOutputFormatter BackOutputFormatter,
            INetworkTeacher Teacher)
        {
            _NeuralProcessor     = NeuralProcessor.NotNull();
            _BackOutputFormatter = BackOutputFormatter.NotNull();
            _Teacher             = Teacher.NotNull();
            var network = Teacher.Network;
            _Input    = new double[network.InputsCount];
            _Output   = new double[network.OutputsCount];
            _Expected = new double[_Output.Length];
        }

        /// <inheritdoc />
        public INeuralNetwork Network => _NeuralProcessor._Network;

        /// <inheritdoc />
        public double Teach(TInput Input, TOutput Expected)
        {
            if (_ClearInput) Array.Clear(_Input, 0, _Input.Length);
            if(_ClearExpected) Array.Clear(_Expected, 0, _Expected.Length);

            _NeuralProcessor._InputFormatter(Input, _Input);
            _BackOutputFormatter(Expected, _Expected);
            return _Teacher.Teach(_Input, _Output, _Expected);
        }

        /// <inheritdoc />
        public double Teach(TInput Input, TOutput Expected, out TOutput Output)
        {
            var error = Teach(Input, Expected);
            Output = _NeuralProcessor._OutputFormatter(_Output);
            return error;
        }
    }

    /// <summary>Получить учитель процессора</summary>
    /// <typeparam name="TNetworkTeacher">Тип учителя</typeparam>
    /// <param name="BackOutputFormatter">Метод обратного преобразования выходного значения в массив значений выхода сети</param>
    /// <param name="Configurator">Метод конфигурации учителя</param>
    /// <returns>Учитель процессора</returns>
    public INeuralProcessorTeacher<TInput, TOutput> CreateTeacher<TNetworkTeacher>(
        BackOutputFormatter BackOutputFormatter,
        Action<TNetworkTeacher>? Configurator)
        where TNetworkTeacher : class, INetworkTeacher =>
        new ProcessorTeacher(this, BackOutputFormatter, CreateTeacher(Configurator));

    /// <summary>Получить учитель процессора</summary>
    /// <param name="BackOutputFormatter">Метод обратного преобразования выходного значения в массив значений выхода сети</param>
    /// <returns>Учитель процессора</returns>
    public INeuralProcessorTeacher<TInput, TOutput> CreateTeacher(BackOutputFormatter BackOutputFormatter) =>
        new ProcessorTeacher(this, BackOutputFormatter, CreateTeacher());
}