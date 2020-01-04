using System;
using MathCore.Annotations;
// ReSharper disable UnusedMember.Global
// ReSharper disable UnusedType.Global

namespace MathCore.AI.NeuralNetworks
{
    /// <summary>Нейронный процессор</summary>
    /// <typeparam name="TInput">Тип входных данных</typeparam>
    /// <typeparam name="TOutput">Тип выходных данных</typeparam>
    public class NeuralProcessor<TInput, TOutput>
    {
        /// <summary>Метод преобразования типа входных данных в массив вещественных чисел - массив признаков</summary>
        /// <param name="InputValue">Входной объект</param>
        /// <param name="NetworkInput">Массив входных признаков, подаваемый на вход нейронной сети</param>
        /// <remarks>Цель метода определить - как входной объект отображается (проецируется) на массив входов сети</remarks>
        public delegate void InputFormatter(TInput InputValue, [NotNull] double[] NetworkInput);

        /// <summary>Метод преобразования массива вещественных чисел - массива выходных признаков нейронной сети в объект выхода</summary>
        /// <param name="NetworkOutput">Массив выходных признаков нейронной сети</param>
        /// <returns>Объект, сформированный на основе массива признаков, рассчитанных нейронной сетью</returns>
        public delegate TOutput OutputFormatter([NotNull] double[] NetworkOutput);

        /// <summary>Метод преобразования выходного значения в массив значений выхода сети (используется в процессе обучения)</summary>
        /// <param name="Output">Выходное значение</param>
        /// <param name="NetworkOutput">Массив значений на выходе нейронной сети</param>
        public delegate void BackOutputFormatter(TOutput Output, double[] NetworkOutput);

        /// <summary>Нейронная сеть, осуществляющая преобразование входного набора признаков в выходной</summary>
        [NotNull] private readonly INeuralNetwork _Network;

        /// <summary>Метод формирования входного набора признаков на основе входного объекта</summary>
        [NotNull] private readonly InputFormatter _InputFormatter;

        /// <summary>Метод формирования выходного объекта на основе набора признаков, сформированного нейронной сетью</summary>
        [NotNull] private readonly OutputFormatter _OutputFormatter;

        /// <summary>Массив вещественных чисел - вектор входных признаков</summary>
        [NotNull] private readonly double[] _Input;

        /// <summary>Массив вещественных чисел - вектор выходных признаков</summary>
        [NotNull] private readonly double[] _Output;

        /// <summary>Создать новый нейронный процессор</summary>
        /// <param name="Network">Нейронная сеть</param>
        /// <param name="InputFormatter">Метод формирования вектора признаков входного воздействия</param>
        /// <param name="OutputFormatter">Метод формирования выходного значения на основе вектора признаков, формируемого сетью</param>
        public NeuralProcessor(
            [NotNull] INeuralNetwork Network,
            [NotNull] InputFormatter InputFormatter,
            [NotNull] OutputFormatter OutputFormatter)
        {
            _Network = Network ?? throw new ArgumentNullException(nameof(Network));
            _InputFormatter = InputFormatter ?? throw new ArgumentNullException(nameof(InputFormatter));
            _OutputFormatter = OutputFormatter ?? throw new ArgumentNullException(nameof(OutputFormatter));
            _Input = new double[_Network.InputsCount];
            _Output = new double[_Network.OutputsCount];
        }

        /// <summary>Обработать значение</summary>
        /// <param name="Input">Входное значение</param>
        /// <returns>Выходное значение</returns>
        public TOutput Process(TInput Input)
        {
            _InputFormatter(Input, _Input);
            _Network.Process(_Input, _Output);
            return _OutputFormatter(_Output);
        }

        /// <summary>Создать учителя сети</summary>
        /// <param name="Configurator">Метод конфигурации учителя</param>
        /// <returns>Учитель нейронной сети</returns>
        [NotNull]
        public TNetworkTeacher CreateTeacher<TNetworkTeacher>([CanBeNull] Action<TNetworkTeacher> Configurator = null)
            where TNetworkTeacher : class, INetworkTeacher
        {
            if (!(_Network is ITeachableNeuralNetwork teachable_network))
                throw new InvalidOperationException("Сеть не является обучаемой");
            return teachable_network.CreateTeacher(Configurator);
        }

        private class ProcessorTeacher : INeuralProcessorTeacher<TInput, TOutput>
        {
            [NotNull] private readonly NeuralProcessor<TInput, TOutput> _NeuralProcessor;
            [NotNull] private readonly BackOutputFormatter _BackOutputFormatter;
            [NotNull] private readonly INetworkTeacher _Teacher;
            [NotNull] private readonly double[] _Input;
            [NotNull] private readonly double[] _Output;
            [NotNull] private readonly double[] _Expected;

            public ProcessorTeacher(
                [NotNull] NeuralProcessor<TInput, TOutput> NeuralProcessor, 
                [NotNull] BackOutputFormatter BackOutputFormatter, 
                [NotNull] INetworkTeacher Teacher)
            {
                _NeuralProcessor = NeuralProcessor;
                _BackOutputFormatter = BackOutputFormatter;
                _Teacher = Teacher;
                var network = Teacher.Network;
                _Input = new double[network.InputsCount];
                _Output = new double[network.OutputsCount];
                _Expected = new double[_Output.Length];
            }

            public INeuralNetwork Network => _NeuralProcessor._Network;

            public double Teach(TInput Input, TOutput Expected)
            {
                _NeuralProcessor._InputFormatter(Input, _Input);
                _BackOutputFormatter(Expected, _Expected);
                return _Teacher.Teach(_Input, _Output, _Expected);
            } 

            public double Teach(TInput Input, TOutput Expected, out TOutput Output)
            {
                _NeuralProcessor._InputFormatter(Input, _Input);
                _BackOutputFormatter(Expected, _Expected);
                var error = _Teacher.Teach(_Input, _Output, _Expected);
                Output = _NeuralProcessor._OutputFormatter(_Output);
                return error;
            }
        }

        /// <summary>Получить учитель процессора</summary>
        /// <typeparam name="TNetworkTeacher">Тип учителя</typeparam>
        /// <param name="BackOutputFormatter">Метод обратного преобразования выходного значения в массив значений выхода сети</param>
        /// <param name="Configurator">Метод конфигурации учителя</param>
        /// <returns>Учитель процессора</returns>
        [NotNull]
        public INeuralProcessorTeacher<TInput, TOutput> CreateTeacher<TNetworkTeacher>(
            [NotNull] BackOutputFormatter BackOutputFormatter,
            [CanBeNull] Action<TNetworkTeacher> Configurator = null)
            where TNetworkTeacher : class, INetworkTeacher =>
            new ProcessorTeacher(this, BackOutputFormatter, CreateTeacher(Configurator));
    }
}
