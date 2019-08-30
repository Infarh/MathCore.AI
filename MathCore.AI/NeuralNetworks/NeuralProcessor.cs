using System;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public class NeuralProcessor<TInput, TOutput>
    {
        public delegate void InputFormatter(TInput InputValue, [NotNull] double[] NetworkInput);

        public delegate TOutput OutputFormatter([NotNull] double[] NetworkOutput);

        public delegate void OutputTeachFormatter(TOutput OutputValue, [NotNull] double[] NetworkInput);

        [NotNull] private readonly INeuralNetwork _Network;
        [NotNull] private readonly InputFormatter _InputFormatter;
        [NotNull] private readonly OutputFormatter _OutputFormatter;
        [CanBeNull] private readonly OutputTeachFormatter _OutputTeachFormatter;
        [NotNull] private readonly double[] _Input;
        [NotNull] private readonly double[] _Output;

        public NeuralProcessor(
            [NotNull] INeuralNetwork Network,
            [NotNull] InputFormatter InputFormatter,
            [NotNull] OutputFormatter OutputFormatter,
            [CanBeNull] OutputTeachFormatter OutputTeachFormatter = null)
        {
            _Network = Network ?? throw new ArgumentNullException(nameof(Network));
            _InputFormatter = InputFormatter ?? throw new ArgumentNullException(nameof(InputFormatter));
            _OutputFormatter = OutputFormatter ?? throw new ArgumentNullException(nameof(OutputFormatter));
            _OutputTeachFormatter = OutputTeachFormatter;
            _Input = new double[_Network.InputsCount];
            _Output = new double[_Network.OutputsCount];
        }

        public TOutput Process(TInput Input)
        {
            _InputFormatter(Input, _Input);
            _Network.Process(_Input, _Output);
            return _OutputFormatter(_Output);
        }

        //public double Teach(TInput Input, TOutput ExpectedOutput)
        //{
        //    if (_OutputTeachFormatter is null)
        //        throw new InvalidOperationException($"Не задан метод преобразвоания выходного типа данных {typeof(TOutput)} в массив double[]");
        //    _InputFormatter(Input, _Input);
        //    _OutputTeachFormatter(ExpectedOutput, _Output);
        //    return _Network.Teach(_Input, _Output);
        //}
    }
}
