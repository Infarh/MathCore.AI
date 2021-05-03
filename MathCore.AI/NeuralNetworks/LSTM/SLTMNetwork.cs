#nullable enable
using System;

using MathCore.AI.NeuralNetworks.ActivationFunctions;

namespace MathCore.AI.NeuralNetworks.LSTM
{
    public class SLTMNetwork
    {
        private readonly double[] _Input;
        private readonly double[] _State;
        private readonly double[] _Outputs;

        private readonly double[,] _StateGate;

        public SLTMNetwork(int Inputs, int Outputs, int State)
        {
            _Input = new double[Inputs];
            _State = new double[State];
            _Outputs = new double[Outputs];

            _StateGate = new double[State, Inputs + Outputs];
        }

        /// <summary>Обработка итерации сети</summary>
        /// <param name="InputOutput">Массив объединённого входа-выхода</param>
        /// <param name="Output">Выход сети</param>
        /// <param name="State">Состояние</param>
        /// <param name="StateGateW">Массив слоя вентиля подавления состояния</param>
        /// <param name="StateGateOffset">Смещение слоя вентиля подавления состояния</param>
        /// <param name="UpdateStateW">Коэффициенты передачи слоя обновления состояния на основе входа-выхода</param>
        /// <param name="UpdateStateOffset">Смещение слоя обновления состояния на основе входа-выхода</param>
        /// <param name="UpdateStateGateW">Коэффициенты передачи слоя вентиля обновления состояния на основе входа-выхода</param>
        /// <param name="UpdateStateGateOffset">Смещение слоя вентиля обновления состояния на основе входа-выхода</param>
        /// <param name="OutputGateW">Коэффициенты слоя вентиля выхода</param>
        /// <param name="OutputGateOffset">Смещения слоя вентиля выхода</param>
        /// <param name="OutputGateState">Состояние вентиля выхода</param>
        /// <param name="UpdateOutputW">Коэффициенты передачи состояния на выход</param>
        /// <param name="UpdateOutputOffset">Смещение слоя выхода</param>
        public static void Process(
            double[] InputOutput,
            Span<double> Output,
            double[] State,
            double[,] StateGateW,
            double[] StateGateOffset,
            double[,] UpdateStateW,
            double[] UpdateStateOffset,
            double[,] UpdateStateGateW,
            double[] UpdateStateGateOffset,
            double[,] OutputGateW,
            double[] OutputGateOffset,
            double[] OutputGateState,
            double[,] UpdateOutputW,
            double[] UpdateOutputOffset)
        {
            var state_length = State.Length;
            var input_output_length = InputOutput.Length;
            var output_length = Output.Length;

            // Расчёт состояния
            for (var i = 0; i < state_length; i++)
            {
                // Инициализация аккумуляторов значениями смещений слоёв
                var forget_state_gate = StateGateOffset[i];         // Аккумулятор вентиля сброса состояния
                var update_state_gate = UpdateStateGateOffset[i];   // Аккумулятор вентиля обновления состояния
                var update_state = UpdateStateOffset[i];            // Аккумулятор обновления состояния

                // Расчёт выходов слоёв для обновления состояния
                for (var j = 0; j < input_output_length; j++)
                {
                    forget_state_gate += StateGateW[i, j] * InputOutput[j];
                    update_state_gate += UpdateStateGateW[i, j] * InputOutput[j];
                    update_state += UpdateStateW[i, j] * InputOutput[j];
                }

                State[i] =
                    State[i] * Sigmoid.Activation(forget_state_gate)                        // Затухание состояния
                    + Th.Activation(update_state) * Sigmoid.Activation(update_state_gate);  // Обновление состояния
            }

            // Расчёт коэффициентов выходного вентиля
            for (var i = 0; i < output_length; i++)
            {
                // Инициализация выходного вентиля значением смещения слоя
                var s = OutputGateOffset[i];                        // Аккумулятор выходного вентиля

                // Расчёт выхода вентиля
                for (var j = 0; j < input_output_length; j++)
                    s += OutputGateW[i, j] * InputOutput[j];

                OutputGateState[i] = Sigmoid.Activation(s);
            }

            // Расчёт выхода на основе состояния
            for (var i = 0; i < output_length; i++)
            {
                // Инициализация аккумулятора выхода значением смещения слоя
                var s = UpdateOutputOffset[i];                      // Аккумулятор выхода

                // Расчёт выхода
                for (var j = 0; j < state_length; j++) 
                    s += UpdateOutputW[i, j] * State[j];

                // Выход определяется как произведение аккумулятора и выходного вентиля
                Output[i] = Th.Activation(s) * OutputGateState[i];
            }
        }
    }
}
