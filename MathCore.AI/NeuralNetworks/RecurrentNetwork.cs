using System;
using System.Collections.Generic;
using System.Linq;
using MathCore.AI.NeuralNetworks.ActivationFunctions;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public class RecurrentNetwork : MultilayerPerceptron
    {
        [NotNull] protected readonly double[][,] _LayerFeedbacks;
        [NotNull] protected readonly double[][] _LastOutputs;
        [NotNull] protected readonly double[] _Output;

        public RecurrentNetwork([NotNull, ItemNotNull] double[][,] Weights, [NotNull, ItemNotNull] double[][,] LayerFeedbacks)
            : base(Weights)
        {
            _LayerFeedbacks = LayerFeedbacks;
            _LastOutputs = new double[LayersCount][];
            _Output = new double[OutputsCount];

            foreach (var layer in Layer)
            {
                var layer_index = layer.LayerIndex;
                var neurons_count = layer.OutputsCount;
                var feedbacks = _LayerFeedbacks[layer_index];
                if (feedbacks.GetLength(0) != neurons_count) throw new ArgumentException($"Число строк матрицы обратной связи ({feedbacks.GetLength(0)}) на уровне {layer_index} не равно числу нейронов слоя ({neurons_count})");
                if (feedbacks.GetLength(1) != neurons_count) throw new ArgumentException($"Число столбцов матрицы обратной связи ({feedbacks.GetLength(1)}) на уровне {layer_index} не равно числу нейронов слоя ({neurons_count})");
                _LastOutputs[layer_index] = new double[neurons_count];
            }
        }

        public delegate double NetworkFeddbackCoefficientInitializer(int Layer, int Input, int Output);

        public RecurrentNetwork(
            int InputsCount,
            [NotNull] IEnumerable<int> NeuronsCount,
            LayerInitializer LayerInitializer,
            NetworkFeddbackCoefficientInitializer FeedbackInitializer)
            : this(
                InputsCount,
                (NeuronsCount ?? throw new ArgumentNullException(nameof(NeuronsCount))).ToArray(),
                LayerInitializer)
        {
            if (FeedbackInitializer is null) return;
            for (var layer_index = 0; layer_index < LayersCount; layer_index++)
            {
                var feedback_weights = _LayerFeedbacks[layer_index];
                var inputs_count = feedback_weights.GetLength(0);
                var outputs_count = feedback_weights.GetLength(1);
                for (var input_index = 0; input_index < inputs_count; input_index++)
                    for (var output_index = 0; output_index < outputs_count; output_index++)
                        FeedbackInitializer(layer_index, input_index, output_index);
            }
        }

        private RecurrentNetwork(int InputsCount, [NotNull] int[] NeuronsCount, LayerInitializer LayerInitializer)
            : base(InputsCount, NeuronsCount, LayerInitializer)
        {
            _LayerFeedbacks = new double[InputsCount][,];
            _LastOutputs = new double[LayersCount][];
            _Output = new double[OutputsCount];

            for (var i = 0; i < InputsCount; i++)
            {
                var neurons_count = NeuronsCount[i];
                _LayerFeedbacks[i] = new double[neurons_count, neurons_count];
                _LastOutputs[i] = new double[neurons_count];
            }
        }

        /// <summary>Обработка данных сетью</summary>
        /// <param name="Input">Массив входа</param>
        /// <param name="Output">Массив выхода</param>
        public override void Process(double[] Input, double[] Output)
        {
            if (Input is null)
                throw new ArgumentNullException(nameof(Input));
            if (Output is null)
                throw new ArgumentNullException(nameof(Output));
            if (Input.Length != InputsCount)
                throw new ArgumentException($"Размер входного вектора ({Input.Length}) не равен количествоу входов сети ({InputsCount})", nameof(Input));
            if (Output.Length != OutputsCount)
                throw new ArgumentException($"Размер выходного вектора ({Output.Length}) не соответвтует количеству выходов сети ({OutputsCount})", nameof(Output));

            var layers = _Layers;                                       // Матрицы коэффициентов передачи слоёв
            var feedbacks = _LayerFeedbacks;                            // Матрица обратных связей
            var layers_count = layers.Length;                           // Количество слоёв
            var layer_activation = _Activations;                        // Активационные функции слоёв
            var layer_offsets = _Offsets;                               // Смещения слоёв
            var layer_offset_weights = _OffsetsWeights;                 // Весовые коэффициенты весов слоёв <= 0

            var outputs = _Outputs;
            var last_outputs = _LastOutputs;

            for (var layer_index = 0; layer_index < layers_count; layer_index++)
            {
                var w = layers[layer_index];
                var feedback_w = feedbacks[layer_index];
                var layer_outputs_count = w.GetLength(0);
                var prev_output = layer_index == 0                       // Если слой первый, то за выходы "предыдущего слоя"
                                  ? Input                                // принимаем входной вектор
                                  : outputs[layer_index - 1];            // иначе берём массив выходов предыдущего слоя

                var current_output = layer_index == layers_count - 1     // Если слой последний, то за выходы "следующего слоя"
                    ? Output                                             // Принимаем массив выходного вектора
                    : outputs[layer_index];                              // иначе берём массив текущего слоя

                var current_layer_offset = layer_offsets[layer_index];
                var current_layer_offset_weights = layer_offset_weights[layer_index];

                var last_current_output = last_outputs[layer_index];
                if (layer_index == layers_count - 1)
                    _Output.CopyTo(last_current_output, 0);
                else
                    current_output.CopyTo(last_current_output, 0);

                for (var output_index = 0; output_index < layer_outputs_count; output_index++)
                {
                    var output = current_layer_offset[output_index] * current_layer_offset_weights[output_index];
                    var inputs_count = w.GetLength(1);
                    for (var input_index = 0; input_index < inputs_count; input_index++)
                        output += w[output_index, input_index] * prev_output[input_index];

                    for (var feedback_output_index = 0; feedback_output_index < layer_outputs_count; feedback_output_index++)
                        output += feedback_w[output_index, feedback_output_index] * last_current_output[output_index];

                    current_output[output_index] = layer_activation[layer_index]?.Value(output) ?? Sigmoid.Activation(output);
                }
            }
            Output.CopyTo(_Output, 0);
        }

        #region Overrides of MultilayerPerceptron

        public double Teach(double[] Input, double[] Output, double[] Expected, double Rho = 0.2)
        {
            Process(Input, Output);

            var layers_count = LayersCount;
            var outputs_count = OutputsCount;
            var outputs = _Outputs;
            var last_outputs = _LastOutputs;
            var layers = _Layers;                                                          // Матрицы коэффициентов передачи слоёв
            var layers_feedbacks = _LayerFeedbacks;

            var layer_activation_inverse = _Activations;                                  // Производные активационных функций слоёв
            var errors = new double[layers_count][];                                       // Массив ошибок в слоях
            var output_layer_error = errors[layers_count - 1] = new double[outputs_count]; // Ошибка выходого слоя

            for (var output_index = 0; output_index < outputs_count; output_index++)
                output_layer_error[output_index] =
                    (Expected[output_index] - Output[output_index]) * (layer_activation_inverse[layers_count - 1]
                                                                           ?.DiffValue(Output[output_index])
                                                                       ?? Sigmoid.DiffActivation(Output[output_index]));

            // Проходим по всем слоям от выхода ко входу
            for (var layer_index = errors.Length - 1; layer_index >= 0; layer_index--)
            {
                var w = layers[layer_index];                // Текущий слой
                var feedback_w = layers_feedbacks[layer_index];
                var layer_inputs_count = w.GetLength(1);    // Количество входов текущего слоя
                var layer_outputs_count = w.GetLength(0);   // Количество выходов (нейронов) текущего слоя
                var error_level = errors[layer_index];      // Ошибка текущего слоя

                #region Обратное распространение ошибки

                // Если слой не последний, то пересчитываем сошибку текущего слоя на предыдущий
                if (layer_index > 0)
                {
                    // Количество выходов (нейронов) в предыдущем слое
                    var prev_layer_outputs_count = layers[layer_index - 1].GetLength(0);
                    // Создаём вектор значений для ошибки предыдущего слоя
                    var prev_error_level = errors[layer_index - 1] = new double[prev_layer_outputs_count];
                    // Извлекаем производную функции активации предыдущего слоя
                    var prev_layer_activation_inverse = layer_activation_inverse[layer_index - 1];
                    // Вектор выхода предыдущего слоя
                    var prev_layer_output = outputs[layer_index - 1];
                    // Для каждого нейрона (выхода) предыдущего слоя
                    for (var i = 0; i < prev_layer_outputs_count; i++)
                    {
                        // Вычисляем ошибку как...
                        var err = 0d;
                        // ...как сумму произведений коэффициентов передачи связей, ведущих к данному нейрону, умноженных на ошибку соответствующего нейрона текущего слоя
                        for (var j = 0; j < layer_outputs_count; j++) // j - номер связи с j-тым нейроном текущего слоя
                            err += error_level[j] * w[j, i];          // i - номер нейрона в предыдущем слое

                        // Значение на выходе расчитываемого нейрона в предыдущем слое
                        var output = prev_layer_output[i];
                        prev_error_level[i] = err * (prev_layer_activation_inverse?.DiffValue(output) ?? Sigmoid.DiffActivation(output));
                        // Ошибка по нейрону = суммарная взвешаная ошибка всех связей умнженная на значение производной функции активации для выхода нейрона
                    }
                }

                #endregion

                var layer_offsets = _Offsets;                   // Смещения слоёв
                var layer_offset_weights = _OffsetsWeights;     // Весовые коэффициенты весов слоёв <= 0
                var offset = layer_offsets[layer_index];                               // Для данного слоя - смещение нейронов
                var w_offset = layer_offset_weights[layer_index];                      //                  - веса смещений
                var layer_inputs = layer_index > 0 ? outputs[layer_index - 1] : Input;
                var layer_last_output = last_outputs[layer_index];
                // Для всех нейронов слоя корректируем коэффициенты их входных связей и весов смещений
                for (var neuron_index = 0; neuron_index < layer_outputs_count; neuron_index++)
                {
                    var error = error_level[neuron_index]; // Ошибка для нейрона в слое
                    // Корректируем вес смещения
                    w_offset[neuron_index] += Rho * error * offset[neuron_index] * w_offset[neuron_index];
                    // Для каждого входа нейрона корректируем вес связи
                    for (var input_index = 0; input_index < layer_inputs_count; input_index++)
                        w[neuron_index, input_index] += Rho * error * layer_inputs[input_index];
                    // Для каждого входа обратной связи нейрона корректируем вес связи
                    for (var output_index = 0; output_index < layer_outputs_count; output_index++)
                        feedback_w[neuron_index, output_index] += Rho * error * layer_last_output[output_index];
                }
            }

            var network_errors = 0d;
            for (var i = 0; i < Output.Length; i++)
            {
                var delta = Expected[i] - Output[i];
                network_errors += delta * delta;
            }

            return network_errors * 0.5;
        }

        #endregion
    }
}
