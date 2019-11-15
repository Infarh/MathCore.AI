using System;
using System.Collections.Generic;
using MathCore.AI.NeuralNetworks.ActivationFunctions;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public partial class MultilayerPerceptron
    {
        public INetworkTeacher CreateTeacher() => new BackPropagationTeacher(this);

        public TNetworkTeacher CreateTeacher<TNetworkTeacher>(Action<TNetworkTeacher> Configurator = null)
            where TNetworkTeacher : class, INetworkTeacher
        {
            var teacher = new BackPropagationTeacher(this) as TNetworkTeacher 
                          ?? throw new InvalidOperationException(
                              $"Учитель сети не может быть представлен в виде {typeof(TNetworkTeacher)}");
            Configurator?.Invoke(teacher);
            return teacher;
        }

        /// <summary>Объект-учитель, выполняющий обучение многослойной сети методом обратного распространения</summary>
        private class BackPropagationTeacher : NetworkTeacher, IBackPropagationTeacher
        {
            /// <summary>Обучаемая сеть</summary>
            [NotNull] private readonly MultilayerPerceptron _Network;

            /// <summary>Ошибки на выходах нейронов в слоях</summary>
            [NotNull] private readonly double[][] _Errors;

            /// <summary>Состояние входов нейронов (аргументы активационной функции)</summary>
            [NotNull] private readonly double[][] _State;

            /// <summary>Величина изменения веса связи нейрона с предыдущей итерации обучения</summary>
            [NotNull] private readonly double[][,] _DW;

            /// <summary>Величина изменения веса смещения нейрона с предыдущей итерации обучения</summary>
            [NotNull] private readonly double[][] _DWoffset;

            /// <summary>Предыдущая ошибка прямого распространения</summary>
            private double _LastError = double.PositiveInfinity;

            /// <summary>Лучший вариант весов входов в слоях</summary>
            [NotNull] private readonly double[][,] _BestVariantW;

            /// <summary>Лучший вариант весов смещений нейронов в слоях</summary>
            [NotNull] private readonly double[][] _BestVariantOffsetW;

            public double Rho { get; set; } = 0.2;

            public double InertialFactor { get; set; }

            public BackPropagationTeacher([NotNull] MultilayerPerceptron Network) : base(Network)
            {
                _Network = Network;
                var layers_count = _Network.LayersCount;
                _Errors = new double[layers_count][];
                _State = new double[layers_count][];
                _DW = new double[layers_count][,];
                _DWoffset = new double[layers_count][];
                _BestVariantW = new double[layers_count][,];
                _BestVariantOffsetW = new double[layers_count][];
                for (var i = 0; i < layers_count; i++)
                {
                    var (neurons_count, inputs_count) = _Network._Layers[i];
                    _Errors[i] = new double[neurons_count];
                    _State[i] = new double[neurons_count];
                    _DW[i] = new double[neurons_count, inputs_count];
                    _DWoffset[i] = new double[neurons_count];
                    _BestVariantW[i] = (double[,])Network._Layers[i].Clone();
                    _BestVariantOffsetW[i] = (double[])Network._OffsetsWeights[i].Clone();
                }
            }

            public override double Teach(double[] Input, double[] Output, double[] Expected)
            {
                if (Input is null) throw new ArgumentNullException(nameof(Input));
                if (Output is null) throw new ArgumentNullException(nameof(Output));
                if (Expected is null) throw new ArgumentNullException(nameof(Expected));
                if (Expected.Length != Output.Length) throw new InvalidOperationException("Длина вектора ожидаемого результата не совпадает с длиной вектора результата сети");

                var inertial_factor = InertialFactor;
                var rho = Math.Max(0, Math.Min(Rho, 1));

                var layers_count = _Network.LayersCount;
                var outputs_count = _Network.OutputsCount;

                var layers = _Network._Layers;                              // Матрицы коэффициентов передачи слоёв
                var outputs = _Network._Outputs;
                var layer_offsets = _Network._Offsets;                      // Смещения слоёв
                var layer_offset_weights = _Network._OffsetsWeights;        // Весовые коэффициенты весов слоёв <= 0
                var layer_activation = _Network._Activations;               // Производные активационных функций слоёв
                var state = _State;

                Process(
                    Input, Output,
                    layers,
                    layer_activation,
                    layer_offsets,
                    layer_offset_weights,
                    state,
                    outputs);

                var dw = inertial_factor.Equals(0d) ? null : _DW;
                var dw_offset = inertial_factor.Equals(0d) ? null : _DWoffset;

                var errors = _Errors;                                       // Массив ошибок в слоях
                var output_layer_error = errors[layers_count - 1];          // Ошибка выходного слоя

                for (var output_index = 0; output_index < outputs_count; output_index++)
                    output_layer_error[output_index] = (Expected[output_index] - Output[output_index])
                        * (layer_activation[layers_count - 1]?.DiffValue(Output[output_index]) ?? Sigmoid.DiffActivation(Output[output_index]));
                // Проходим по всем слоям от выхода ко входу
                for (var layer_index = errors.Length - 1; layer_index >= 0; layer_index--)
                {
                    var w = layers[layer_index];                // Текущий слой
                    var layer_inputs_count = w.GetLength(1);    // Количество входов текущего слоя
                    var layer_outputs_count = w.GetLength(0);   // Количество выходов (нейронов) текущего слоя
                    var error_level = errors[layer_index];      // Ошибка текущего слоя

                    #region Обратное распространение ошибки

                    // Если слой не последний, то пересчитываем ошибку текущего слоя на предыдущий
                    if (layer_index > 0)
                    {
                        // Количество выходов (нейронов) в предыдущем слое
                        var prev_layer_outputs_count = layers[layer_index - 1].GetLength(0);
                        // Создаём вектор значений для ошибки предыдущего слоя
                        var prev_layer_error = errors[layer_index - 1];
                        // Извлекаем производную функции активации предыдущего слоя
                        var prev_layer_activation = layer_activation[layer_index - 1];

                        var layer_state = state[layer_index];

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

                            // Ошибка по нейрону = суммарная взвешенная ошибка всех связей умноженная на значение производной функции активации для выхода нейрона
                            switch (prev_layer_activation)
                            {
                                case null:
                                    prev_layer_error[i] = err * Sigmoid.DiffActivation(prev_layer_output[i]);
                                    break;
                                case DiffSiplifiedActivationFunction activation:
                                    prev_layer_error[i] = err * activation.DiffFunc(prev_layer_output[i]);
                                    break;
                                default:
                                    prev_layer_error[i] = err * prev_layer_activation.DiffValue(layer_state[i]);
                                    break;
                            }
                        }
                    }

                    #endregion

                    var offset = layer_offsets[layer_index];           // Для данного слоя - смещение нейронов
                    var w_offset = layer_offset_weights[layer_index];  //                  - веса смещений
                    var layer_inputs = layer_index > 0 ? outputs[layer_index - 1] : Input;
                    var layer_dw = dw?[layer_index];
                    var layer_dw_offset = dw_offset?[layer_index];
                    // Для всех нейронов слоя корректируем коэффициенты их входных связей и весов смещений
                    for (var neuron = 0; neuron < layer_outputs_count; neuron++)
                    {
                        var error = error_level[neuron]; // Ошибка для нейрона в слое
                                                         // Корректируем вес смещения
                        var neuron_delta_w = rho * error * offset[neuron] * w_offset[neuron];
                        if (layer_dw_offset != null)
                        {
                            neuron_delta_w = inertial_factor * layer_dw_offset[neuron]
                                             + (1 - inertial_factor) * neuron_delta_w;
                            layer_dw_offset[neuron] = neuron_delta_w;
                        }
                        w_offset[neuron] += neuron_delta_w;

                        // Для каждого входа нейрона корректируем вес связи
                        for (var input = 0; input < layer_inputs_count; input++)
                        {
                            var neuron_dw = rho * error * layer_inputs[input];
                            if (layer_dw != null)
                            {
                                neuron_dw = inertial_factor * layer_dw[neuron, input]
                                            + (1 - inertial_factor) * neuron_dw;
                                layer_dw[neuron, input] = neuron_dw;
                            }
                            w[neuron, input] += neuron_dw;
                        }
                    }
                }

                var network_error = GetError(Output, Expected);

                if (network_error >= _LastError) return network_error;
                _LastError = network_error;

                CopyArchitecture(layers, _BestVariantW, layer_offset_weights, _BestVariantOffsetW);

                return network_error;
            }

            /// <summary>Скопировать архитектуру</summary>
            /// <param name="SourceW">Набор матриц коэффициентов источника операции копирования</param>
            /// <param name="DestinationW">Набор матриц коэффициентов приёмника операции копирования</param>
            /// <param name="SourceOffsetW">Набор весов коэффициентов смещения источника операции копирования</param>
            /// <param name="DestinationOffsetW">Набор весов коэффициентов смещения приёмника операции копирования</param>
            private static void CopyArchitecture(
                [NotNull] IReadOnlyList<double[,]> SourceW, 
                [NotNull] IReadOnlyList<double[,]> DestinationW,
                [NotNull] IReadOnlyList<double[]> SourceOffsetW, 
                [NotNull] IReadOnlyList<double[]> DestinationOffsetW)
            {
                var layers_count = SourceW.Count;
                for (var i = 0; i < layers_count; i++)
                    Buffer.BlockCopy(SourceW[i], 0, DestinationW[i], 0, SourceW[i].GetLength(0) * SourceW[i].GetLength(1) * 8);

                for (var i = 0; i < SourceOffsetW.Count; i++)
                    SourceOffsetW[i].CopyTo(DestinationOffsetW[i], 0);
            }

            /// <summary>Расчёт ошибки</summary>
            /// <param name="Output">Вектор результата вычисления прямого распространения</param>
            /// <param name="Expected">Вектор ожидаемого значения</param>
            /// <returns>Величина квадратичной ошибки</returns>
            private static double GetError([NotNull] double[] Output, [NotNull] double[] Expected)
            {
                var network_errors = 0d;
                for (var i = 0; i < Output.Length; i++)
                {
                    var delta = Expected[i] - Output[i];
                    network_errors += delta * delta;
                }

                return network_errors * 0.5;
            }

            /// <summary>Установить значение оптимальной архитектуры сети</summary>
            public override void SetBestVariant() => CopyArchitecture(_BestVariantW, _Network._Layers, _BestVariantOffsetW, _Network._OffsetsWeights);
        }
    }
}
