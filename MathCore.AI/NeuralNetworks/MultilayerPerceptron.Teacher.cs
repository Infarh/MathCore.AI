using System;
using MathCore.AI.NeuralNetworks.ActivationFunctions;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public partial class MultilayerPerceptron
    {
        [NotNull] public INetworkTeacher CreateTeacher() => new BackPropagationTeacher(this);

        private class BackPropagationTeacher : NetworkTeacher
        {
            [NotNull] private readonly MultilayerPerceptron _Network;
            [NotNull] private readonly double[][] _Errors;
            [NotNull] private readonly double[][] _State;
            [NotNull] private readonly double[][] _DeltaW;
            private double _LastError = double.PositiveInfinity;
            [NotNull] private readonly double[][,] _BestVariantW;
            [NotNull] private readonly double[][] _BestVariantOffsetW;

            public BackPropagationTeacher([NotNull] MultilayerPerceptron Network) : base(Network)
            {
                _Network = Network;
                var layers_count = _Network.LayersCount;
                _Errors = new double[layers_count][];
                _State = new double[layers_count][];
                _DeltaW = new double[layers_count][];
                _BestVariantW = new double[layers_count][,];
                _BestVariantOffsetW = new double[layers_count][];
                for (var i = 0; i < layers_count; i++)
                {
                    var neurons_count = _Network._Layers[i].GetLength(0);
                    _Errors[i] = new double[neurons_count];
                    _State[i] = new double[neurons_count];
                    _DeltaW[i] = new double[neurons_count];
                    _BestVariantW[i] = (double[,])Network._Layers[i].Clone();
                    _BestVariantOffsetW[i] = (double[])Network._OffsetsWeights[i].Clone();
                }
            }

            public override double Teach(double[] Input, double[] Output, double[] Expected, double Rho = 0.2D, double InertialFactor = 0D)
            {
                if (Input is null) throw new ArgumentNullException(nameof(Input));
                if (Output is null) throw new ArgumentNullException(nameof(Output));
                if (Expected is null) throw new ArgumentNullException(nameof(Expected));
                if (Expected.Length != Output.Length) throw new InvalidOperationException("Длина вектора ожидаемого результата не совпадает с длиной вектора результата сети");
                if (InertialFactor < 0 || InertialFactor >= 1) throw new ArgumentOutOfRangeException(nameof(InertialFactor), InertialFactor, "Коэффициент инерции должен быть больше, либо равен 0 и меньше 1");

                Rho = Math.Max(0, Math.Min(Rho, 1));

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

                var delta_w = InertialFactor.Equals(0d) ? null : _DeltaW;

                var errors = _Errors;                                       // Массив ошибок в слоях
                var output_layer_error = errors[layers_count - 1];          // Ошибка выходого слоя

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

                    // Если слой не последний, то пересчитываем сошибку текущего слоя на предыдущий
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

                            // Ошибка по нейрону = суммарная взвешаная ошибка всех связей умнженная на значение производной функции активации для выхода нейрона
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

                    var offset = layer_offsets[layer_index];                               // Для данного слоя - смещение нейронов
                    var w_offset = layer_offset_weights[layer_index];                      //                  - веса смещений
                    var layer_inputs = layer_index > 0 ? outputs[layer_index - 1] : Input;
                    var layer_delta_w = delta_w?[layer_index];
                    // Для всех нейронов слоя корректируем коэффициенты их входных связей и весов смещений
                    for (var neuron_index = 0; neuron_index < layer_outputs_count; neuron_index++)
                    {
                        var error = error_level[neuron_index]; // Ошибка для нейрона в слое
                                                               // Корректируем вес смещения
                        var neuron_delta_w = Rho * error * offset[neuron_index] * w_offset[neuron_index];
                        if (layer_delta_w != null)
                        {
                            neuron_delta_w = InertialFactor * layer_delta_w[neuron_index]
                                             + (1 - InertialFactor) * neuron_delta_w;
                            layer_delta_w[neuron_index] = neuron_delta_w;
                        }
                        w_offset[neuron_index] += neuron_delta_w;

                        // Для каждого входа нейрона корректируем вес связи
                        for (var input_index = 0; input_index < layer_inputs_count; input_index++)
                            w[neuron_index, input_index] += Rho * error * layer_inputs[input_index];
                    }
                }

                var network_error = GetError(Output, Expected);

                if (network_error >= _LastError) return network_error;
                _LastError = network_error;

                CopyArchitecture(layers, _BestVariantW, layer_offset_weights, _BestVariantOffsetW);

                return network_error;
            }

            private static void CopyArchitecture(
                double[][,] SourceW, double[][,] DestinationW,
                double[][] SourceOffsetW, double[][] DestinationOffsetW)
            {
                var layers_count = SourceW.Length;
                for (var i = 0; i < layers_count; i++)
                    Buffer.BlockCopy(SourceW[i], 0, DestinationW[i], 0, SourceW[i].GetLength(0) * SourceW[i].GetLength(1) * 8);

                for (var i = 0; i < SourceOffsetW.Length; i++)
                    SourceOffsetW[i].CopyTo(DestinationOffsetW[i], 0);
            }

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

            public override void SetBestVariant() => CopyArchitecture(_BestVariantW, _Network._Layers, _BestVariantOffsetW, _Network._OffsetsWeights);
        }
    }
}
