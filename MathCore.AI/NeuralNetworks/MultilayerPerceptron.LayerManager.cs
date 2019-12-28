using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using MathCore.AI.NeuralNetworks.ActivationFunctions;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public partial class MultilayerPerceptron
    {
        /* --------------------------------------------------------------------------------------------- */

        #region Классы

        /// <summary>Менеджер слоя</summary>
        public sealed class LayerManager
        {
            /// <summary>Функции активации слоёв</summary>
            [NotNull] private readonly ActivationFunction[] _Activations;

            [NotNull] private readonly double[][] _Offsets;
            [NotNull] private readonly double[][] _OffsetWeights;
            [NotNull] private readonly double[][,] _Layers;
            [NotNull] private readonly double[][] _Outputs;

            /// <summary>Число входов слоя</summary>
            public int InputsCount => _Layers[LayerIndex].GetLength(1);

            /// <summary>Число выходов слоя</summary>
            public int OutputsCount => _Layers[LayerIndex].GetLength(0);

            /// <summary>Активационная функция слоя</summary>
            [CanBeNull] public ActivationFunction Activation { get => _Activations[LayerIndex]; set => _Activations[LayerIndex] = value; }

            /// <summary>Смещения слоя</summary>
            [NotNull] public double[] Offsets => _Offsets[LayerIndex];

            /// <summary>Коэффициенты смещений слоя</summary>
            [NotNull] public double[] OffsetWeights => _OffsetWeights[LayerIndex];

            [NotNull] public double[] Outputs => _Outputs[LayerIndex];

            /// <summary>Матрица коэффициентов передачи входов слоя</summary>
            [NotNull] public double[,] Weights => _Layers[LayerIndex];

            /// <summary>Индекс слоя</summary>
            public int LayerIndex { get; }

            /// <summary>Предыдущий слой</summary>
            [CanBeNull] public LayerManager PreviousLayer => LayerIndex == 0 ? null : new LayerManager(LayerIndex - 1, _Layers, _Offsets, _OffsetWeights, _Outputs, _Activations);

            /// <summary>Следующий слой</summary>
            [CanBeNull] public LayerManager NextLayer => LayerIndex == _Layers.Length - 1 ? null : new LayerManager(LayerIndex + 1, _Layers, _Offsets, _OffsetWeights, _Outputs, _Activations);

            /// <summary>Инициализация нового установщика активационной функции слоя</summary>
            /// <param name="LayerIndex">номер слоя</param>
            /// <param name="Layers">Матрицы коэффициентов слоёв</param>
            /// <param name="Offsets">Смещения слоёв</param>
            /// <param name="OffsetWeights">Веса смещений слоёв</param>
            /// <param name="Outputs">Выходы слоёв</param>
            /// <param name="Activations">Массив всех активационных функций сети</param>
            internal LayerManager(int LayerIndex,
                [NotNull] double[][,] Layers,
                [NotNull] double[][] Offsets,
                [NotNull] double[][] OffsetWeights,
                [NotNull] double[][] Outputs,
                [NotNull] ActivationFunction[] Activations)
            {
                #region Проверка входных переменных

                if (Layers is null) throw new ArgumentNullException(nameof(Layers));

                if (LayerIndex < 0) throw new ArgumentOutOfRangeException(nameof(LayerIndex), "Индекс слоя меньше 0");
                if (LayerIndex >= Layers.Length) throw new ArgumentOutOfRangeException(nameof(LayerIndex), "Индекс больше, либо равен размеру массива активационных функций сети");

                #endregion

                this.LayerIndex = LayerIndex;
                _Activations = Activations ?? throw new ArgumentNullException(nameof(Activations));
                _Offsets = Offsets ?? throw new ArgumentNullException(nameof(Offsets));
                _OffsetWeights = OffsetWeights ?? throw new ArgumentNullException(nameof(OffsetWeights));
                _Outputs = Outputs ?? throw new ArgumentNullException(nameof(Outputs));
                _Layers = Layers;
            }

            /// <summary>Инициализатор весов</summary>
            /// <param name="Neuron">Номер нейрона в слое</param>
            /// <param name="Input">Номер входа нейрона</param>
            /// <returns>Значение веса входа</returns>
            public delegate double LayerWeightsInitializer(int Neuron, int Input);

            /// <summary>Инициализация весов входов нейронов</summary>
            /// <param name="Initializer">Функция инициализации весов входов нейронов</param>
            public void SetWeights([NotNull] LayerWeightsInitializer Initializer)
            {
                if (Initializer is null) throw new ArgumentNullException(nameof(Initializer));
                var weights = Weights;
                var inputs_count = InputsCount;
                var neurons_count = OutputsCount;
                for (var neuron = 0; neuron < neurons_count; neuron++)
                    for (var input = 0; input < inputs_count; input++)
                        weights[neuron, input] = Initializer(neuron, input);
            }

            /// <summary>Инициализация весовых коэффициентов слоя случайными числами с равномерным распределением</summary>
            /// <param name="Sigma">Среднеквадратичное отклонение</param>
            /// <param name="Mu">Математическое ожидание</param>
            public void SetWeightsUniform(double Sigma = 1, double Mu = 0)
            {
                var weights = Weights;
                var inputs_count = InputsCount;
                var neurons_count = OutputsCount;

                if (Sigma.Equals(0d))
                {
                    for (var neuron = 0; neuron < neurons_count; neuron++)
                        for (var input = 0; input < inputs_count; input++)
                            weights[neuron, input] = Mu;
                    return;
                }

                var rnd = new Random();

                for (var neuron = 0; neuron < neurons_count; neuron++)
                    for (var input = 0; input < inputs_count; input++)
                        weights[neuron, input] = (rnd.NextDouble() - 0.5) * Sigma + Mu;
            }

            /// <summary>Инициализация весовых коэффициентов слоя случайными числами с нормальным распределением</summary>
            /// <param name="Sigma">Среднеквадратичное отклонение</param>
            /// <param name="Mu">Математическое ожидание</param>
            public void SetWeightsNormal(double Sigma = 1, double Mu = 0)
            {
                var weights = Weights;
                var inputs_count = InputsCount;
                var neurons_count = OutputsCount;

                if (Sigma.Equals(0d))
                {
                    for (var neuron = 0; neuron < neurons_count; neuron++)
                        for (var input = 0; input < inputs_count; input++)
                            weights[neuron, input] = Mu;
                    return;
                }

                var rnd = new Random();

                for (var neuron = 0; neuron < neurons_count; neuron++)
                    for (var input = 0; input < inputs_count; input++)
                        weights[neuron, input] = rnd.NextNormal(Sigma, Mu);
            }

            /// <summary>Установка значений смещений нейронов</summary>
            /// <param name="Offset">Требуемое значение смещений для всех нейронов слоя</param>
            public void SetOffsets(double Offset = 0)
            {
                var offsets = Offsets;
                for (var i = 0; i < offsets.Length; i++)
                    offsets[i] = Offset;
            }

            /// <summary>Установка значений смещений нейронов</summary>
            /// <param name="OffsetValues">Требуемое значение смещений для всех нейронов слоя</param>
            public void SetOffsets([NotNull] ICollection<double> OffsetValues) => (OffsetValues ?? throw new ArgumentNullException(nameof(OffsetValues))).CopyTo(Offsets, 0);

            /// <summary>Инициализатор смещений нейронов</summary>
            /// <param name="Neuron">Номер нейрона</param>
            /// <returns>Значение смещения для указанного нейрона</returns>
            public delegate double LayerOffsetsInitializer(int Neuron);

            /// <summary>Установка значений смещений нейронов</summary>
            /// <param name="Setter">Инициализатор смещений</param>
            public void SetOffsets([NotNull] LayerOffsetsInitializer Setter)
            {
                if (Setter is null) throw new ArgumentNullException(nameof(Setter));
                var offsets = Offsets;
                for (var i = 0; i < offsets.Length; i++)
                    offsets[i] = Setter(i);
            }

            /// <summary>Установка требуемого значения весов смещений для нейронов слоя</summary>
            /// <param name="Weight">Требуемые значения весов смещений для всех нейронов слоя</param>
            public void SetOffsetWeights(double Weight = 0)
            {
                var offset_weights = OffsetWeights;
                for (var i = 0; i < offset_weights.Length; i++)
                    offset_weights[i] = Weight;
            }

            /// <summary>Установка требуемого значения весов смещений для нейронов слоя</summary>
            /// <param name="WeightValues">Требуемые значения весов смещений для всех нейронов слоя</param>
            public void SetOffsetWeights([NotNull] ICollection<double> WeightValues) => (WeightValues ?? throw new ArgumentNullException(nameof(WeightValues))).CopyTo(OffsetWeights, 0);

            /// <summary>Установка значений весов смещений нейронов</summary>
            /// <param name="Setter">Инициализатор весов смещений</param>
            public void SetOffsetWeights([NotNull] LayerOffsetsInitializer Setter)
            {
                if (Setter is null) throw new ArgumentNullException(nameof(Setter));
                var offset_weights = OffsetWeights;
                for (var i = 0; i < offset_weights.Length; i++)
                    offset_weights[i] = Setter(i);
            }

            /// <summary>Установка значений выходов слоя</summary>
            /// <param name="OutputValues">Требуемые значения выходов слоя</param>
            public void SetOutputs([NotNull] ICollection<double> OutputValues) => (OutputValues ?? throw new ArgumentNullException(nameof(OutputValues))).CopyTo(Outputs, 0);

            /// <summary>Загрузка данных весовых коэффициентов слоя из CSV-файла</summary>
            /// <param name="FileName">Имя файла данных</param>
            /// <param name="Separator">Символ-разделитель данных</param>
            /// <param name="HeaderLinesCount">Количество строк заголовков</param>
            public void LoadCSV([NotNull] string FileName, char Separator = ',', int HeaderLinesCount = 0)
            {
                if (FileName is null) throw new ArgumentNullException(nameof(FileName));
                var file = new FileInfo(FileName);
                if (!file.Exists) throw new FileNotFoundException($"Файл ({FileName}) с данными коэффициентов слоя не найден", FileName);
                var lines = file.GetStringLines();
                if (HeaderLinesCount > 0)
                    lines = lines.Skip(HeaderLinesCount);
                var w = lines
                    .Where(line => !string.IsNullOrEmpty(line))
                    .Select(Line => Line.Split(Separator).Select(S => double.Parse(S, NumberFormatInfo.InvariantInfo)).ToArray())
                    .ToArray();
                var layer = _Layers[LayerIndex];
                var inputs_count = layer.GetLength(1);
                var neurons_count = layer.GetLength(0);
                for (var i = 0; i < neurons_count; i++)
                    for (var j = 0; j < inputs_count; j++)
                        layer[i, j] = w[i][j];
            }

            #region Overrides of Object

            public override int GetHashCode()
            {
                const int hash_base = 0x18d;
                var hash = Consts.BigPrime_int;
                unchecked
                {
                    hash ^= LayerIndex;
                    hash ^= OffsetWeights.GetComplexHashCode() * hash_base;
                    hash ^= Offsets.GetComplexHashCode() * hash_base;

                }
                var outputs_count = OutputsCount;
                var inputs_count = InputsCount;
                var weights = Weights;
                var double_equality = EqualityComparer<double>.Default;
                for (var neuron = 0; neuron < outputs_count; neuron++)
                    for (var input = 0; input < inputs_count; input++)
                        unchecked
                        {
                            hash ^= double_equality.GetHashCode(weights[neuron, input]) * hash_base;
                        }
                return hash;

            }

            #endregion
        }

        /// <summary>Менеджер слоёв сети</summary>
        public sealed class LayersManager : IEnumerable<LayerManager>
        {
            /// <summary>Функции активации слоёв сети</summary>
            [NotNull] private readonly ActivationFunction[] _Activations;

            /// <summary>Смещения нейронов</summary>
            [NotNull] private readonly double[][] _Offsets;

            /// <summary>Весовые коэффициенты смещений нейронов</summary>
            [NotNull] private readonly double[][] _OffsetWeights;

            /// <summary>Массив матриц коэффициентов передачи сети</summary>
            [NotNull] private readonly double[][,] _Layers;

            /// <summary>Выходы сети</summary>
            [NotNull] private readonly double[][] _Outputs;

            /// <summary>Установщик активационной функции указанного слоя</summary>
            /// <param name="LayerIndex">Номер слоя функцию которого требуется установить</param>
            /// <returns>Установщик активационной функции слоя</returns>
            [NotNull]
            public LayerManager this[int LayerIndex] => new LayerManager(LayerIndex, _Layers, _Offsets, _OffsetWeights, _Outputs, _Activations);

            /// <summary>Инициализация нового менеджера активационных функций сети</summary>
            /// <param name="Layers">Массив матриц передачи слоёв</param>
            /// <param name="Offsets">Массив смещений слоёв</param>
            /// <param name="OffsetWeights">Массив весовых коэффициентов смещений слоёв</param>
            /// <param name="Outputs">Выходы слоёв сети</param>
            /// <param name="Activations">Массив активационных функций сети</param>
            internal LayersManager(
                [NotNull] double[][,] Layers,
                [NotNull] double[][] Offsets,
                [NotNull] double[][] OffsetWeights,
                [NotNull] double[][] Outputs,
                [NotNull] ActivationFunction[] Activations)
            {
                _Activations = Activations ?? throw new ArgumentNullException(nameof(Activations));
                _Offsets = Offsets ?? throw new ArgumentNullException(nameof(Offsets));
                _OffsetWeights = OffsetWeights ?? throw new ArgumentNullException(nameof(OffsetWeights));
                _Outputs = Outputs ?? throw new ArgumentNullException(nameof(Outputs));
                _Layers = Layers ?? throw new ArgumentNullException(nameof(Layers));
            }

            #region IEnumerator<LayerManager>

            IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

            public IEnumerator<LayerManager> GetEnumerator()
            {
                for (var i = 0; i < _Layers.Length; i++)
                    yield return this[i];
            }

            #endregion
        }

        #endregion

        /* --------------------------------------------------------------------------------------------- */
    }
}
