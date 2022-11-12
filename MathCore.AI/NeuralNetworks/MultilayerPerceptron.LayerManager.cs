using System.Collections;
using System.Globalization;

using MathCore.AI.NeuralNetworks.ActivationFunctions;

// ReSharper disable UnusedMember.Global

namespace MathCore.AI.NeuralNetworks;

public partial class MultilayerPerceptron
{
    /* --------------------------------------------------------------------------------------------- */

    #region Классы

    /// <summary>Менеджер слоя</summary>
    public sealed class LayerManager
    {
        /// <summary>Функции активации слоёв</summary>
        private readonly ActivationFunction[] _Activations;

        private readonly double[][] _Offsets;
        private readonly double[][] _OffsetWeights;
        private readonly double[][,] _Layers;
        private readonly double[][] _Outputs;

        /// <summary>Число входов слоя</summary>
        public int InputsCount => _Layers[LayerIndex].GetLength(1);

        /// <summary>Число выходов слоя</summary>
        public int OutputsCount => _Layers[LayerIndex].GetLength(0);

        /// <summary>Активационная функция слоя</summary>
        public ActivationFunction? Activation { get => _Activations[LayerIndex]; set => _Activations[LayerIndex] = value; }

        /// <summary>Смещения слоя</summary>
        public double[] Offsets => _Offsets[LayerIndex];

        /// <summary>Коэффициенты смещений слоя</summary>
        public double[] OffsetWeights => _OffsetWeights[LayerIndex];

        public double[] Outputs => _Outputs[LayerIndex];

        /// <summary>Матрица коэффициентов передачи входов слоя</summary>
        public double[,] Weights => _Layers[LayerIndex];

        /// <summary>Индекс слоя</summary>
        public int LayerIndex { get; }

        /// <summary>Предыдущий слой</summary>
        public LayerManager? PreviousLayer =>
            LayerIndex == 0 ? null : new LayerManager(LayerIndex - 1, _Layers, _Offsets, _OffsetWeights, _Outputs, _Activations);

        /// <summary>Следующий слой</summary>
        public LayerManager? NextLayer => LayerIndex == _Layers.Length - 1
            ? null
            : new LayerManager(LayerIndex + 1, _Layers, _Offsets, _OffsetWeights, _Outputs, _Activations);

        /// <summary>Инициализация нового установщика активационной функции слоя</summary>
        /// <param name="LayerIndex">номер слоя</param>
        /// <param name="Layers">Матрицы коэффициентов слоёв</param>
        /// <param name="Offsets">Смещения слоёв</param>
        /// <param name="OffsetWeights">Веса смещений слоёв</param>
        /// <param name="Outputs">Выходы слоёв</param>
        /// <param name="Activations">Массив всех активационных функций сети</param>
        internal LayerManager(
            int LayerIndex,
            double[][,] Layers,
            double[][] Offsets,
            double[][] OffsetWeights,
            double[][] Outputs,
            ActivationFunction[] Activations)
        {
            #region Проверка входных переменных

            if (Layers is null) throw new ArgumentNullException(nameof(Layers));

            if (LayerIndex < 0) throw new ArgumentOutOfRangeException(nameof(LayerIndex), "Индекс слоя меньше 0");
            if (LayerIndex >= Layers.Length)
                throw new ArgumentOutOfRangeException(nameof(LayerIndex), "Индекс больше, либо равен размеру массива активационных функций сети");

            #endregion

            this.LayerIndex = LayerIndex;
            _Activations    = Activations.NotNull();
            _Offsets        = Offsets.NotNull();
            _OffsetWeights  = OffsetWeights.NotNull();
            _Outputs        = Outputs.NotNull();
            _Layers         = Layers;
        }

        /// <summary>Инициализатор весов</summary>
        /// <param name="Neuron">Номер нейрона в слое</param>
        /// <param name="Input">Номер входа нейрона</param>
        /// <returns>Значение веса входа</returns>
        public delegate double LayerWeightsInitializer(int Neuron, int Input);

        /// <summary>Инициализация весов входов нейронов</summary>
        /// <param name="Initializer">Функция инициализации весов входов нейронов</param>
        public void SetWeights(LayerWeightsInitializer Initializer)
        {
            if (Initializer is null) throw new ArgumentNullException(nameof(Initializer));

            var weights       = Weights;
            var inputs_count  = InputsCount;
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
            var weights       = Weights;
            var inputs_count  = InputsCount;
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
            var weights       = Weights;
            var inputs_count  = InputsCount;
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
        public void SetOffsets(ICollection<double> OffsetValues) => (OffsetValues.NotNull()).CopyTo(Offsets, 0);

        /// <summary>Инициализатор смещений нейронов</summary>
        /// <param name="Neuron">Номер нейрона</param>
        /// <returns>Значение смещения для указанного нейрона</returns>
        public delegate double LayerOffsetsInitializer(int Neuron);

        /// <summary>Установка значений смещений нейронов</summary>
        /// <param name="Setter">Инициализатор смещений</param>
        public void SetOffsets(LayerOffsetsInitializer Setter)
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
        public void SetOffsetWeights(ICollection<double> WeightValues) => (WeightValues.NotNull()).CopyTo(OffsetWeights, 0);

        /// <summary>Установка значений весов смещений нейронов</summary>
        /// <param name="Setter">Инициализатор весов смещений</param>
        public void SetOffsetWeights(LayerOffsetsInitializer Setter)
        {
            if (Setter is null) throw new ArgumentNullException(nameof(Setter));

            var offset_weights = OffsetWeights;
            for (var i = 0; i < offset_weights.Length; i++)
                offset_weights[i] = Setter(i);
        }

        /// <summary>Установка значений выходов слоя</summary>
        /// <param name="OutputValues">Требуемые значения выходов слоя</param>
        public void SetOutputs(ICollection<double> OutputValues) => (OutputValues.NotNull()).CopyTo(Outputs, 0);

        /// <summary>Загрузка данных весовых коэффициентов слоя из CSV-файла</summary>
        /// <param name="FileName">Имя файла данных</param>
        /// <param name="Separator">Символ-разделитель данных</param>
        /// <param name="HeaderLinesCount">Количество строк заголовков</param>
        public void LoadCSV(string FileName, char Separator = ',', int HeaderLinesCount = 0)
        {
            var file  = new FileInfo(FileName.NotNull());
            var lines = file.ThrowIfNotFound().GetStringLines();

            if (HeaderLinesCount > 0)
                lines = lines.Skip(HeaderLinesCount);

            var w = lines
               .Where(line => line is { Length: > 0 })
               .Select(line => line!.Split(Separator).Select(S => double.Parse(S, NumberFormatInfo.InvariantInfo)).ToArray())
               .ToArray();

            var layer         = _Layers[LayerIndex];
            var inputs_count  = layer.GetLength(1);
            var neurons_count = layer.GetLength(0);
            for (var i = 0; i < neurons_count; i++)
                for (var j = 0; j < inputs_count; j++)
                    layer[i, j] = w[i][j];
        }

        #region Overrides of Object

        public override int GetHashCode()
        {
            const int hash_base = 0x18d;
            var       hash      = Consts.BigPrime_int;
            hash = unchecked(hash * hash_base ^ LayerIndex.GetHashCode());
            for (var i = 0; i < OffsetWeights.Length; i++)
                hash = unchecked(hash * hash_base ^ OffsetWeights[i].GetHashCode());
            for (var i = 0; i < Offsets.Length; i++)
                hash = unchecked(hash * hash_base ^ Offsets[i].GetHashCode());

            var outputs_count   = OutputsCount;
            var inputs_count    = InputsCount;
            var weights         = Weights;
            var double_equality = EqualityComparer<double>.Default;
            for (var neuron = 0; neuron < outputs_count; neuron++)
                for (var input = 0; input < inputs_count; input++)
                    hash = unchecked(hash * hash_base ^ double_equality.GetHashCode(weights[neuron, input]));

            return hash;
        }

        #endregion
    }

    /// <summary>Менеджер слоёв сети</summary>
    public sealed class LayersManager : IEnumerable<LayerManager>
    {
        /// <summary>Функции активации слоёв сети</summary>
        private readonly ActivationFunction[] _Activations;

        /// <summary>Смещения нейронов</summary>
        private readonly double[][] _Offsets;

        /// <summary>Весовые коэффициенты смещений нейронов</summary>
        private readonly double[][] _OffsetWeights;

        /// <summary>Массив матриц коэффициентов передачи сети</summary>
        private readonly double[][,] _Layers;

        /// <summary>Выходы сети</summary>
        private readonly double[][] _Outputs;

        /// <summary>Установщик активационной функции указанного слоя</summary>
        /// <param name="LayerIndex">Номер слоя функцию которого требуется установить</param>
        /// <returns>Установщик активационной функции слоя</returns>
        public LayerManager this[int LayerIndex] => new(LayerIndex, _Layers, _Offsets, _OffsetWeights, _Outputs, _Activations);

        /// <summary>Инициализация нового менеджера активационных функций сети</summary>
        /// <param name="Layers">Массив матриц передачи слоёв</param>
        /// <param name="Offsets">Массив смещений слоёв</param>
        /// <param name="OffsetWeights">Массив весовых коэффициентов смещений слоёв</param>
        /// <param name="Outputs">Выходы слоёв сети</param>
        /// <param name="Activations">Массив активационных функций сети</param>
        internal LayersManager(
            double[][,] Layers,
            double[][] Offsets,
            double[][] OffsetWeights,
            double[][] Outputs,
            ActivationFunction[] Activations)
        {
            _Activations   = Activations.NotNull();
            _Offsets       = Offsets.NotNull();
            _OffsetWeights = OffsetWeights.NotNull();
            _Outputs       = Outputs.NotNull();
            _Layers        = Layers.NotNull();
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