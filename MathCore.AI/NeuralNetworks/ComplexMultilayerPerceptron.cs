using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using MathCore.AI.NeuralNetworks.ActivationFunctions;
using MathCore.Annotations;
// ReSharper disable InconsistentNaming
// ReSharper disable UnusedMember.Global

namespace MathCore.AI.NeuralNetworks
{
    public class ComplexMultilayerPerceptron : IComplexNeuralNetwork
    {
        /* --------------------------------------------------------------------------------------------- */

        #region Классы

        /// <summary>Менеджер слоя</summary>
        public sealed class LayerManager
        {
            /// <summary>Функции активации слоёв</summary>
            [NotNull] private readonly ComplexActivationFunction[] _Activations;

            [NotNull] private readonly Complex[][] _Offsets;
            [NotNull] private readonly Complex[][] _OffsetWeights;
            [NotNull] private readonly Complex[][,] _Layers;
            [NotNull] private readonly Complex[][] _Outputs;

            /// <summary>Число входов слоя</summary>
            public int InputsCount => _Layers[LayerIndex].GetLength(1);
            /// <summary>Число выходов слоя</summary>
            public int OutputsCount => _Layers[LayerIndex].GetLength(0);

            /// <summary>Активационная функция слоя</summary>
            [CanBeNull] public ComplexActivationFunction Activation { get => _Activations[LayerIndex]; set => _Activations[LayerIndex] = value; }

            /// <summary>Смещения слоя</summary>
            [NotNull] public Complex[] Offsets => _Offsets[LayerIndex];

            /// <summary>Коэффициенты смещений слоя</summary>
            [NotNull] public Complex[] OffsetWeights => _OffsetWeights[LayerIndex];

            [NotNull] public Complex[] Outputs => _Outputs[LayerIndex];

            /// <summary>Матрица коэффициентов передачи входов слоя</summary>
            [NotNull] public Complex[,] Weights => _Layers[LayerIndex];

            /// <summary>Индекс слоя</summary>
            public int LayerIndex { get; }

            /// <summary>Инициализация нового установщика активационной функции слоя</summary>
            /// <param name="LayerIndex">номер слоя</param>
            /// <param name="Layers">Матрицы коэффициентов слоёв</param>
            /// <param name="Offsets">Смещения слоёв</param>
            /// <param name="OffsetWeights">Веса смещений слоёв</param>
            /// <param name="Outputs">Выходы слоёв</param>
            /// <param name="Activations">Массив всех активационных функций сети</param>
            internal LayerManager(int LayerIndex,
                [NotNull] Complex[][,] Layers,
                [NotNull] Complex[][] Offsets,
                [NotNull] Complex[][] OffsetWeights,
                [NotNull] Complex[][] Outputs,
                [NotNull] ComplexActivationFunction[] Activations)
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
                _Outputs = Outputs;
                _Layers = Layers;
            }

            /// <summary>Инициализатор весов</summary>
            /// <param name="Neuron">Номер нейрона в слое</param>
            /// <param name="Input">Номер входа нейрона</param>
            /// <returns>Значение веса входа</returns>
            public delegate Complex LayerWeightsInitializer(int Neuron, int Input);

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

            /// <summary>Установка значений смещений нейронов</summary>
            /// <param name="Offset">Требуемое значение смещений для всех нейронов слоя</param>
            public void SetOffsets(Complex Offset = default)
            {
                var offsets = Offsets;
                for (var i = 0; i < offsets.Length; i++)
                    offsets[i] = Offset;
            }

            /// <summary>Установка значений смещений нейронов</summary>
            /// <param name="OffsetValues">Требуемое значение смещений для всех нейронов слоя</param>
            public void SetOffsets([NotNull] ICollection<Complex> OffsetValues) => (OffsetValues ?? throw new ArgumentNullException(nameof(OffsetValues))).CopyTo(Offsets, 0);

            /// <summary>Инициализатор смещений нейронов</summary>
            /// <param name="Neuron">Номер нейрона</param>
            /// <returns>Значение смещения для указанного нейрона</returns>
            public delegate Complex LayerOffsetsInitializer(int Neuron);

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
            public void SetOffsetWeights(Complex Weight = default)
            {
                var offset_weights = OffsetWeights;
                for (var i = 0; i < offset_weights.Length; i++)
                    offset_weights[i] = Weight;
            }

            /// <summary>Установка требуемого значения весов смещений для нейронов слоя</summary>
            /// <param name="WeightValues">Требуемые значения весов смещений для всех нейронов слоя</param>
            public void SetOffsetWeights([NotNull] ICollection<Complex> WeightValues) => (WeightValues ?? throw new ArgumentNullException(nameof(WeightValues))).CopyTo(OffsetWeights, 0);

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
            public void SetOutputs([NotNull] ICollection<Complex> OutputValues) => (OutputValues ?? throw new ArgumentNullException(nameof(OutputValues))).CopyTo(Outputs, 0);

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
                    .Select(Line => Line.Split(Separator).Select(Complex.Parse).ToArray())
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
                var double_equality = EqualityComparer<Complex>.Default;
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
            [NotNull] private readonly ComplexActivationFunction[] _Activations;

            /// <summary>Смещения нейронов</summary>
            [NotNull] private readonly Complex[][] _Offsets;

            /// <summary>Весовые коэффициенты смещений нейронов</summary>
            [NotNull] private readonly Complex[][] _OffsetWeights;

            /// <summary>Массив матриц коэффициентов передачи сети</summary>
            [NotNull] private readonly Complex[][,] _Layers;

            /// <summary>Выходы сети</summary>
            [NotNull] private readonly Complex[][] _Outputs;

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
                [NotNull] Complex[][,] Layers,
                [NotNull] Complex[][] Offsets,
                [NotNull] Complex[][] OffsetWeights,
                [NotNull] Complex[][] Outputs,
                [NotNull] ComplexActivationFunction[] Activations)
            {
                _Activations = Activations ?? throw new ArgumentNullException(nameof(Activations));
                _Offsets = Offsets ?? throw new ArgumentNullException(nameof(Offsets));
                _OffsetWeights = OffsetWeights ?? throw new ArgumentNullException(nameof(OffsetWeights));
                _Outputs = Outputs;
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

        #region Поля 

        /// <summary>Матрицы коэффициентов передачи слоёв (номер строки - номер нейрона; номер столбца - номер входа нейрона; последний столбец - смещение нейрона)</summary>
        [NotNull] protected readonly Complex[][,] _Layers;

        /// <summary>Массив смещений нейронов в слоях (первый индекс - номер слоя; второй - номер нейрона в слое)</summary>
        [NotNull] protected readonly Complex[][] _Offsets;

        /// <summary>Массив весовых коэффициентов смещений нейронов в слоях (первый индекс - номер слоя; второй - номер нейрона в слое)</summary>
        [NotNull] protected readonly Complex[][] _OffsetsWeights;

        /// <summary>Массив выходов скрытых слоёв</summary>
        [NotNull] protected readonly Complex[][] _Outputs;

        /// <summary>Функции активации слоёв</summary>
        [NotNull] protected readonly ComplexActivationFunction[] _Activations;

        #endregion

        /* --------------------------------------------------------------------------------------------- */

        #region Свойства

        /// <summary>Слой</summary>
        [NotNull] public LayersManager Layer => new LayersManager(_Layers, _Offsets, _OffsetsWeights, _Outputs, _Activations);

        /// <summary>Входной слой</summary>
        [NotNull] public LayerManager LayerInput => Layer[0];

        /// <summary>Выходной слой</summary>
        [NotNull] public LayerManager LayerOutput => Layer[LayersCount - 1];

        /// <inheritdoc />
        public int InputsCount => _Layers[0].GetLength(1);

        /// <inheritdoc />
        public int OutputsCount => _Layers[_Layers.Length - 1].GetLength(0);

        /// <summary>Число слоёв</summary>
        public int LayersCount => _Layers.Length;

        /// <summary>Скрытые выходы</summary>
        [NotNull]
        public IReadOnlyList<Complex[]> HiddenOutputs => _Outputs;

        /// <summary>Индекс матриц весовых коэффициентов слоёв</summary>
        /// <param name="layer">Номер слоя</param>
        /// <returns>Матрица весовых коэффициентов слоёв</returns>
        [NotNull] public Complex[,] this[int layer] => _Layers[layer];

        /// <summary>Смещения слоёв</summary>
        [NotNull] public IReadOnlyList<Complex[]> Offests => _Offsets;

        /// <summary>Весовые коэффициенты смещений слоёв</summary>
        [NotNull] public IReadOnlyList<Complex[]> OffsetWeights => _OffsetsWeights;

        #endregion

        /* --------------------------------------------------------------------------------------------- */

        #region Конструкторы

        /// <summary>Инициализация новой многослойной нейронной сети</summary>
        /// <param name="Layers">Набор матриц коэффициентов передачи слоёв</param>
        public ComplexMultilayerPerceptron([NotNull] params Complex[][,] Layers)
        {
            _Layers = Layers ?? throw new ArgumentNullException(nameof(Layers));
            if (Layers.Length == 0) throw new ArgumentException("Число слоёв должно быть больше 0", nameof(Layers));

            var layers_count = Layers.Length;
            _Outputs = new Complex[layers_count - 1][];
            _Offsets = new Complex[layers_count][];
            _OffsetsWeights = new Complex[layers_count][];

            _Activations = new ComplexActivationFunction[layers_count];

            // Создаём структуры слоёв
            for (var i = 0; i < layers_count; i++)
            {
                // Количество входов слоя
                var inputs_count = _Layers[i].GetLength(1);

                // Проверяем - если слой не первый и количество выходов предыдущего слоя не совпадает с количеством входов текущего слоя, то это ошибка структуры сети
                if (i > 0 && _Layers[i - 1].GetLength(0) != inputs_count)
                    throw new FormatException($"Количество входов слоя {i} не равно количеству выходов слоя {i - 1}");

                //Количество выходов слоя (количество нейронов)
                var outputs_count = _Layers[i].GetLength(0);

                if (i < layers_count - 1)
                    _Outputs[i] = new Complex[outputs_count];                     // Выходы слоя
                _Offsets[i] = new Complex[outputs_count].Initialize(1);           // Создаём массив смещений нейронов слоя и инициализируем его единицами
                _OffsetsWeights[i] = new Complex[outputs_count].Initialize(1);    // Создаём массив коэффициентов смещений для слоя и инициализируем его единицами
            }
        }

        /// <summary>Инициализация матрицы весовых коэффициентов слоя</summary>
        /// <param name="LayerWeights">Матрица весовых коэффициентов слоя</param>
        /// <param name="LayerIndex">Индекс слоя</param>
        /// <param name="Initializer">Функция инициализации весовых коэффициентов слоя</param>
        private static void InitializeLayerWeightsMatrix(
            [NotNull] Complex[,] LayerWeights,
            int LayerIndex,
            [CanBeNull] NetworkCoefficientInitializer Initializer)
        {
            for (var i = 0; i < LayerWeights.GetLength(0); i++)
                for (var j = 0; j < LayerWeights.GetLength(1); j++)
                    LayerWeights[i, j] = Initializer?.Invoke(LayerIndex, i, j) ?? 1;
        }

        /// <summary>Создать массив матриц передачи слоёв</summary>
        /// <param name="InputsCount">Количество входов сети</param>
        /// <param name="NeuronsCount">Количество нейронов в слоях</param>
        /// <param name="Initialize">Функция инициализации весовых коэффициентов</param>
        /// <returns>Массив матриц коэффициентов передачи слоёв сети</returns>
        [NotNull]
        private static Complex[][,] CreateLayersMatrix(
            int InputsCount,
            [NotNull] IEnumerable<int> NeuronsCount,
            [CanBeNull] NetworkCoefficientInitializer Initialize)
        {
            var neurons_count = NeuronsCount.ToArray();
            var layers_count = neurons_count.Length;
            var weights = new Complex[layers_count][,];

            var w = new Complex[neurons_count[0], InputsCount];
            weights[0] = w;
            InitializeLayerWeightsMatrix(w, 0, Initialize);

            for (var layer = 1; layer < layers_count; layer++)
            {
                w = new Complex[neurons_count[layer], neurons_count[layer - 1]];
                weights[layer] = w;
                InitializeLayerWeightsMatrix(w, layer, Initialize);
            }

            return weights;
        }

        /// <summary>Инициализатор нейронной связи</summary>
        /// <param name="Layer">Номер слоя</param>
        /// <param name="Neuron">Номер нейрона в слое</param>
        /// <param name="Input">Номер входа нейрона</param>
        /// <returns>Коэффициент передачи входа нейрона</returns>
        public delegate Complex NetworkCoefficientInitializer(int Layer, int Neuron, int Input);

        /// <summary>Инициализатор слоя</summary>
        /// <param name="Layer">Менеджер инициализируемого слоя</param>
        public delegate void LayerInitializer([NotNull] LayerManager Layer);

        /// <summary>Инициализация новой многослойной нейронной сети</summary>
        /// <param name="InputsCount">Количество входов сети</param>
        /// <param name="NeuronsCount">Количество нейронов в слоях</param>
        /// <param name="Initialize">Функция инициализации коэффициентов матриц передачи слоёв</param>
        public ComplexMultilayerPerceptron(
            int InputsCount,
            [NotNull] IEnumerable<int> NeuronsCount,
            [CanBeNull] NetworkCoefficientInitializer Initialize = null)
            : this(CreateLayersMatrix(InputsCount, NeuronsCount, Initialize)) { }

        /// <summary>Инициализация новой многослойной нейронной сети</summary>
        /// <param name="InputsCount">Количество входов сети</param>
        /// <param name="NeuronsCount">Количество нейронов в слоях</param>
        /// <param name="rnd">Генератор случайных чисел для заполнения матриц коэффициентов передачи слоёв</param>
        public ComplexMultilayerPerceptron(
            int InputsCount,
            [NotNull] IEnumerable<int> NeuronsCount,
            [NotNull] Random rnd)
            : this(CreateLayersMatrix(InputsCount, NeuronsCount, (L, N, I) => rnd.NextDouble())) { }

        /// <summary>Инициализация новой многослойной нейронной сети</summary>
        /// <param name="InputsCount">Количество входов сети</param>
        /// <param name="NeuronsCount">Количество нейронов в слоях</param>
        /// <param name="Initializer">Функция инициализации слоёв сети</param>
        public ComplexMultilayerPerceptron(
            int InputsCount,
            [NotNull] IEnumerable<int> NeuronsCount,
            [CanBeNull] LayerInitializer Initializer)
            : this(InputsCount, NeuronsCount)
        {
            if (Initializer != null)
                foreach (var layer in Layer)
                    Initializer(layer);
        }

        #endregion

        /* --------------------------------------------------------------------------------------------- */

        #region Методы

        /// <summary>Обработка данных сетью</summary>
        /// <param name="Input">Массив входа</param>
        /// <param name="Output">Массив выхода</param>
        public virtual void Process(Complex[] Input, Complex[] Output)
        {
            if (Input is null) throw new ArgumentNullException(nameof(Input));
            if (Output is null) throw new ArgumentNullException(nameof(Output));
            if (Input.Length != InputsCount) throw new ArgumentException($"Размер входного вектора ({Input.Length}) не равен количество входов сети ({InputsCount})", nameof(Input));
            if (Output.Length != OutputsCount) throw new ArgumentException($"Размер выходного вектора ({Output.Length}) не соответствует количеству выходов сети ({OutputsCount})", nameof(Output));

            var layers = _Layers;                                       // Матрицы коэффициентов передачи слоёв
            var layers_count = layers.Length;                           // Количество слоёв
            var layer_activation = _Activations;                        // Активационные функции слоёв
            var layer_offsets = _Offsets;                               // Смещения слоёв
            var layer_offset_weights = _OffsetsWeights;                 // Весовые коэффициенты весов слоёв <= 0

            var outputs = _Outputs;

            for (var layer_index = 0; layer_index < layers_count; layer_index++)
            {
                var w = layers[layer_index];
                var layer_outputs_count = w.GetLength(0);
                var prev_output = layer_index == 0                       // Если слой первый, то за выходы "предыдущего слоя"
                                  ? Input                                // принимаем входной вектор
                                  : outputs[layer_index - 1];            // иначе берём массив выходов предыдущего слоя

                var current_output = layer_index == layers_count - 1     // Если слой последний, то за выходы "следующего слоя"
                    ? Output                                             // Принимаем массив выходного вектора
                    : outputs[layer_index];                              // иначе берём массив текущего слоя

                var current_layer_offset = layer_offsets[layer_index];
                var current_layer_offset_weights = layer_offset_weights[layer_index];

                for (var output_index = 0; output_index < layer_outputs_count; output_index++)
                {
                    var output = current_layer_offset[output_index] * current_layer_offset_weights[output_index];
                    var inputs_count = w.GetLength(1);
                    for (var input_index = 0; input_index < inputs_count; input_index++)
                        output += w[output_index, input_index] * prev_output[input_index];
                    current_output[output_index] = layer_activation[layer_index]?.Value(output) ?? ComplexExponent.Activation(output);
                }
            }
        }

        public virtual double Teach(
           [NotNull] Complex[] Input,
           [NotNull] Complex[] Output,
           [NotNull] Complex[] Expected,
           double Rho = 0.2)
        {
            Process(Input, Output);

            var layers_count = LayersCount;
            var outputs_count = OutputsCount;
            var outputs = _Outputs;
            var layers = _Layers;                                                          // Матрицы коэффициентов передачи слоёв

            var layer_activation_inverse = _Activations;                                  // Производные активационных функций слоёв
            var errors = new Complex[layers_count][];                                       // Массив ошибок в слоях
            var output_layer_error = errors[layers_count - 1] = new Complex[outputs_count]; // Ошибка выходного слоя

            for (var output_index = 0; output_index < outputs_count; output_index++)
                output_layer_error[output_index] =
                    (Expected[output_index] - Output[output_index]) * (layer_activation_inverse[layers_count - 1]
                                                                        ?.dValue(Output[output_index])
                                                                       ?? ComplexExponent.dActivation(Output[output_index]));
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
                    var prev_error_level = errors[layer_index - 1] = new Complex[prev_layer_outputs_count];
                    // Извлекаем производную функции активации предыдущего слоя
                    var prev_layer_activation_inverse = layer_activation_inverse[layer_index - 1];
                    // Вектор выхода предыдущего слоя
                    var prev_layer_output = outputs[layer_index - 1];
                    // Для каждого нейрона (выхода) предыдущего слоя
                    for (var i = 0; i < prev_layer_outputs_count; i++)
                    {
                        // Вычисляем ошибку как...
                        Complex err = default;
                        // ...как сумму произведений коэффициентов передачи связей, ведущих к данному нейрону, умноженных на ошибку соответствующего нейрона текущего слоя
                        for (var j = 0; j < layer_outputs_count; j++) // j - номер связи с j-тым нейроном текущего слоя
                            err += error_level[j] * w[j, i];          // i - номер нейрона в предыдущем слое

                        // Значение на выходе рассчитываемого нейрона в предыдущем слое
                        var output = prev_layer_output[i];
                        prev_error_level[i] = err * (prev_layer_activation_inverse?.dValue(output) ?? ComplexExponent.dActivation(output));
                        // Ошибка по нейрону = суммарная взвешенная ошибка всех связей умноженная на значение производной функции активации для выхода нейрона
                    }
                }

                #endregion

                var layer_offsets = _Offsets;                   // Смещения слоёв
                var layer_offset_weights = _OffsetsWeights;     // Весовые коэффициенты весов слоёв <= 0
                var offset = layer_offsets[layer_index];                               // Для данного слоя - смещение нейронов
                var w_offset = layer_offset_weights[layer_index];                      //                  - веса смещений
                var layer_inputs = layer_index > 0 ? outputs[layer_index - 1] : Input;
                // Для всех нейронов слоя корректируем коэффициенты их входных связей и весов смещений
                for (var neuron_index = 0; neuron_index < layer_outputs_count; neuron_index++)
                {
                    var error = error_level[neuron_index]; // Ошибка для нейрона в слое
                    // Корректируем вес смещения
                    w_offset[neuron_index] += Rho * error * offset[neuron_index] * w_offset[neuron_index];
                    // Для каждого входа нейрона корректируем вес связи
                    for (var input_index = 0; input_index < layer_inputs_count; input_index++)
                        w[neuron_index, input_index] += Rho * error * layer_inputs[input_index];
                }
            }

            var network_errors = 0d;
            // ReSharper disable once LoopCanBeConvertedToQuery
            for (var i = 0; i < Output.Length; i++)
            {
                var delta = Expected[i] - Output[i];
                network_errors += (delta * delta).Abs;
            }

            return network_errors * 0.5;
        }

        #endregion

        /* --------------------------------------------------------------------------------------------- */
    }
}