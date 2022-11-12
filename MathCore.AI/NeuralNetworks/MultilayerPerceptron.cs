using MathCore.AI.NeuralNetworks.ActivationFunctions;
using System.Globalization;
using System.Text;
using System.Xml;
// ReSharper disable UnusedMember.Global
// ReSharper disable MemberCanBePrivate.Global
namespace MathCore.AI.NeuralNetworks;

/// <summary>Многослойная полносвязная нейронная сеть прямого распространения</summary>
public partial class MultilayerPerceptron : ITeachableNeuralNetwork, IEquatable<MultilayerPerceptron>
{
    /* --------------------------------------------------------------------------------------------- */

    #region Поля 

    /// <summary>Матрицы коэффициентов передачи слоёв (номер строки - номер нейрона; номер столбца - номер входа нейрона; последний столбец - смещение нейрона)</summary>
    protected readonly double[][,] _Layers;

    /// <summary>Массив смещений нейронов в слоях (первый индекс - номер слоя; второй - номер нейрона в слое)</summary>
    protected readonly double[][] _Offsets;

    /// <summary>Массив весовых коэффициентов смещений нейронов в слоях (первый индекс - номер слоя; второй - номер нейрона в слое)</summary>
    protected readonly double[][] _OffsetsWeights;

    /// <summary>Массив выходов скрытых слоёв</summary>
    protected readonly double[][] _Outputs;

    /// <summary>Функции активации слоёв</summary>
    protected readonly ActivationFunction[] _Activations;

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Свойства

    /// <summary>Слой</summary>
    public LayersManager Layer => new(_Layers, _Offsets, _OffsetsWeights, _Outputs, _Activations);

    /// <summary>Входной слой</summary>
    public LayerManager LayerInput => Layer[0];

    /// <summary>Выходной слой</summary>
    public LayerManager LayerOutput => Layer[LayersCount - 1];

    /// <inheritdoc />
    public int InputsCount => _Layers[0].GetLength(1);

    /// <inheritdoc />
    public int OutputsCount => _Layers[_Layers.Length - 1].GetLength(0);

    /// <summary>Число слоёв</summary>
    public int LayersCount => _Layers.Length;

    /// <summary>Скрытые выходы</summary>
    public IReadOnlyList<double[]> HiddenOutputs => _Outputs;

    /// <summary>Индекс матриц весовых коэффициентов слоёв</summary>
    /// <param name="layer">Номер слоя</param>
    /// <returns>Матрица весовых коэффициентов слоёв</returns>
    public ref readonly double[,] this[int layer] => ref _Layers[layer];

    /// <summary>Смещения слоёв</summary>
    public IReadOnlyList<double[]> Offests => _Offsets;

    /// <summary>Весовые коэффициенты смещений слоёв</summary>
    public IReadOnlyList<double[]> OffsetWeights => _OffsetsWeights;

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Конструкторы

    /// <summary>Инициализация новой многослойной нейронной сети</summary>
    /// <param name="Layers">Набор матриц коэффициентов передачи слоёв</param>
    public MultilayerPerceptron(params double[][,] Layers)
    {
        _Layers = Layers.NotNull();
        if (Layers.Length == 0) throw new ArgumentException("Число слоёв должно быть больше 0", nameof(Layers));

        var layers_count = Layers.Length;
        _Outputs        = new double[layers_count - 1][];
        _Offsets        = new double[layers_count][];
        _OffsetsWeights = new double[layers_count][];

        _Activations = new ActivationFunction[layers_count];

        // Создаём структуры слоёв
        for (var layer_index = 0; layer_index < layers_count; layer_index++)
        {
            // Количество входов слоя
            var inputs_count = _Layers[layer_index].GetLength(1);

            // Проверяем - если слой не первый и количество выходов предыдущего слоя не совпадает с количеством входов текущего слоя, то это ошибка структуры сети
            if (layer_index > 0 && _Layers[layer_index - 1].GetLength(0) != inputs_count)
                throw new FormatException($"Количество входов слоя {layer_index} не равно количеству выходов слоя {layer_index - 1}");

            //Количество выходов слоя (количество нейронов)
            var outputs_count = _Layers[layer_index].GetLength(0);

            if (layer_index < layers_count - 1)
                _Outputs[layer_index] = new double[outputs_count];                  // Выходы слоя
            _Offsets[layer_index] = new double[outputs_count].Initialize(1);        // Создаём массив смещений нейронов слоя и инициализируем его единицами
            _OffsetsWeights[layer_index] = new double[outputs_count].Initialize(1); // Создаём массив коэффициентов смещений для слоя и инициализируем его единицами
        }
    }

    /// <summary>Инициализация матрицы весовых коэффициентов слоя</summary>
    /// <param name="LayerWeights">Матрица весовых коэффициентов слоя</param>
    /// <param name="LayerIndex">Индекс слоя</param>
    /// <param name="Initializer">Функция инициализации весовых коэффициентов слоя</param>
    private static void InitializeLayerWeightsMatrix(
        double[,] LayerWeights,
        int LayerIndex,
        NetworkCoefficientInitializer Initializer)
    {
        for (var i = 0; i < LayerWeights.GetLength(0); i++)
            for (var j = 0; j < LayerWeights.GetLength(1); j++)
                LayerWeights[i, j] = Initializer.Invoke(LayerIndex, i, j);
    }

    /// <summary>Создать массив матриц передачи слоёв</summary>
    /// <param name="InputsCount">Количество входов сети</param>
    /// <param name="NeuronsCount">Количество нейронов в слоях</param>
    /// <param name="Initialize">Функция инициализации весовых коэффициентов</param>
    /// <returns>Массив матриц коэффициентов передачи слоёв сети</returns>
    private static double[][,] CreateLayersMatrix(
        int InputsCount,
        IEnumerable<int> NeuronsCount,
        NetworkCoefficientInitializer Initialize)
    {
        var neurons_count = NeuronsCount.ToArray();
        var layers_count  = neurons_count.Length;
        var weights       = new double[layers_count][,];

        var w = new double[neurons_count[0], InputsCount];
        weights[0] = w;
        InitializeLayerWeightsMatrix(w, 0, Initialize);

        for (var layer = 1; layer < layers_count; layer++)
        {
            w              = new double[neurons_count[layer], neurons_count[layer - 1]];
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
    public delegate double NetworkCoefficientInitializer(int Layer, int Neuron, int Input);

    /// <summary>Инициализатор слоя</summary><param name="Layer">Менеджер инициализируемого слоя</param>
    public delegate void LayerInitializer(LayerManager Layer);

    private static NetworkCoefficientInitializer GetStandardRandomInitializer()
    {
        var rnd = new Random();
        return (_, _, _) => rnd.NextDouble() - 0.5;
    }

    /// <summary>Проверка массива числа входов</summary>
    /// <param name="Counts">Проверяемый массив входов</param>
    /// <returns>Возвращает проверяемый массив входов</returns>
    /// <exception cref="ArgumentNullException">если <paramref name="Counts"/> == <see langword="null"/></exception>
    /// <exception cref="ArgumentException">если длина <paramref name="Counts"/> == 0</exception>
    private static int[] CheckNeuronsCounts(int[] Counts)
    {
        if (Counts is null) throw new ArgumentNullException(nameof(Counts));
        if(Counts.Length == 0) throw new ArgumentException("Длина массива не может быть равен 0", nameof(Counts));
        return Counts;
    }

    /// <summary>Инициализация новой многослойной нейронной сети</summary>
    /// <param name="Counts">Число входов и число нейронов (выходов) на соответствующих слоях</param>
    public MultilayerPerceptron(params int[] Counts) : this(CheckNeuronsCounts(Counts)[0], Counts.Skip(1).ToArray()) { }

    /// <summary>Инициализация новой многослойной нейронной сети</summary>
    /// <param name="InputsCount">Количество входов сети</param>
    /// <param name="NeuronsCount">Количество нейронов в слоях</param>
    /// <param name="Initialize">Функция инициализации коэффициентов матриц передачи слоёв</param>
    public MultilayerPerceptron(
        int InputsCount,
        IEnumerable<int> NeuronsCount,
        NetworkCoefficientInitializer? Initialize = null)
        : this(CreateLayersMatrix(InputsCount, NeuronsCount, Initialize ?? GetStandardRandomInitializer())) { }

    /// <summary>Инициализация новой многослойной нейронной сети</summary>
    /// <param name="InputsCount">Количество входов сети</param>
    /// <param name="NeuronsCount">Количество нейронов в слоях</param>
    /// <param name="rnd">Генератор случайных чисел для заполнения матриц коэффициентов передачи слоёв</param>
    public MultilayerPerceptron(
        int InputsCount,
        IEnumerable<int> NeuronsCount,
        Random rnd)
        : this(CreateLayersMatrix(InputsCount, NeuronsCount, (_, _, _) => rnd.NextDouble() - 0.5)) { }

    /// <summary>Инициализация новой многослойной нейронной сети</summary>
    /// <param name="InputsCount">Количество входов сети</param>
    /// <param name="NeuronsCount">Количество нейронов в слоях</param>
    /// <param name="Initializer">Функция инициализации слоёв сети</param>
    public MultilayerPerceptron(
        int InputsCount,
        IEnumerable<int> NeuronsCount,
        LayerInitializer? Initializer)
        : this(InputsCount, NeuronsCount)
    {
        if (Initializer is null) return;
        foreach (var layer in Layer)
            Initializer(layer);
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Методы

    /// <summary>Обработка данных сетью</summary>
    /// <param name="Input">Массив входа</param>
    /// <param name="Output">Массив выхода</param>
    public virtual void Process(Span<double> Input, Span<double> Output) => Process(Input, Output, _Layers, _Activations, _Offsets, _OffsetsWeights, null, _Outputs);

    /// <summary>Обработка данных сетью</summary>
    /// <param name="Input">Массив входа</param>
    /// <param name="Output">Массив выхода</param>
    /// <param name="Layers">Массив матриц коэффициентов передачи слоёв</param>
    /// <param name="Activations">Массив активационных функций слоёв (если функция не задана, используется Сигмоид)</param>
    /// <param name="Offsets">Массив векторов смещений</param>
    /// <param name="OffsetsWeights">Массив векторов весовых коэффициентов смещений</param>
    /// <param name="State">
    /// Массив состояний (входов функций активации) слоёв.
    /// Может применяться в процессе обучения сети.
    /// Достаточно создать массив длиной, равной количеству слоёв с пустыми элементами.
    /// </param>
    /// <param name="Outputs">Массив векторов выходных значений слоёв</param>
    private static void Process(
        Span<double> Input,
        Span<double> Output,
        double[][,] Layers,
        ActivationFunction[] Activations,
        double[][] Offsets,
        double[][] OffsetsWeights,
        double[][]? State,
        double[][] Outputs
    )
    {
        if (Layers is null) throw new ArgumentNullException(nameof(Layers));
        if (Activations is null) throw new ArgumentNullException(nameof(Activations));
        if (Offsets is null) throw new ArgumentNullException(nameof(Offsets));
        if (OffsetsWeights is null) throw new ArgumentNullException(nameof(OffsetsWeights));
        if (Outputs is null) throw new ArgumentNullException(nameof(Outputs));

        if (Input.Length != Layers[0].GetLength(1)) throw new ArgumentException($"Размер входного вектора ({Input.Length}) не равен количествоу входов сети ({Layers[0].GetLength(1)})", nameof(Input));
        if (Output.Length != Layers[Layers.Length - 1].GetLength(0)) throw new ArgumentException($"Размер выходного вектора ({Output.Length}) не соответвтует количеству выходов сети ({Layers[Layers.Length - 1].GetLength(0)})", nameof(Output));
        if (Activations.Length != Layers.Length) throw new InvalidOperationException("Размер массива функций активации не соответствует количеству слоёв сети");

        var layer                = Layers;         // Матрицы коэффициентов передачи слоёв
        var layers_count         = layer.Length;   // Количество слоёв
        var layer_activation     = Activations;    // Активационные функции слоёв
        var layer_offsets        = Offsets;        // Смещения слоёв
        var layer_offset_weights = OffsetsWeights; // Весовые коэффициенты весов слоёв <= 0

        var state   = State;
        var outputs = Outputs;

        for (var layer_index = 0; layer_index < layers_count; layer_index++)
        {
            // Определяем матрицу слоя W
            var current_layer_weights = layer[layer_index];
            // Определяем вектор входа X
            var prev_layer_output = layer_index == 0 // Если слой первый, то за выходы "предыдущего слоя"
                ? Input                              // принимаем входной вектор
                : outputs[layer_index - 1];          // иначе берём массив выходов предыдущего слоя

            // Определяем вектор входа следующего слоя X_next         
            var current_output = layer_index == layers_count - 1   // Если слой последний, то за выходы "следующего слоя"
                ? Output                                           // Принимаем массив выходного вектора
                : outputs[layer_index]                             // иначе берём массив текущего слоя
                ?? new double[current_layer_weights.GetLength(0)]; // Если выходного вектора нет, то создаём его!

            // Определяем вектор входа функции активации Net
            double[]? current_state          = null;
            if (state != null) current_state = state[layer_index] ?? new double[current_output.Length];

            // Определяем вектор смещения O (Offset)
            var current_layer_offset = layer_offsets[layer_index];
            // Определяем вектор весов смещения Wo (Weight of offset)
            var current_layer_offset_weights = layer_offset_weights[layer_index];

            var current_layer_activation = layer_activation[layer_index];

            ProcessLayer(
                current_layer_weights,
                current_layer_offset,
                current_layer_offset_weights,
                prev_layer_output,
                current_output, current_layer_activation, current_state);
        }
    }

    /// <summary>Метод обработки одного слоя</summary>
    /// <param name="LayerWeights">Матрица коэффициентов передачи</param>
    /// <param name="Offset">Вектор смещений нейронов</param>
    /// <param name="OffsetWeight">Вектор весовых коэффициентов</param>
    /// <param name="Input">Вектор входного воздействия</param>
    /// <param name="Output">Вектор выходных значений нейронов</param>
    /// <param name="Activation">Активационная функция слоя (если не задана, то используется Сигмоид)</param>
    /// <param name="State">Вектор входа функции активации</param>
    private static void ProcessLayer(
        double[,] LayerWeights,
        double[] Offset,
        double[] OffsetWeight,
        Span<double> Input,
        Span<double> Output,
        ActivationFunction? Activation = null,
        double[]? State = null)
    {
        // Вычисляем X_next = f(Net = W * X + Wo*O)
        var layer_outputs_count = LayerWeights.GetLength(0);
        var layer_inputs_count  = LayerWeights.GetLength(1);
        for (var output_index = 0; output_index < layer_outputs_count; output_index++)
        {
            var output = Offset[output_index] * OffsetWeight[output_index];
            for (var input_index = 0; input_index < layer_inputs_count; input_index++)
                output += LayerWeights[output_index, input_index] * Input[input_index];
            if (State != null) State[output_index] = output;
            Output[output_index] = Activation?.Value(output) ?? Sigmoid.Activation(output);
        }
    }

    #endregion

    #region Сохранение/чтение xml

    #region Сохранение

    private const string __NumberFormat = "G17";

    /// <summary>Сохранить структуру сети в файл XMl</summary>
    /// <param name="FileName">Имя файла для сохранения данных</param>
    public void SaveXmlTo(string FileName)
    {
        using var file = File.CreateText(FileName.NotNull());
        SaveXmlTo(file);
    }

    /// <summary>Сохранить структуру сети в файл XMl</summary>
    /// <param name="FileName">Имя файла для сохранения данных</param>
    public async Task SaveXmlToAsync(string FileName)
    {
        using var file = File.CreateText(FileName.NotNull());
        await SaveXmlToAsync(file).ConfigureAwait(false);
    }

    /// <summary>Сохранить структуру сети в поток в формате XML</summary>
    /// <param name="Stream">Поток для сохранения</param>
    public void SaveXmlTo(Stream Stream) => SaveXmlTo(XmlWriter.Create(Stream.NotNull(), new XmlWriterSettings { ConformanceLevel = ConformanceLevel.Fragment }));

    /// <summary>Сохранить структуру сети в поток в формате XML</summary>
    /// <param name="Stream">Поток для сохранения</param>
    public async Task SaveXmlToAsync(Stream Stream) => await SaveXmlToAsync(XmlWriter.Create(Stream.NotNull(), new XmlWriterSettings { ConformanceLevel = ConformanceLevel.Fragment }));

    /// <summary>Сохранить структуру сети в текст в формате XML</summary>
    /// <param name="Writer">Объект записи текста</param>
    public void SaveXmlTo(TextWriter Writer) => SaveXmlTo(XmlWriter.Create(Writer.NotNull(), new XmlWriterSettings { ConformanceLevel = ConformanceLevel.Fragment }));

    /// <summary>Сохранить структуру сети в текст в формате XML</summary>
    /// <param name="Writer">Объект записи текста</param>
    public async Task SaveXmlToAsync(TextWriter Writer) => await SaveXmlToAsync(XmlWriter.Create(Writer.NotNull(), new XmlWriterSettings { ConformanceLevel = ConformanceLevel.Fragment }));

    /// <summary>Сохранить структуру сети в формате XML</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    public void SaveXmlTo(XmlWriter Writer)
    {
        if (Writer is null) throw new ArgumentNullException(nameof(Writer));

        Writer.WriteStartElement("Network"); // <Network>
        WriteLayers(Writer);
        Writer.WriteEndElement(); // </Network>
        Writer.Flush();
    }

    /// <summary>Сохранить структуру сети в формате XML</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    public async Task SaveXmlToAsync(XmlWriter Writer)
    {
        if (Writer is null) throw new ArgumentNullException(nameof(Writer));

        await Writer.WriteStartElementAsync(null, "Network", null).ConfigureAwait(false);
        await WriteLayersAsync(Writer).ConfigureAwait(false);
        await Writer.WriteEndElementAsync().ConfigureAwait(false);
        await Writer.FlushAsync().ConfigureAwait(false);
    }

    /// <summary>Записать данные слоёв</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    private void WriteLayers(XmlWriter Writer)
    {
        for (var layer_index = 0; layer_index < _Layers.Length; layer_index++)
        {
            var activation = _Activations[layer_index];
            if (activation?.GetType() == typeof(Lambda)) throw new InvalidOperationException("Невозможно серилизовать лямда-функцию активации");

            Writer.WriteStartElement("Layer"); // <Layer Index="****">
            Writer.WriteAttributeString("Index", layer_index.ToString());
            Writer.WriteAttributeString("Activation", (activation is null ? typeof(Sigmoid) : activation.GetType()).ToString());
            WriteNeurons(Writer, layer_index);
            Writer.WriteEndElement(); // </Layer>
        }
    }

    /// <summary>Записать данные слоёв</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    private async Task WriteLayersAsync(XmlWriter Writer)
    {
        for (var layer_index = 0; layer_index < _Layers.Length; layer_index++)
        {
            var activation = _Activations[layer_index];
            if (activation?.GetType() == typeof(Lambda)) throw new InvalidOperationException("Невозможно сериaлизовать лямбда-функцию активации");

            await Writer.WriteStartElementAsync(null, "Layer", null).ConfigureAwait(false);
            await Writer.WriteAttributeStringAsync(null, "Index", null, layer_index.ToString()).ConfigureAwait(false);
            await Writer.WriteAttributeStringAsync(null, "Activation", null, (activation is null ? typeof(Sigmoid) : activation.GetType()).ToString()).ConfigureAwait(false);
            await WriteNeuronsAsync(Writer, layer_index).ConfigureAwait(false);
            await Writer.WriteEndElementAsync().ConfigureAwait(false); // </Layer>
        }
    }

    /// <summary>Записать данные нейронов слоя</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    /// <param name="LayerIndex">Индекс слоя</param>
    private void WriteNeurons(XmlWriter Writer, int LayerIndex)
    {
        var layer         = _Layers[LayerIndex];
        var neurons_count = layer.GetLength(0);
        for (var neuron_index = 0; neuron_index < neurons_count; neuron_index++)
            WriteNeuron(Writer, LayerIndex, neuron_index, layer);
    }

    /// <summary>Записать данные нейронов слоя</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    /// <param name="LayerIndex">Индекс слоя</param>
    private async Task WriteNeuronsAsync(XmlWriter Writer, int LayerIndex)
    {
        var layer         = _Layers[LayerIndex];
        var neurons_count = layer.GetLength(0);
        for (var neuron_index = 0; neuron_index < neurons_count; neuron_index++)
            await WriteNeuronAsync(Writer, LayerIndex, neuron_index, layer).ConfigureAwait(false);
    }

    /// <summary>Записать данные нейрона</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    /// <param name="LayerIndex">Индекс слоя</param>
    /// <param name="NeuronIndex">Индекс нейрона</param>
    /// <param name="LayerWeights">Матрица весовых коэффициентов слоя</param>
    private void WriteNeuron(XmlWriter Writer, int LayerIndex, int NeuronIndex, double[,] LayerWeights)
    {
        Writer.WriteStartElement("Neuron"); // <Neuron index="****" Offset="****" OffsetWeight="****">
        Writer.WriteAttributeString("Index", NeuronIndex.ToString());
        var culture = CultureInfo.InvariantCulture;
        Writer.WriteAttributeString("Offset", _Offsets[LayerIndex][NeuronIndex].ToString(__NumberFormat, culture));
        Writer.WriteAttributeString("OffsetWeight", _OffsetsWeights[LayerIndex][NeuronIndex].ToString(__NumberFormat, culture));
        var layer_out = LayerIndex < _Layers.Length - 1 ? _Outputs[LayerIndex] : null;
        if (layer_out != null)
            Writer.WriteAttributeString("Out", layer_out[NeuronIndex].ToString(__NumberFormat, culture));

        var weights_buffer = new StringBuilder();
        var inputs_count   = LayerWeights.GetLength(1);
        for (var input_index = 0; input_index < inputs_count - 1; input_index++)
            weights_buffer.AppendFormat("{0}; ", LayerWeights[NeuronIndex, input_index].ToString(__NumberFormat, culture));
        weights_buffer.Append(LayerWeights[NeuronIndex, inputs_count - 1].ToString(__NumberFormat, culture));
        Writer.WriteString(weights_buffer.ToString());

        Writer.WriteEndElement(); // </Neuron>
    }

    /// <summary>Асинхронно записать данные нейрона</summary>
    /// <param name="Writer">Объект записи данных XML</param>
    /// <param name="LayerIndex">Индекс слоя</param>
    /// <param name="NeuronIndex">Индекс нейрона</param>
    /// <param name="LayerWeights">Матрица весовых коэффициентов слоя</param>
    private async Task WriteNeuronAsync(XmlWriter Writer, int LayerIndex, int NeuronIndex, double[,] LayerWeights)
    {
        var create_buffer_task = LayerWeights.Async(NeuronIndex, (layers, index) =>
        {
            var weights_buffer    = new StringBuilder();
            var invariant_culture = CultureInfo.InvariantCulture;
            var inputs_count      = layers.GetLength(1);
            for (var input_index = 0; input_index < inputs_count - 1; input_index++)
                weights_buffer.AppendFormat("{0}; ", layers[index, input_index].ToString(__NumberFormat, invariant_culture));
            weights_buffer.Append(layers[index, inputs_count - 1].ToString(__NumberFormat, invariant_culture));
            return weights_buffer.ToString();
        });

        await Writer.WriteStartElementAsync(null, "Neuron", null).ConfigureAwait(false);
        await Writer.WriteAttributeStringAsync(null, "Index", null, NeuronIndex.ToString()).ConfigureAwait(false);
        var culture = CultureInfo.InvariantCulture;
        await Writer.WriteAttributeStringAsync(null, "Offset", null, _Offsets[LayerIndex][NeuronIndex].ToString(__NumberFormat, culture)).ConfigureAwait(false);
        await Writer.WriteAttributeStringAsync(null, "OffsetWeight", null, _OffsetsWeights[LayerIndex][NeuronIndex].ToString(__NumberFormat, culture)).ConfigureAwait(false);
        var layer_out = LayerIndex < _Layers.Length - 1 ? _Outputs[LayerIndex] : null;
        if (layer_out != null)
            await Writer.WriteAttributeStringAsync(null, "Out", null, layer_out[NeuronIndex].ToString(__NumberFormat, culture)).ConfigureAwait(false);

        await Writer.WriteStringAsync(await create_buffer_task.ConfigureAwait(false) ?? throw new InvalidOperationException()).ConfigureAwait(false);

        await Writer.WriteEndElementAsync().ConfigureAwait(false);
    }

    #endregion

    #region Чтение

    /// <summary>Чтение данных сети в формате XML из файла</summary>
    /// <param name="FileName">Имя файла, содержащего структуру сети в формате XML</param>
    /// <returns>Прочитанная из файла нейронная сеть</returns>
    public static MultilayerPerceptron LoadXmlFrom(string FileName)
    {
        if (FileName is null) throw new ArgumentNullException(nameof(FileName));
        if (!File.Exists(FileName)) throw new FileNotFoundException($"Файл {FileName} не найден", FileName);
using var file = File.OpenText(FileName);
        return LoadXmlFrom(file);
    }

    /// <summary>Чтение данных сети в формате XML из файла</summary>
    /// <param name="FileName">Имя файла, содержащего структуру сети в формате XML</param>
    /// <returns>Прочитанная из файла нейронная сеть</returns>
    public static async Task<MultilayerPerceptron> LoadXmlFromAsync(string FileName)
    {
        if (FileName is null) throw new ArgumentNullException(nameof(FileName));
        if (!File.Exists(FileName)) throw new FileNotFoundException($"Файл {FileName} не найден", FileName);
using var file = File.OpenText(FileName);
        return await LoadXmlFromAsync(file);
    }

    /// <summary>Загрузка данных нейронной сети из потока в формате XML</summary>
    /// <param name="Data">Поток данных сети</param>
    /// <returns>Прочитанная нейронная сеть</returns>
    public static MultilayerPerceptron LoadFromXml(Stream Data) => LoadXmlFrom(XmlReader.Create(Data.NotNull()));

    /// <summary>Загрузка данных нейронной сети из потока в формате XML</summary>
    /// <param name="Data">Поток данных сети</param>
    /// <returns>Прочитанная нейронная сеть</returns>
    public static async Task<MultilayerPerceptron> LoadFromXmlAsync(Stream Data) => await LoadXmlFromAsync(XmlReader.Create(Data.NotNull()));

    /// <summary>Загрузка данных сети из текста в формате XML</summary>
    /// <param name="Reader">Источник текстовых данных, хранящий структуру сети в формате XML</param>
    /// <returns>Прочитанная нейронная сеть</returns>
    public static MultilayerPerceptron LoadXmlFrom(TextReader Reader) => LoadXmlFrom(XmlReader.Create(Reader.NotNull()));

    /// <summary>Загрузка данных сети из текста в формате XML</summary>
    /// <param name="Reader">Источник текстовых данных, хранящий структуру сети в формате XML</param>
    /// <returns>Прочитанная нейронная сеть</returns>
    public static async Task<MultilayerPerceptron> LoadXmlFromAsync(TextReader Reader) => await LoadXmlFromAsync(XmlReader.Create(Reader.NotNull()));

    /// <summary>Чтение структуры сети из XML</summary>
    /// <param name="Reader">Источник XML, хранящий структуру сети</param>
    /// <returns>Прочитанная нейронная сеть</returns>
    public static MultilayerPerceptron LoadXmlFrom(XmlReader Reader)
    {
        if (Reader is null) throw new ArgumentNullException(nameof(Reader));

        Reader.ReadStartElement("Network");

        var activation_functions_pool = new Dictionary<string, ActivationFunction>();
        var activation_functions      = new List<ActivationFunction>();
        var weights                   = new List<double[,]>();
        var offsets                   = new List<IList<double>>();
        var offset_weights            = new List<IList<double>>();
        var outputs                   = new List<IList<double>>();

        while (!Reader.EOF)
        {
            if (Reader.NodeType is XmlNodeType.Whitespace or XmlNodeType.EndElement || Reader.Name != "Layer")
            {
                Reader.Skip();
                continue;
            }

            var activation_type_name = Reader.GetAttribute("Activation");
            if (activation_type_name == null)
                activation_functions.Add(null);
            else
            {
                var activation = activation_functions_pool.GetValueOrAddNew(activation_type_name, ActivationFunctionCreator);
                activation_functions.Add(activation is Sigmoid ? null : activation);
            }
            ReadLayer(Reader, weights, offsets, offset_weights, outputs);
        }

        var network      = new MultilayerPerceptron(weights.ToArray());
        var layers_count = network.LayersCount;
        foreach (var layer in network.Layer)
        {
            var layer_index = layer.LayerIndex;
            layer.SetOffsets(offsets[layer_index]);
            layer.SetOffsetWeights(offset_weights[layer_index]);
            if (layer_index < layers_count - 1) // Установка значений выходов сети для всех слоёв кроме выходного
                layer.SetOutputs(outputs[layer_index]);
            layer.Activation = activation_functions[layer_index];
        }

        return network;
    }

    /// <summary>Чтение структуры сети из XML</summary>
    /// <param name="Reader">Источник XML, хранящий структуру сети</param>
    /// <returns>Прочитанная нейронная сеть</returns>
    public static async Task<MultilayerPerceptron> LoadXmlFromAsync(XmlReader Reader)
    {
        if (Reader is null) throw new ArgumentNullException(nameof(Reader));

        await Reader.Async(r => r.ReadStartElement("Network")).ConfigureAwait(false);

        var activation_functions_pool = new Dictionary<string, ActivationFunction>();
        var activation_functions      = new List<ActivationFunction>();
        var weights                   = new List<double[,]>();
        var offsets                   = new List<IList<double>>();
        var offset_weights            = new List<IList<double>>();
        var outputs                   = new List<IList<double>>();

        while (!Reader.EOF)
        {
            if (Reader.NodeType is XmlNodeType.Whitespace or XmlNodeType.EndElement || Reader.Name != "Layer")
            {
                await Reader.SkipAsync().ConfigureAwait(false);
                continue;
            }

            var activation_type_name = Reader.GetAttribute("Activation");
            if (activation_type_name == null)
                activation_functions.Add(null);
            else
            {
                var activation = activation_functions_pool.GetValueOrAddNew(activation_type_name, ActivationFunctionCreator);
                activation_functions.Add(activation is Sigmoid ? null : activation);
            }
            await ReadLayerAsync(Reader, weights, offsets, offset_weights, outputs).ConfigureAwait(false);
        }

        var network      = new MultilayerPerceptron(weights.ToArray());
        var layers_count = network.LayersCount;
        foreach (var layer in network.Layer)
        {
            var layer_index = layer.LayerIndex;
            layer.SetOffsets(offsets[layer_index]);
            layer.SetOffsetWeights(offset_weights[layer_index]);
            if (layer_index < layers_count - 1) // Установка значений выходов сети для всех слоёв кроме выходного
                layer.SetOutputs(outputs[layer_index]);
            layer.Activation = activation_functions[layer_index];
        }

        return network;
    }

    /// <summary>Чтение данных слоя</summary>
    /// <param name="Reader">Источник XML-данных</param>
    /// <param name="Weights">Коллекция весовых коэффициентов слоёв</param>
    /// <param name="Offsets">Коллекция смещений слоёв</param>
    /// <param name="OffsetWeights">Коллекция весовых коэффициентов смещений слоёв</param>
    /// <param name="Outputs">Коллекция значений выходов слоёв</param>
    private static void ReadLayer(
        XmlReader Reader,
        ICollection<double[,]> Weights,
        ICollection<IList<double>> Offsets,
        ICollection<IList<double>> OffsetWeights,
        ICollection<IList<double>> Outputs)
    {
        Reader.ReadStartElement("Layer");
        var  neurons_weights     = new List<double[]>();
        var  layer_offsets       = new List<double>();
        var  layer_offet_weights = new List<double>();
        var  layer_outputs       = new List<double>();
        int? inputs_count        = null;
        var  index               = 0;
        while (!Reader.EOF)
        {
            if (Reader.NodeType == XmlNodeType.Whitespace)
            {
                Reader.Skip();
                continue;
            }

            if (Reader.NodeType == XmlNodeType.EndElement)
                break;

            if (Reader.Name != "Neuron")
            {
                Reader.Skip();
                continue;
            }

            layer_offsets.Add(Reader.GetAttributeDouble("Offset") ?? 0);
            layer_offet_weights.Add(Reader.GetAttributeDouble("OffsetWeight") ?? 0);
            layer_outputs.Add(Reader.GetAttributeDouble("Out") ?? 0);
            var content = Reader.ReadElementContentAsString();
            var values  = content.Split(';').Select(S => double.Parse(S.Trim(), NumberFormatInfo.InvariantInfo)).ToArray();
            if (inputs_count is null)
                inputs_count = values.Length;
            else if (values.Length != inputs_count)
                throw new InvalidOperationException(
                    $"Количество весов ({values.Length}) для нейрона {index} не равно количеству весов для предыдущих нейронов ({inputs_count})");
            neurons_weights.Add(values);

            index++;
        }

        if (inputs_count is null) throw new InvalidOperationException("В слое отсутствуют нейроны");
        var layer_weights = new double[neurons_weights.Count, (int)inputs_count];
        for (var neuron = 0; neuron < neurons_weights.Count; neuron++)
            for (var input = 0; input < inputs_count; input++)
            {
                var neuron_weights = neurons_weights[neuron];
                layer_weights[neuron, input] = neuron_weights[input];
            }

        Weights.Add(layer_weights);
        Offsets.Add(layer_offsets);
        OffsetWeights.Add(layer_offet_weights);
        Outputs.Add(layer_outputs);
    }

    /// <summary>Чтение данных слоя</summary>
    /// <param name="Reader">Источник XML-данных</param>
    /// <param name="Weights">Коллекция весовых коэффициентов слоёв</param>
    /// <param name="Offsets">Коллекция смещений слоёв</param>
    /// <param name="OffsetWeights">Коллекция весовых коэффициентов смещений слоёв</param>
    /// <param name="Outputs">Коллекция значений выходов слоёв</param>
    private static async Task ReadLayerAsync(
        XmlReader Reader,
        ICollection<double[,]> Weights,
        ICollection<IList<double>> Offsets,
        ICollection<IList<double>> OffsetWeights,
        ICollection<IList<double>> Outputs)
    {
        await Reader.Async(r => r.ReadStartElement("Layer")).ConfigureAwait(false);
        var  neurons_weights     = new List<double[]>();
        var  layer_offsets       = new List<double>();
        var  layer_offet_weights = new List<double>();
        var  layer_outputs       = new List<double>();
        int? inputs_count        = null;
        var  index               = 0;
        while (!Reader.EOF)
        {
            if (Reader.NodeType == XmlNodeType.Whitespace)
            {
                await Reader.SkipAsync().ConfigureAwait(false);
                continue;
            }

            if (Reader.NodeType == XmlNodeType.EndElement) break;

            if (Reader.Name != "Neuron")
            {
                await Reader.SkipAsync().ConfigureAwait(false);
                continue;
            }

            layer_offsets.Add(Reader.GetAttributeDouble("Offset") ?? 0);
            layer_offet_weights.Add(Reader.GetAttributeDouble("OffsetWeight") ?? 0);
            layer_outputs.Add(Reader.GetAttributeDouble("Out") ?? 0);
            var content = await Reader.ReadElementContentAsStringAsync().ConfigureAwait(false);
            var values  = content.Split(';').Select(S => double.Parse(S.Trim(), NumberFormatInfo.InvariantInfo)).ToArray();
            if (inputs_count is null)
                inputs_count = values.Length;
            else if (values.Length != inputs_count)
                throw new InvalidOperationException(
                    $"Количество весов ({values.Length}) для нейрона {index} не равно количеству весов для предыдущих нейронов ({inputs_count})");
            neurons_weights.Add(values);

            index++;
        }

        if (inputs_count is null) throw new InvalidOperationException("В слое отсутствуют нейроны");
        var layer_weights = new double[neurons_weights.Count, (int)inputs_count];
        for (var neuron = 0; neuron < neurons_weights.Count; neuron++)
            for (var input = 0; input < inputs_count; input++)
            {
                var neuron_weights = neurons_weights[neuron];
                layer_weights[neuron, input] = neuron_weights[input];
            }

        Weights.Add(layer_weights);
        Offsets.Add(layer_offsets);
        OffsetWeights.Add(layer_offet_weights);
        Outputs.Add(layer_outputs);
    }

    /// <summary>Метод создания активационной функции на основе её имени</summary>
    /// <param name="FunctionTypeName">Имя типа активационной функции</param>
    /// <returns>Активационная функция с указанным именем</returns>
    private static ActivationFunction ActivationFunctionCreator(string FunctionTypeName)
    {
        if (string.IsNullOrWhiteSpace(FunctionTypeName)) throw new InvalidOperationException("Некорректный тип функции активации");
        if (!FunctionTypeName.Contains("."))
            FunctionTypeName = $"{typeof(ActivationFunction).Namespace}.{FunctionTypeName}";

        var type = Type.GetType(FunctionTypeName);
        if (type is null) throw new InvalidOperationException($"Тип активационной функции {FunctionTypeName} не найден");
        return (ActivationFunction)Activator.CreateInstance(type);
    }

    #endregion

    #endregion

    #region IEquatable<MultilayerPerceptron>

    public bool Equals(MultilayerPerceptron? other)
    {
        if (other is null) return false;
        var layers_count = LayersCount;
        if (layers_count != other.LayersCount) return false;
        if (InputsCount != other.InputsCount) return false;
        if (OutputsCount != other.OutputsCount) return false;
        var double_equality = EqualityComparer<double>.Default;
        for (var layer_index = 0; layer_index < layers_count; layer_index++)
        {
            var current_layer = Layer[layer_index];
            var other_layer   = other.Layer[layer_index];
            var inputs_count  = current_layer.InputsCount;
            if (inputs_count != other_layer.InputsCount) return false;
            var outputs_count = current_layer.OutputsCount;
            if (outputs_count != other_layer.OutputsCount) return false;
            var current_offsets = current_layer.Offsets;
            var other_offsets   = other_layer.Offsets;
            for (var i = 0; i < current_offsets.Length; i++)
                if (!double_equality.Equals(current_offsets[i], other_offsets[i]))
                    return false;
            var current_offset_weights = current_layer.OffsetWeights;
            var other_offset_weights   = other_layer.OffsetWeights;
            for (var i = 0; i < current_offset_weights.Length; i++)
                if (!double_equality.Equals(current_offset_weights[i], other_offset_weights[i]))
                    return false;
            var current_weights = current_layer.Weights;
            var other_weights   = other_layer.Weights;
            for (var neuron = 0; neuron < outputs_count; neuron++)
                for (var input = 0; input < inputs_count; input++)
                    if (!double_equality.Equals(current_weights[neuron, input], other_weights[neuron, input]))
                        return false;
        }

        return true;
    }

    #endregion

    #region Overrides of Object

    public override bool Equals(object? obj) => Equals(obj as MultilayerPerceptron);

    public override int GetHashCode()
    {
        const int hash_base = 0x18d;
        var       hash      = Consts.BigPrime_int;
        foreach (var layer in Layer)
            unchecked
            {
                hash ^= layer.GetHashCode() * hash_base;
            }
        unchecked
        {
            hash ^= LayersCount * hash_base;
            hash ^= InputsCount * hash_base;
        }
        return hash;

    }

    public static bool operator ==(MultilayerPerceptron? net1, MultilayerPerceptron? net2) => ReferenceEquals(net1, net2) || net1?.Equals(net2) == true;

    public static bool operator !=(MultilayerPerceptron? net1, MultilayerPerceptron? net2) => !(net1 == net2);

    #endregion

    /* --------------------------------------------------------------------------------------------- */
}