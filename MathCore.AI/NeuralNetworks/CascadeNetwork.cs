namespace MathCore.AI.NeuralNetworks;

/// <summary>Каскадная нейронная сеть, объединяющая несколько сетей в последовательную или разветвлённую архитектуру</summary>
/// <remarks>
/// Реализует <see cref="INeuralNetwork"/>, что позволяет встраивать каскад как элемент другого каскада.
/// Управление буферами выполняется внутри класса, пользователю не нужно заботиться о выделении памяти.
/// </remarks>
public class CascadeNetwork : INeuralNetwork
{
    /* --------------------------------------------------------------------------------------------- */

    #region Слот — элемент каскада

    /// <summary>Слот каскада, описывающий одно преобразование вектора данных</summary>
    public abstract class Slot
    {
        /// <summary>Число входов слота</summary>
        public abstract int InputsCount { get; }

        /// <summary>Число выходов слота</summary>
        public abstract int OutputsCount { get; }

        /// <summary>Выполнить преобразование данных через слот</summary>
        /// <param name="Input">Входной массив данных</param>
        /// <param name="Output">Выходной массив для результата</param>
        public abstract void Process(Span<double> Input, Span<double> Output);
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Слот-сеть — последовательное выполнение одной INeuralNetwork

    /// <summary>Слот, выполняющий обработку через одну <see cref="INeuralNetwork"/></summary>
    internal sealed class NetworkSlot : Slot
    {
        private readonly INeuralNetwork _Network;

        /// <inheritdoc />
        public override int InputsCount => _Network.InputsCount;

        /// <inheritdoc />
        public override int OutputsCount => _Network.OutputsCount;

        /// <summary>Сеть, выполняющая обработку</summary>
        public INeuralNetwork Network => _Network;

        /// <summary>Инициализация нового слота-сети</summary>
        /// <param name="Network">Нейронная сеть, выполняющая обработку</param>
        public NetworkSlot(INeuralNetwork Network) => _Network = Network.NotNull();

        /// <inheritdoc />
        public override void Process(Span<double> Input, Span<double> Output) => _Network.Process(Input, Output);
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Слот-конкатенация — объединение нескольких выходов

    /// <summary>Слот, выполняющий конкатенацию нескольких источников в один выходной массив</summary>
    public sealed class ConcatSlot : Slot
    {
        /// <summary>Массив слотов-источников</summary>
        private readonly Slot[] _Sources;

        /// <summary>Смещения выходов каждого источника в результирующем массиве</summary>
        private readonly int[] _Offsets;

        /// <inheritdoc />
        public override int InputsCount => _Sources.Max(s => s.InputsCount);

        /// <inheritdoc />
        public override int OutputsCount { get; }

        /// <summary>Массив слотов-источников</summary>
        public IReadOnlyList<Slot> Sources => _Sources;

        /// <summary>Инициализация нового слота конкатенации</summary>
        /// <param name="Sources">Слоты-источники, чьи выходы необходимо объединить</param>
        /// <exception cref="ArgumentNullException">Если <paramref name="Sources"/> равен null</exception>
        /// <exception cref="ArgumentException">Если массив <paramref name="Sources"/> пуст</exception>
        public ConcatSlot(params Slot[] Sources)
        {
            if (Sources is null) throw new ArgumentNullException(nameof(Sources));
            if (Sources.Length == 0) throw new ArgumentException("Массив источников не может быть пустым", nameof(Sources));

            _Sources = Sources;
            _Offsets = new int[Sources.Length];

            var total = 0;
            for (var i = 0; i < Sources.Length; i++)
            {
                _Offsets[i] = total;
                total += Sources[i].OutputsCount;
            }
            OutputsCount = total;
        }

        /// <inheritdoc />
        public override void Process(Span<double> Input, Span<double> Output)
        {
            for (var i = 0; i < _Sources.Length; i++)
            {
                var source = _Sources[i];
                var offset = _Offsets[i];
                var segment = Output.Slice(offset, source.OutputsCount);
                source.Process(Input, segment);
            }
        }
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Поля

    /// <summary>Слоты каскада в порядке выполнения</summary>
    private readonly Slot[] _Slots;

    /// <summary>Промежуточные буферы между слотами</summary>
    private readonly double[][] _Buffers;

    /// <summary>Смещения для буферов — ссылка на буфер предыдущего слота</summary>
    private readonly int _OutputBufferIndex;

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Свойства

    /// <inheritdoc />
    public int InputsCount { get; }

    /// <inheritdoc />
    public int OutputsCount { get; }

    /// <summary>Количество элементов (слотов) в каскаде</summary>
    public int StepsCount => _Slots.Length;

    /// <summary>Слоты каскада</summary>
    public IReadOnlyList<Slot> Steps => _Slots;

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Конструктор

    /// <summary>Инициализация нового каскада нейронных сетей</summary>
    /// <param name="Slots">Слоты каскада в порядке выполнения</param>
    /// <exception cref="ArgumentNullException">Если <paramref name="Slots"/> равен null</exception>
    /// <exception cref="ArgumentException">Если массив <paramref name="Slots"/> пуст</exception>
    public CascadeNetwork(params Slot[] Slots)
    {
        if (Slots is null) throw new ArgumentNullException(nameof(Slots));
        if (Slots.Length == 0) throw new ArgumentException("Каскад должен содержать хотя бы один слот", nameof(Slots));

        _Slots = Slots;
        InputsCount = Slots[0].InputsCount;

        // Рассчитываем буферы: между каждыми двумя слотами нужен промежуточный буфер
        // исключая последний — он пишет сразу в выходной массив
        var buffers_count = Slots.Length > 1 ? Slots.Length - 2 : 0;
        _Buffers = new double[Math.Max(0, buffers_count + 1)][];

        // Выходной слот — последний
        var last_slot = Slots[^1];
        OutputsCount = last_slot.OutputsCount;

        // Проверяем согласованность размеров
        for (var i = 0; i < Slots.Length - 1; i++)
        {
            var current = Slots[i];
            var next = Slots[i + 1];
            if (current.OutputsCount != next.InputsCount)
                throw new InvalidOperationException(
                    $"Несогласованность размеров: выход слота {i} ({current.OutputsCount}) " +
                    $"не соответствует входу слота {i + 1} ({next.InputsCount})");
        }

        // Инициализируем буферы
        for (var i = 0; i < _Buffers.Length; i++)
        {
            // Буфер i передаётся от слота i к слоту i+1
            // Размер буфера определяется выходом слота i
            _Buffers[i] = new double[Slots[i].OutputsCount];
        }

        _OutputBufferIndex = _Buffers.Length - 1;
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Обработка

    /// <inheritdoc />
    /// <remarks>Выполняет последовательную обработку данных через все слоты каскада</remarks>
    public void Process(Span<double> Input, Span<double> Output)
    {
        if (Input.Length != InputsCount) throw new ArgumentException(
            $"Размер входного вектора ({Input.Length}) не равен количеству входов каскада ({InputsCount})", nameof(Input));
        if (Output.Length != OutputsCount) throw new ArgumentException(
            $"Размер выходного вектора ({Output.Length}) не равен количеству выходов каскада ({OutputsCount})", nameof(Output));

        // Первый слой — читает из Input
        var slots = _Slots;
        var buffers = _Buffers;
        var slots_count = slots.Length;

        if (slots_count == 1)
        {
            // Единственный слот — пишем сразу в Output
            slots[0].Process(Input, Output);
            return;
        }

        // Первый слот → первый буфер
        slots[0].Process(Input, buffers[0]);

        // Промежуточные слоты: i-й слот читает из буфера i-1, пишет в буфер i
        for (var i = 1; i < slots_count - 1; i++)
            slots[i].Process(buffers[i - 1], buffers[i]);

        // Последний слот читает из последнего буфера, пишет в Output
        slots[slots_count - 1].Process(buffers[_OutputBufferIndex], Output);
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */
}
