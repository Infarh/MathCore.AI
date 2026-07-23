namespace MathCore.AI.NeuralNetworks;

/// <summary>Построитель каскадных нейронных сетей с fluent-интерфейсом</summary>
/// <remarks>
/// Позволяет выстраивать последовательные и разветвлённые архитектуры,
/// комбинируя любые сети, реализующие <see cref="INeuralNetwork"/>.
///
/// Примеры использования:
/// <example>
/// // Простой последовательный каскад
/// var cascade = CascadeBuilder.From(net1).Then(net2).Then(net3).Build();
///
/// // Каскад с разветвлением (Tee) — выход net1 подаётся на net2 и net3 параллельно,
/// // их выходы конкатенируются и подаются на net4
/// var cascade = CascadeBuilder
///     .From(net1)
///     .Tee(net2, net3)
///     .Then(net4)
///     .Build();
///
/// // Параллельный вход: конкатенация выходов netA и netB → netC
/// var cascade = CascadeBuilder
///     .Join(netA, netB)
///     .Then(netC)
///     .Build();
///
/// // Вложенный каскад
/// var inner = CascadeBuilder.From(net1).Then(net2).Build();
/// var outer = CascadeBuilder.From(inner).Then(net3).Build();
/// </example>
/// </remarks>
public class CascadeBuilder
{
    /* --------------------------------------------------------------------------------------------- */

    #region Вспомогательные классы

    /// <summary>Источник-объединение нескольких сетей, работающих параллельно</summary>
    /// <remarks>
    /// Используется в <see cref="CascadeBuilder.Join(INeuralNetwork,INeuralNetwork)"/> и
    /// подобных методах. Входной вектор подаётся на каждую из сетей, их выходы конкатенируются.
    /// </remarks>
    public sealed class ConcatSources : INeuralNetwork
    {
        private readonly INeuralNetwork[] _Networks;
        private readonly int[] _Offsets;

        /// <inheritdoc />
        public int InputsCount { get; }

        /// <inheritdoc />
        public int OutputsCount { get; }

        /// <summary>Сети-источники</summary>
        public IReadOnlyList<INeuralNetwork> Sources => _Networks;

        /// <summary>Инициализация нового объединения источников</summary>
        /// <param name="Networks">Сети, работающие параллельно от одного входа</param>
        /// <exception cref="ArgumentNullException">Если <paramref name="Networks"/> равен null</exception>
        /// <exception cref="ArgumentException">Если массив <paramref name="Networks"/> пуст или входы не совпадают</exception>
        public ConcatSources(params INeuralNetwork[] Networks)
        {
            if (Networks is null) throw new ArgumentNullException(nameof(Networks));
            if (Networks.Length == 0) throw new ArgumentException("Массив сетей не может быть пустым", nameof(Networks));

            _Networks = Networks;
            _Offsets = new int[Networks.Length];

            InputsCount = Networks[0].InputsCount;
            var total = 0;
            for (var i = 0; i < Networks.Length; i++)
            {
                if (Networks[i].InputsCount != InputsCount)
                    throw new ArgumentException(
                        $"Количество входов сети {i} ({Networks[i].InputsCount}) не совпадает " +
                        $"с количеством входов первой сети ({InputsCount})", nameof(Networks));
                _Offsets[i] = total;
                total += Networks[i].OutputsCount;
            }
            OutputsCount = total;
        }

        /// <inheritdoc />
        public void Process(Span<double> Input, Span<double> Output)
        {
            for (var i = 0; i < _Networks.Length; i++)
            {
                var network = _Networks[i];
                var offset = _Offsets[i];
                var segment = Output.Slice(offset, network.OutputsCount);
                network.Process(Input, segment);
            }
        }
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Шаг построения

    /// <summary>Промежуточный шаг построения каскада</summary>
    public sealed class StepBuilder
    {
        /// <summary>Накопленный список слотов</summary>
        private readonly List<CascadeNetwork.Slot> _Slots;

        /// <summary>Инициализация нового шага построителя</summary>
        /// <param name="FirstSlot">Первый слот каскада</param>
        internal StepBuilder(CascadeNetwork.Slot FirstSlot) => _Slots = [FirstSlot];

        /// <summary>Добавить следующий слот в последовательность</summary>
        /// <param name="Slot">Слот, следующий в цепочке</param>
        /// <returns>Тот же построитель для дальнейшего fluent-вызова</returns>
        private StepBuilder Then(CascadeNetwork.Slot Slot)
        {
            _Slots.Add(Slot);
            return this;
        }

        /// <summary>Добавить следующую сеть в последовательность каскада</summary>
        /// <param name="Network">Сеть, выполняющая следующий этап обработки</param>
        /// <returns>Тот же построитель для дальнейшего fluent-вызова</returns>
        public StepBuilder Then(INeuralNetwork Network)
        {
            Network.NotNull();
            return Then(new CascadeNetwork.NetworkSlot(Network));
        }

        /// <summary>Добавить разветвление: выход текущего звена подаётся на несколько сетей параллельно,
        /// их выходы конкатенируются</summary>
        /// <param name="Networks">Сети, работающие параллельно от одного входа</param>
        /// <returns>Тот же построитель для дальнейшего fluent-вызова</returns>
        public StepBuilder Tee(params INeuralNetwork[] Networks)
        {
            Networks.NotNull();
            if (Networks.Length == 0) throw new ArgumentException("Массив сетей не может быть пустым", nameof(Networks));

            // Создаём ConcatSources, он же INeuralNetwork → оборачиваем в NetworkSlot
            var concat = new ConcatSources(Networks);
            return Then(new CascadeNetwork.NetworkSlot(concat));
        }

        /// <summary>Добавить разветвление с явным указанием слотов вместо INeuralNetwork</summary>
        /// <param name="Slots">Слоты, работающие параллельно от одного входа</param>
        /// <returns>Тот же построитель для дальнейшего fluent-вызова</returns>
        public StepBuilder TeeSlots(params CascadeNetwork.Slot[] Slots)
        {
            Slots.NotNull();
            if (Slots.Length == 0) throw new ArgumentException("Массив слотов не может быть пустым", nameof(Slots));

            var concat = new CascadeNetwork.ConcatSlot(Slots);
            return Then(concat);
        }

        /// <summary>Добавить в каскад произвольный слот</summary>
        /// <param name="Slot">Произвольный слот</param>
        /// <returns>Тот же построитель для дальнейшего fluent-вызова</returns>
        public StepBuilder ThenSlot(CascadeNetwork.Slot Slot)
        {
            Slot.NotNull();
            return Then(Slot);
        }

        /// <summary>Построить каскадную сеть</summary>
        /// <returns>Готовая каскадная сеть</returns>
        public CascadeNetwork Build() => new([.. _Slots]);
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */

    #region Фабричные методы

    /// <summary>Начать построение каскада с указанной сети</summary>
    /// <param name="Network">Первая сеть в каскаде</param>
    /// <returns>Промежуточный построитель для fluent-цепочки</returns>
    public static StepBuilder From(INeuralNetwork Network)
    {
        Network.NotNull();
        return new StepBuilder(new CascadeNetwork.NetworkSlot(Network));
    }

    /// <summary>Начать построение каскада с произвольного слота</summary>
    /// <param name="Slot">Первый слот в каскаде</param>
    /// <returns>Промежуточный построитель для fluent-цепочки</returns>
    public static StepBuilder FromSlot(CascadeNetwork.Slot Slot)
    {
        Slot.NotNull();
        return new StepBuilder(Slot);
    }

    /// <summary>Начать построение каскада с параллельного объединения нескольких сетей,
    /// работающих от одного входа</summary>
    /// <param name="Network1">Первая сеть объединения</param>
    /// <param name="Network2">Вторая сеть объединения</param>
    /// <param name="Networks">Дополнительные сети объединения</param>
    /// <returns>Промежуточный построитель для fluent-цепочки</returns>
    public static StepBuilder Join(INeuralNetwork Network1, INeuralNetwork Network2, params INeuralNetwork[] Networks)
    {
        Network1.NotNull();
        Network2.NotNull();
        Networks.NotNull();

        var all = new INeuralNetwork[Networks.Length + 2];
        all[0] = Network1;
        all[1] = Network2;
        Array.Copy(Networks, 0, all, 2, Networks.Length);

        var concat = new ConcatSources(all);
        return new StepBuilder(new CascadeNetwork.NetworkSlot(concat));
    }

    #endregion

    /* --------------------------------------------------------------------------------------------- */
}
