namespace MathCore.AI.NeuralNetworks;

/// <summary>Слой SoftMax — преобразует вектор вещественных чисел в распределение вероятностей</summary>
/// <remarks>
/// SoftMax применяется как выходной слой в задачах многоклассовой классификации.
/// Реализует <see cref="INeuralNetwork"/>, что позволяет встраивать его в каскад
/// через <see cref="CascadeBuilder"/>.
///
/// Число входов равно числу выходов: каждый вход нормируется относительно суммы экспонент всех входов.
/// Не содержит обучаемых весов — это чистый пост-процессор.
/// </remarks>
public sealed class SoftMaxLayer : INeuralNetwork
{
    /* --------------------------------------------------------------------------------------------- */

    /// <inheritdoc />
    public int InputsCount { get; }

    /// <inheritdoc />
    public int OutputsCount => InputsCount;

    /// <summary>Инициализация нового слоя SoftMax</summary>
    /// <param name="Size">Размер входного/выходного вектора</param>
    /// <exception cref="ArgumentOutOfRangeException">Если <paramref name="Size"/> меньше 1</exception>
    public SoftMaxLayer(int Size)
    {
        if (Size < 1) throw new ArgumentOutOfRangeException(nameof(Size), Size, "Размер SoftMax слоя должен быть больше 0");
        InputsCount = Size;
    }

    /* --------------------------------------------------------------------------------------------- */

    /// <inheritdoc />
    /// <remarks>
    /// Вычисляет SoftMax с численной стабилизацией:
    /// \[ \text{SoftMax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}} \]
    /// </remarks>
    public void Process(Span<double> Input, Span<double> Output)
    {
        if (Input.Length != InputsCount)
            throw new ArgumentException(
                $"Размер входного вектора ({Input.Length}) не равен размеру слоя ({InputsCount})", nameof(Input));
        if (Output.Length != OutputsCount)
            throw new ArgumentException(
                $"Размер выходного вектора ({Output.Length}) не равен размеру слоя ({OutputsCount})", nameof(Output));

        // Ищем максимум для численной стабильности
        var max = double.NegativeInfinity;
        for (var i = 0; i < Input.Length; i++)
            if (Input[i] > max) max = Input[i];

        // Вычисляем exp(x_i - max) и сумму
        var sum = 0.0;
        for (var i = 0; i < Input.Length; i++)
        {
            var exp_val = Math.Exp(Input[i] - max);
            Output[i] = exp_val;
            sum += exp_val;
        }

        // Нормируем
        var inv_sum = 1.0 / sum;
        for (var i = 0; i < Input.Length; i++)
            Output[i] *= inv_sum;
    }

    /* --------------------------------------------------------------------------------------------- */
}
