#nullable enable
namespace MathCore.AI.Tests.Infrastructure;

internal static class VectorsEx
{
    public static double[] Add(this double[] a, double[] b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new InvalidOperationException("Размеры векторов не совпадают");

        var length = a.Length;
        var result = new double[length];
        for (var i = 0; i < length; i++)
            result[i] = a[i] + b[i];

        return result;
    }

    public static double[] Sub(this double[] a, double[] b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new InvalidOperationException("Размеры векторов не совпадают");

        var length = a.Length;
        var result = new double[length];
        for (var i = 0; i < length; i++)
            result[i] = a[i] - b[i];

        return result;
    }

    public static double[] Mul(this double[] a, double[] b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Length != b.Length) throw new InvalidOperationException("Размеры векторов не совпадают");

        var length = a.Length;
        var result = new double[length];
        for (var i = 0; i < length; i++)
            result[i] = a[i] * b[i];

        return result;
    }

    public static double[] Mul(this double[,] A, double[] b)
    {
        if (A is null) throw new ArgumentNullException(nameof(A));
        if (b is null) throw new ArgumentNullException(nameof(b));

        var cols_count = A.GetLength(0);
        var rows_count = A.GetLength(1);
        if (b.Length != cols_count) 
            throw new InvalidOperationException("Число столбцов матрицы не совпадает с размерностью вектора");

        var result = new double[cols_count];
        for (var i = 0; i < cols_count; i++)
        {
            var s  = 0d;
            var bi = b[i];
            for (var j = 0; j < rows_count; j++)
                s += A[i, j] * bi;

            result[i] = s;
        }

        return result;
    }

    public static double[] MulT(this double[,] A, double[] b)
    {
        if (b is null) throw new ArgumentNullException(nameof(b));
        var (cols_count, rows_count) = A ?? throw new ArgumentNullException(nameof(A));

        if (b.Length != cols_count) throw new InvalidOperationException("Число строк матрицы не совпадает с размерностью вектора");

        var result = new double[rows_count];
        for (var i = 0; i < rows_count; i++)
        {
            var s = 0d;
            for (var j = 0; j < cols_count; j++)
                s += A[j, i] * b[j];

            result[i] = s;
        }

        return result;
    }

    public static double[,] Mul(this double[,] A, double[,] B)
    {
        var (cols_a_count, rows_a_count) = A ?? throw new ArgumentNullException(nameof(A));
        var (cols_b_count, rows_b_count) = B ?? throw new ArgumentNullException(nameof(B));

        if (rows_a_count != cols_b_count)
            throw new InvalidOperationException("Число столбцов матрицы A не равно числу строк матрицы B");

        var result = new double[cols_a_count, cols_b_count];

        for (var i = 0; i < cols_a_count; i++)
            for (var j = 0; j < rows_b_count; j++)
            {
                var s = 0d;
                for (var k = 0; k < rows_a_count; k++)
                    s += A[i, k] * B[k, j];
                result[i, j] = s;
            }

        return result;
    }
}
