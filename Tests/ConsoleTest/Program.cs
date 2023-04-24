using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;


const int N = 2560;
const int M = 1600;

//const int N = 5;
//const int M = 3;

//const int N = 40;
//const int M = 20;

var input = new Bitmap(N, M);

using (var g = Graphics.FromImage(input))
    g.CopyFromScreen(0, 0, 0, 0, input.Size);

int[,] core1 =
{
    { -1, 0, 1 },
    { -1, 0, 1 },
    { -1, 0, 1 }
};

//int[,] core1 =
//{
//    { 1 }
//};

var N1 = N - core1.GetLength(0) + 1;
var M1 = M - core1.GetLength(1) + 1;
var output = new Bitmap(N1, M1);

Convolution(input, core1, output);

input.Save("input.bmp");
output.Save("out.bmp");

//Console.WriteLine("End.");
//Console.ReadLine();


static unsafe void Convolution(Bitmap Input, int[,] core, Bitmap Out)
{
    var input_cols = Input.Size.Width;
    var (out_cols, out_rows) = Out.Size;

    var (core_cols, core_rows) = (core.GetLength(0), core.GetLength(1));

    var input_bits = Input.LockBits(new(new(), Input.Size), ImageLockMode.ReadOnly, Input.PixelFormat);
    var output_bits = Out.LockBits(new(new(), Out.Size), ImageLockMode.WriteOnly, Input.PixelFormat);

    var input_bytes = new ReadOnlySpan<byte>(input_bits.Scan0.ToPointer(), input_bits.Stride * Input.Height);
    var output_bytes = new Span<byte>(output_bits.Scan0.ToPointer(), output_bits.Stride * Out.Height);

    var input_pixels = MemoryMarshal.Cast<byte, int>(input_bytes);
    var output_pixels = MemoryMarshal.Cast<byte, int>(output_bytes);

    for(var row = 0; row < out_rows; row++)
        for (var col = 0; col < out_cols; col++) 
        {
            var (r, g, b) = (0, 0, 0);

            for(var core_row = 0; core_row < core_rows; core_row++)
                for(var core_col = 0; core_col < core_cols; core_col++)
                {
                    static void ToRGB(int v, out int  r, out int g, out int b)
                    {
                        r = (byte)(v >> 16);
                        g = (byte)(v >> 8);
                        b = (byte)v;
                    }

                    //var (r1, g1, b1) = Color.FromArgb(input_pixels[(n + i) * Mi + m + j]);

                    var col0 = col + core_col;
                    var row0 = row + core_row;

                    var mm = row0 * input_cols;
                    var index = mm + col0;

                    var pixel = input_pixels[index];

                    ToRGB(pixel, out var r1, out var g1, out var b1);
                    r += r1 * core[core_col, core_row];
                    g += g1 * core[core_col, core_row];
                    b += b1 * core[core_col, core_row];
                }

            var i2 = row * out_cols + col;

            output_pixels[i2] = Color.FromArgb(
                Math.Max(0, Math.Min(r, byte.MaxValue - 1)), 
                Math.Max(0, Math.Min(g, byte.MaxValue - 1)),
                Math.Max(0, Math.Min(b, byte.MaxValue - 1)))
                .ToArgb();
        }

    Input.UnlockBits(input_bits);
    Out.UnlockBits(output_bits);
}

static class Ex
{
    public static void Deconstruct(this Color c, out byte r, out byte g, out byte b) => (r, g, b) = (c.R, c.G, c.B);
}