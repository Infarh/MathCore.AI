﻿using System.Diagnostics;

// ReSharper disable ArgumentsStyleAnonymousFunction
// ReSharper disable ArgumentsStyleNamedExpression
// ReSharper disable ArgumentsStyleLiteral
// ReSharper disable ArgumentsStyleOther
namespace MathCore.AI.Tests.NeuralNetworks.DomainTests;

[TestClass]
public class DigitsRecognition
{
    private static readonly Dictionary<char, int[]> __Symbols = new()
    {
        ['\0'] =
        [
            0,0,0,0,0, //|     
            0,0,0,0,0, //|     
            0,0,0,0,0, //|     
            0,0,0,0,0, //|     
            0,0,0,0,0, //|     
            0,0,0,0,0, //|     
            0,0,0,0,0, //|     
        ],
        ['0'] =
        [
            0,1,1,1,0, //| 111 
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            0,1,1,1,0, //|1   1
            0,1,1,1,0, //| 111 
        ],
        ['1'] =
        [
            0,0,1,0,0, //|  1  
            0,1,1,0,0, //| 11  
            1,0,1,0,0, //|1 1  
            0,0,1,0,0, //|  1  
            0,0,1,0,0, //|  1  
            0,0,1,0,0, //|  1  
            1,1,1,1,1, //|11111
        ],
        ['2'] =
        [
            0,1,1,1,0, //| 111 
            1,0,0,0,1, //|1   1
            0,0,0,0,1, //|    1
            0,0,0,1,0, //|   1 
            0,0,1,0,0, //|  1  
            0,1,0,0,0, //| 1   
            1,1,1,1,1, //|11111
        ],
        ['3'] =
        [
            0,1,1,1,0, //| 111 
            1,0,0,0,1, //|1   1
            0,0,0,0,1, //|    1
            0,0,0,1,0, //|   1 
            0,0,0,0,1, //|    1
            1,0,0,0,1, //|1   1
            0,1,1,1,0, //| 111 
        ],
        ['4'] =
        [
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            1,1,1,1,1, //|11111
            0,0,0,0,1, //|    1
            0,0,0,0,1, //|    1
            0,0,0,0,1, //|    1
        ],
        ['5'] =
        [
            1,1,1,1,1, //|11111
            1,0,0,0,0, //|1    
            1,1,1,1,0, //|1111 
            1,0,0,0,1, //|1   1
            0,0,0,0,1, //|    1
            1,0,0,0,1, //|1   1
            0,1,1,1,0, //| 111 
        ],
        ['6'] =
        [
            0,1,1,1,0, //| 111 
            1,0,0,0,1, //|1   1
            1,0,0,0,0, //|1    
            1,1,1,1,0, //|1111 
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            0,1,1,1,0, //| 111 
        ],
        ['7'] =
        [
            1,1,1,1,1, //|11111
            0,0,0,0,1, //|    1
            0,0,0,1,0, //|   1 
            0,0,1,0,0, //|  1  
            0,1,0,0,0, //| 1   
            1,0,0,0,0, //|1    
            1,0,0,0,0, //|1    
        ],
        ['8'] =
        [
            0,1,1,1,0, //| 111 
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            0,1,1,1,0, //| 111 
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            0,1,1,1,0, //| 111 
        ],
        ['9'] =
        [
            0,1,1,1,0, //| 111 
            1,0,0,0,1, //|1   1
            1,0,0,0,1, //|1   1
            0,1,1,1,1, //| 1111
            0,0,0,0,1, //|    1
            1,0,0,0,1, //|1   1
            0,1,1,1,0, //| 111 
        ],
    };

    private static readonly Dictionary<char, int[]> __BoldSymbols = new()
    {
        ['\0'] =
        [
            0,0,0,0,0,0,0,0, //|        
            0,0,0,0,0,0,0,0, //|        
            0,0,0,0,0,0,0,0, //|        
            0,0,0,0,0,0,0,0, //|        
            0,0,0,0,0,0,0,0, //|        
            0,0,0,0,0,0,0,0, //|        
            0,0,0,0,0,0,0,0, //|        
        ],
        ['0'] =
        [
            0,1,1,1,1,1,1,0, //| 111111 
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,0, //| 111111 
        ],
        ['1'] =
        [
            0,0,0,1,1,0,0,0, //|   11   
            0,1,1,1,1,0,0,0, //| 1111   
            1,1,0,1,1,0,0,0, //|11 11   
            0,0,0,1,1,0,0,0, //|   11   
            0,0,0,1,1,0,0,0, //|   11   
            0,0,0,1,1,0,0,0, //|   11   
            1,1,1,1,1,1,1,1, //|11111111
        ],
        ['2'] =
        [
            0,1,1,1,1,1,1,0, //| 111111 
            1,1,0,0,0,0,1,1, //|11    11
            0,0,0,0,0,1,1,0, //|     11 
            0,0,0,0,1,1,0,0, //|    11  
            0,0,1,1,0,0,0,0, //|  11    
            1,1,0,0,0,0,0,0, //|11      
            1,1,1,1,1,1,1,1, //|11111111
        ],
        ['3'] =
        [
            0,1,1,1,1,1,1,0, //| 111111 
            1,1,0,0,0,0,1,1, //|11    11
            0,0,0,0,0,1,1,1, //|     111
            0,0,0,0,1,1,1,0, //|    111
            0,0,0,0,0,1,1,1, //|     111
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,0, //| 111111 
        ],
        ['4'] =
        [
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,1, //| 1111111
            0,0,0,0,0,0,1,1, //|      11
            0,0,0,0,0,0,1,1, //|      11
            0,0,0,0,0,0,1,1, //|      11
        ],
        ['5'] =
        [
            1,1,1,1,1,1,1,1, //|11111111
            1,1,0,0,0,0,0,0, //|11      
            1,1,1,1,1,1,1,0, //|1111111 
            1,1,1,0,0,0,1,1, //|111   11
            0,0,0,0,0,0,1,1, //|      11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,0, //| 111111 
        ],
        ['6'] =
        [
            0,1,1,1,1,1,1,0, //| 111111 
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,0,0, //|11      
            1,1,1,1,1,1,1,0, //|1111111 
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,0, //| 111111 
        ],
        ['7'] =
        [
            1,1,1,1,1,1,1,1, //|11111111
            0,0,0,0,0,0,1,1, //|      11
            0,0,0,0,0,1,1,0, //|     11 
            0,0,0,0,1,1,0,0, //|    11  
            0,0,0,1,1,0,0,0, //|   11   
            0,0,1,1,0,0,0,0, //|  11    
            0,1,1,0,0,0,0,0, //| 11     
        ],
        ['8'] =
        [
            0,1,1,1,1,1,1,0, //| 111111 
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,0, //| 111111 
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,0, //| 111111 
        ],
        ['9'] =
        [
            0,1,1,1,1,1,1,0, //| 111111 
            1,1,0,0,0,0,1,1, //|11    11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,1, //| 1111111
            0,0,0,0,0,0,1,1, //|      11
            1,1,0,0,0,0,1,1, //|11    11
            0,1,1,1,1,1,1,0, //| 111111 
        ],
    };

    private static int[] AddBinaryNoise(int[] Data, Random? rnd = null)
    {
        rnd ??= new();
        var result = Data.CloneObject();
        var index  = rnd.Next(Data.Length);
        result[index] = Data[index] > 0 ? 0 : 1;
        return result;
    }

    private static int[] AddBinaryNoise(int[] Data, int DistortionsCount, Random? rnd = null)
    {
        switch (DistortionsCount)
        {
            case 0: return Data;
            case 1: return AddBinaryNoise(Data, rnd);
        }

        var result = Data.CloneObject();
        rnd ??= new Random();

        if (DistortionsCount < result.Length)
        {
            var indexes = Enumerable.Range(0, Data.Length).ToArray().Mix(rnd).Take(DistortionsCount).ToArray();
            foreach (var index in indexes)
                result[index] = Data[index] > 0 ? 0 : 1;
        }
        else
            foreach (var index in rnd.SequenceInt(0, Data.Length, DistortionsCount))
                result[index] = Data[index] > 0 ? 0 : 1;

        return result;
    }

    private static int[][] GetDigitSymbolsImages(Dictionary<char, int[]>? Symbols = null) => (Symbols ?? __Symbols)
       .Where(v => char.IsDigit(v.Key))
       .ToDictionary(v => (int)char.GetNumericValue(v.Key), v => v.Value)
       .OrderBy(v => v.Key)
       .Aggregate(
            new int[10][],
            (S, v) =>
            {
                var (i, value) = v;
                S[i]           = value;
                return S;
            });

    private static (NeuralProcessor<int[], int> Processor, double Error) GetProcessor(Dictionary<char, int[]>? Symbols = null, int MaxEpochCount = 5000)
    {
        var chars = GetDigitSymbolsImages(Symbols);

        var       char_length         = chars[0].Length;
        var       chars_count         = chars.Length;
        const int hidden_layer_length = 15;
        var processor = new NeuralProcessor<int[], int>(
            Network: new MultilayerPerceptron(InputsCount: char_length, NeuronsCount: new[] { hidden_layer_length, chars_count }),
            InputFormatter: (vv, inputs) => vv.Foreach((v, i) => inputs[i] = v),
            OutputFormatter: outputs => outputs.GetMaxIndex());

        var processor_teacher = processor.CreateTeacher(BackOutputFormatter: (index, outputs) => outputs[index] = 1);

        var epoch_errors = Enumerable.Range(0, MaxEpochCount)
           .Select(_ => chars.Select((vv, i) => processor_teacher.Teach(vv, i)).Max())
           .TakeWhile(error => error > 0.001)
           .ToArray();

        epoch_errors = [epoch_errors[0], epoch_errors[^1]];

        var last_error = epoch_errors[^1];
        Assert.That.Value(epoch_errors[0]).GreaterThan(last_error);

        Debug.WriteLine("Ошибка обучения процессора составила {0}", last_error.RoundAdaptive(3));
        return (processor, last_error);
    }

    [TestMethod]
    public void DigitsRecognitionTest()
    {
        var processor = GetProcessor().Processor;

        var chars = GetDigitSymbolsImages();

        var results = chars.ToArray(processor.Process);

        var expected_results = Enumerable.Range(0, 10).ToArray();
        CollectionAssert.That.Collection(results).IsEqualTo(expected_results);
    }

    private static int[] GetDiffIndexes(int[] V1, int[] V2)
    {
        if (V1.NotNull().Length != V2.NotNull().Length) throw new InvalidOperationException("Размеры массивов не совпадают");
        if (V1.Length == 0) return Array.Empty<int>();

        var result = new List<int>(V1.Length);

        for (var i = 0; i < V1.Length; i++)
            if (V1[i] != V2[i])
                result.Add(i);

        return result.ToArray();
    }

    [TestMethod]
    public void DigitsRecognition_With_DistortionsCount_1_Test()
    {
        var processor = GetProcessor().Processor;

        var rnd     = new Random();
        var chars   = GetDigitSymbolsImages();
        var results = chars.ToArray(processor.Process);

        var diff_results = new int[1000][];
        for (var i = 0; i < 1000; i++)
        {
            var noisy_chars   = chars.ToArray(c => AddBinaryNoise(c, rnd));
            var noisy_results = noisy_chars.ToArray(processor.Process);

            diff_results[i] = GetDiffIndexes(results, noisy_results);
        }

        var average_error = diff_results.Average(errors => errors.Length);
        Debug.WriteLine("Средняя ошибка распознавания символов составила {0:p2}", average_error);
        const double error_threshold = 0.15;
        Assert.That.Value(average_error).LessThan(error_threshold);
    }

    [DataTestMethod]
    [DataRow(2, 0.25)]
    [DataRow(3, 0.35)]
    public void DigitsRecognition_With_Distortions_Test(int DistortionsCount, double ErrorThreshold)
    {
        var processor = GetProcessor().Processor;

        var rnd     = new Random();
        var chars   = GetDigitSymbolsImages();
        var results = chars.ToArray(processor.Process);

        var diff_results = new int[1000][];
        for (var i = 0; i < 1000; i++)
        {
            var noisy_chars   = chars.ToArray(c => AddBinaryNoise(c, DistortionsCount, rnd));
            var noisy_results = noisy_chars.ToArray(processor.Process);

            diff_results[i] = GetDiffIndexes(results, noisy_results);
        }

        var average_error = diff_results.Average(errors => errors.Length);
        Debug.WriteLine("Средняя ошибка распознавания символов при {0} искажениях составила {1:p2}",
            DistortionsCount, average_error);
        Assert.That.Value(average_error).LessThan(ErrorThreshold);
    }
}