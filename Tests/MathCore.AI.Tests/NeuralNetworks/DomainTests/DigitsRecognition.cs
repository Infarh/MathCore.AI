using System;
using System.Collections.Generic;
using System.Linq;
using MathCore.AI.NeuralNetworks;
using MathCore.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;
// ReSharper disable ArgumentsStyleAnonymousFunction
// ReSharper disable ArgumentsStyleNamedExpression
// ReSharper disable ArgumentsStyleLiteral
// ReSharper disable ArgumentsStyleOther

namespace MathCore.AI.Tests.NeuralNetworks.DomainTests
{
    [TestClass]
    public class DigitsRecognition
    {
        private static readonly Dictionary<char, int[]> __Symbols = new Dictionary<char, int[]>
        {
            ['\0'] = new[]
            {
                0,0,0,0,0, //|     
                0,0,0,0,0, //|     
                0,0,0,0,0, //|     
                0,0,0,0,0, //|     
                0,0,0,0,0, //|     
                0,0,0,0,0, //|     
                0,0,0,0,0, //|     
            },
            ['0'] = new[]
            {
                0,1,1,1,0, //| 111 
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                0,1,1,1,0, //|1   1
                0,1,1,1,0, //| 111 
            },
            ['1'] = new[]
            {
                0,0,1,0,0, //|  1  
                0,1,1,0,0, //| 11  
                1,0,1,0,0, //|1 1  
                0,0,1,0,0, //|  1  
                0,0,1,0,0, //|  1  
                0,0,1,0,0, //|  1  
                1,1,1,1,1, //|11111
            },
            ['2'] = new[]
            {
                0,1,1,1,0, //| 111 
                1,0,0,0,1, //|1   1
                0,0,0,0,1, //|    1
                0,0,0,1,0, //|   1 
                0,0,1,0,0, //|  1  
                0,1,0,0,0, //| 1   
                1,1,1,1,1, //|11111
            },
            ['3'] = new[]
            {
                0,1,1,1,0, //| 111 
                1,0,0,0,1, //|1   1
                0,0,0,0,1, //|    1
                0,0,0,1,0, //|   1 
                0,0,0,0,1, //|    1
                1,0,0,0,1, //|1   1
                0,1,1,1,0, //| 111 
            },
            ['4'] = new[]
            {
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                1,1,1,1,1, //|11111
                0,0,0,0,1, //|    1
                0,0,0,0,1, //|    1
                0,0,0,0,1, //|    1
            },
            ['5'] = new[]
            {
                1,1,1,1,1, //|11111
                1,0,0,0,0, //|1    
                1,1,1,1,0, //|1111 
                1,0,0,0,1, //|1   1
                0,0,0,0,1, //|    1
                1,0,0,0,1, //|1   1
                0,1,1,1,0, //| 111 
            },
            ['6'] = new[]
            {
                0,1,1,1,0, //| 111 
                1,0,0,0,1, //|1   1
                1,0,0,0,0, //|1    
                1,1,1,1,0, //|1111 
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                0,1,1,1,0, //| 111 
            },
            ['7'] = new[]
            {
                1,1,1,1,1, //|11111
                0,0,0,0,1, //|    1
                0,0,0,1,0, //|   1 
                0,0,1,0,0, //|  1  
                0,1,0,0,0, //| 1   
                1,0,0,0,0, //|1    
                1,0,0,0,0, //|1    
            },
            ['8'] = new[]
            {
                0,1,1,1,0, //| 111 
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                0,1,1,1,0, //| 111 
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                0,1,1,1,0, //| 111 
            },
            ['9'] = new[]
            {
                0,1,1,1,0, //| 111 
                1,0,0,0,1, //|1   1
                1,0,0,0,1, //|1   1
                0,1,1,1,1, //| 1111
                0,0,0,0,1, //|    1
                1,0,0,0,1, //|1   1
                0,1,1,1,0, //| 111 
            },
        };

        private static int[][] GetDigitSymbolsImages() => __Symbols
           .Where(v => char.IsDigit(v.Key))
           .ToDictionary(v => (int)char.GetNumericValue(v.Key), v => v.Value)
           .Aggregate(
                new int[10][],
                (S, v) =>
                {
                    var (i, value) = v;
                    S[i] = value;
                    return S;
                });

        [NotNull]
        private static NeuralProcessor<int[], int> GetProcessor(int MaxEpochCount = 1000)
        {
            var chars = GetDigitSymbolsImages();

            var network = new MultilayerPerceptron(InputsCount: 35, NeuronsCount: new[] { 15, 10 });

            var processor = new NeuralProcessor<int[], int>(
                Network: network,
                InputFormatter: (vv, inputs) => vv.Foreach((v, i) => inputs[i] = v),
                OutputFormatter: outputs => outputs.GetMaxIndex());

            var teacher = processor.CreateTeacher<IBackPropagationTeacher>(
                BackOutputFormatter: (v, outputs) =>
                {
                    Array.Clear(outputs, 0, outputs.Length);
                    outputs[v] = 1;
                });

            var epoch_errors = Enumerable.Range(0, MaxEpochCount)
               .Select(_ => chars.Select((vv, i) => teacher.Teach(vv, i)).Max())
               .TakeWhile(error => error > 0.01)
               .ToArray();

            Assert.That.Value(epoch_errors[0]).GreaterThan(epoch_errors[^1]);

            return processor;
        }

        [TestMethod]
        public void DigitsRecognitionTest()
        {
            var processor = GetProcessor();

            var chars = GetDigitSymbolsImages();

            var results = chars.ToArray(processor.Process);

            var expected_results = Enumerable.Range(0, 10).ToArray();
            CollectionAssert.That.Collection(results).IsEqualTo(expected_results);
        }


    }
}
