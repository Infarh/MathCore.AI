using System;
using MathCore.AI.NeuralNetworks;
using MathCore.AI.NeuralNetworks.ActivationFunctions;

namespace MathCore.AI
{
    internal static class Program
    {
        private static void Main()
        {
            Console.WriteLine();

            var network = new MultilayerPerceptron(10, 5, 7)
            {
                LayerInput = { Activation = ActivationFunction.GetLinear(5) },
                LayerOutput = { Activation = ActivationFunction.Linear },
                Layer =
                {
                    [1] = { Activation = ActivationFunction.Th }
                }

            };

            var processor = new NeuralProcessor<double, double>(network, (v, vv) => vv[0] = v, vv => vv[0]);

            Console.WriteLine(processor.Process(Math.PI));

            Console.ReadLine();
        }
    }
}
