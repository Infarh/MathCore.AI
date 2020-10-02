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

            //var network = new MultilayerPerceptron(10, 5, 7)
            //{
            //    LayerInput = { Activation = ActivationFunction.GetLinear(5) },
            //    LayerOutput = { Activation = ActivationFunction.Linear },
            //    Layer =
            //    {
            //        [1] = { Activation = ActivationFunction.Th }
            //    }

            //};

            //var processor = new NeuralProcessor<double, double>(network, (v, vv) => vv[0] = v, vv => vv[0]);

            //Console.WriteLine(processor.Process(Math.PI));

            
            double[] Wa =           // Коэффициенты передачи слоя активации по входу
            {
                0.45,
                0.25
            };

            double[] Ua = { 0.15 }; // Коэффициенты передачи слоя активации по входу для предыдущего выхода
            double[] ba = { 0.2 };  // Коэффициенты смещения слоя активации по входу

            double[] Wi =           // Коэффициенты передачи слоя вентелей входа
            {
                0.95,
                0.8
            };

            double[] Ui = { 0.8 };
            double[] bi = { 0.65 };

            double[] Wf =
            {
                0.7,
                0.45
            };

            double[] Uf = { 0.1 };
            double[] bf = { 0.15 };

            double[] Wo =
            {
                0.6,
                0.4
            };

            double[] Uo = { 0.25 };
            double[] bo = { 0.1 };

            Console.ReadLine();
        }
    }
}
