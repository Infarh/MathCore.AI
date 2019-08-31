using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using MathCore.AI.NeuralNetworks;
using MathCore.AI.NeuralNetworks.ActivationFunctions;
using MathCore.AI.Tests.Service;
using MathCore.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MathCore.AI.Tests.NeuralNetworks
{
    [TestClass]
    public class MultilayerPerceptron_Tests
    {
        private static void ProcessLayer(
            [NotNull] double[] Input,
            [NotNull] double[,] W,
            [NotNull] double[] NeuronOffsets,
            [NotNull] double[] OffsetsWeights,
            [NotNull] double[] Output)
        {
            for (var output_index = 0; output_index < Output.Length; output_index++)
            {
                Output[output_index] = 0;
                for (var input_index = 0; input_index < Input.Length; input_index++)
                    Output[output_index] += W[output_index, input_index] * Input[input_index];

                Output[output_index] += NeuronOffsets[output_index] * OffsetsWeights[output_index];
            }
        }

        private static double activation(double x) => 1 / (1 + Math.Exp(-x));

        private static void Activation([NotNull] double[] X, [NotNull] double[] FX)
        {
            for (var i = 0; i < X.Length; i++)
                FX[i] = activation(X[i]);
        }

        private static void DirectDistribution(
            [NotNull] double[][] Inputs,
            [NotNull] double[][,] Layers,
            [NotNull] double[][] Offsets,
            [NotNull] double[][] OffsetsWeights,
            [NotNull] double[][] Outputs,
            [NotNull] double[] NetworkOutput)
        {
            var layer_index = -1;
            do
            {
                if (layer_index++ == 0)
                    Activation(Outputs[layer_index - 1], Inputs[layer_index]);

                ProcessLayer(
                    Inputs[layer_index],
                    Layers[layer_index],
                    Offsets[layer_index],
                    OffsetsWeights[layer_index],
                    Outputs[layer_index]);
            } while (layer_index < Layers.Length - 1);

            Activation(Outputs[layer_index], NetworkOutput);
        }

        [TestMethod]
        public void NeuralNetwork_Integral_Test()
        {

            double activation_inverse(double x) => x * (1 - x);

            double[,] W0 =
            {
                {  1.0, 0.5 },
                { -1.0, 2.0 }
            };
            double[] Offsets0 = { 1, 1 };
            double[] OffsetsW0 = { 1, 1 };

            double[,] W1 =
            {
                { 1.5, -1.0 }
            };
            double[] Offsets1 = { 1 };
            double[] OffsetsW1 = { 1 };


            double[][,] layers = { (double[,])W0.Clone(), (double[,])W1.Clone() };
            double[][] Offsets = { Offsets0.CloneObject(), Offsets1.CloneObject() };
            double[][] OffsetsW = { OffsetsW0.CloneObject(), OffsetsW1.CloneObject() };

            double[] network_input = { 0, 1 };

            double[][] inputs =
            {
                network_input,
                new double[2],    // создаём массив из 2 чисел
            };

            double[][] outputs =
            {
                new double[2],
                new double[1]
            };

            var errors = new double[outputs.Length][];
            for (var i = 0; i < errors.Length; i++)
                errors[i] = new double[outputs[i].Length];

            var network_output = new double[1];

            // прямое распространение


            DirectDistribution(inputs, layers, Offsets, OffsetsW, outputs, network_output);

            CollectionAssert.That.Collection(outputs[0]).AreEqualValues(1.5, 3);
            CollectionAssert.That.Collection(inputs[1]).AreEquals(new[] { 0.81757, 0.952574 }, 4.48e-6);
            CollectionAssert.That.Collection(outputs[1]).AreEquals(new[] { 1.2738 }, 1.25e-5);
            CollectionAssert.That.Collection(network_output).AreEquals(new[] { 0.78139 }, 4.31e-7);

            double[] correct_output = { 1 };

            var error = 0d;
            for (var i = 0; i < network_output.Length; i++)
                error += (correct_output[i] - network_output[i]).Pow2();
            error *= 0.5;

            Assert.That.Value(error).AreEqual(0.023895, 7.19e-8);

            var output_error = errors[errors.Length - 1];
            for (var i = 0; i < output_error.Length; i++)
                output_error[i] = (correct_output[i] - network_output[i]) * activation_inverse(network_output[i]);

            CollectionAssert.That.Collection(errors[errors.Length - 1]).AreEquals(new[] { 0.0373 }, 4.28e-5);

            for (var level = errors.Length - 2; level >= 0; level--)
            {
                var error_level = errors[level];
                var prev_error_level = errors[level + 1];
                var w = layers[level + 1];
                var level_inputs = inputs[level + 1];
                for (var i = 0; i < error_level.Length; i++)
                {
                    var err = 0d;
                    for (var j = 0; j < prev_error_level.Length; j++)
                        err += prev_error_level[j] * w[j, i];
                    error_level[i] = err * activation_inverse(level_inputs[i]);
                }
            }

            CollectionAssert.That.Collection(errors[0]).AreEquals(new[] { 0.0083449, -0.0016851 }, 9.42e-6);

            var rho = 0.5;
            for (var level = 0; level < layers.Length; level++)
            {
                var w = layers[level];
                var layer_offset = Offsets[level];
                var err = errors[level];
                var level_inputs = inputs[level];
                var outputs_count = w.GetLength(0);
                var inputs_count = w.GetLength(1);
                for (var i = 0; i < outputs_count; i++)
                {
                    for (var j = 0; j < inputs_count; j++)
                        w[i, j] += rho * err[i] * level_inputs[j];

                    layer_offset[i] += rho * err[i];
                }
            }

            var expected_w = new[]
            {
                new [,]
                {
                    {  1, 0.50417 },
                    { -1, 1.99916 }
                },
                new [,]
                {
                    { 1.51525, -0.9882 }
                }
            };

            double[][] expected_offsets =
            {
                new [] { 1.00417,  0.99915 },
                new [] { 1.01865 }
            };

            for (var level = 0; level < layers.Length; level++)
            {
                CollectionAssert.That.Collection(layers[level]).AreEquals(expected_w[level], 6e-3);
                CollectionAssert.That.Collection(Offsets[level]).AreEquals(expected_offsets[level], 2.15e-5);
            }

            DirectDistribution(inputs, layers, Offsets, OffsetsW, outputs, network_output);

            CollectionAssert.That.Collection(network_output).AreEquals(new[] { 0.7898 }, 1.99e-5);

            error = 0d;
            for (var i = 0; i < network_output.Length; i++)
                error += (correct_output[i] - network_output[i]).Pow2();
            error *= 0.5;

            Assert.That.Value(error).AreEqual(0.022, 8.8e-5);

            double[][,] layers2 = { (double[,])W0.Clone(), (double[,])W1.Clone() };
            var network = new MultilayerPerceptron(layers2);

            var network_output2 = new double[1];
            var teacher = network.CreateTeacher();
            error = teacher.Teach(network_input, network_output2, correct_output);
            CollectionAssert.That.Collection(network_output2).AreEquals(new[] { 0.78139 }, 4.31e-7);
            Assert.That.Value(error).AreEqual(0.023895, 7.19e-8);
        }

        [TestMethod]
        public void ThreeLayersNetworkCreation_Test()
        {
            double[][,] network_structure =
            {
                new double[,]
                {
                    { 1, 1, 1 },
                    { 1, 1, 1 }
                },
                new double[,]
                {
                    { 1, 1 },
                    { 1, 1 },
                    { 1, 1 }
                },
                new double[,]
                {
                    { 1, 1, 1 }
                }
            };

            var network = new MultilayerPerceptron(network_structure);

            CheckNetwork(network, network_structure);
        }

        // ReSharper disable once ObjectCreationAsStatement
        [TestMethod, ExpectedException(typeof(ArgumentException))]
        public void ZerroLayersNetworkCreation_Test() => new MultilayerPerceptron();

        [TestMethod]
        public void InconsistentLayersInputsCount_Test()
        {
            Assert.ThrowsException<FormatException>(() =>
                new MultilayerPerceptron(
                    new double[,]
                    {
                        { 1, 1, 1 },
                        { 1, 1, 1 }
                    },
                    new double[,]
                    {
                        { 1, 1 },
                        { 1, 1 },
                        { 1, 1 }
                    },
                    new double[,]
                    {
                        { 1, 1 }
                    }));
        }

        [TestMethod]
        public void RandomWeightCreation_Test()
        {
            const int inputs_count = 5;
            const int outputs_count = 2;

            var rnd = new Random();
            int[] neurons_counts = { 5, 7, outputs_count };

            var network = new MultilayerPerceptron(inputs_count, neurons_counts, rnd);

            Assert.That.Value(network.LayersCount).AreEqual(neurons_counts.Length);
            Assert.That.Value(network.InputsCount).AreEqual(inputs_count);
            Assert.That.Value(network.OutputsCount).AreEqual(outputs_count);
        }

        private static void CheckNetwork([NotNull] MultilayerPerceptron Network, [NotNull] double[][,] W)
        {
            if (Network is null) throw new ArgumentNullException(nameof(Network));
            if (W is null) throw new ArgumentNullException(nameof(W));

            var layers_count = W.Length;

            Assert.That.Value(Network.LayersCount).AreEqual(layers_count);                       // Число слоёв должно быть равно числу матрицы коэффициентов
            Assert.That.Value(Network.InputsCount).AreEqual(W[0].GetLength(1));                  // Число входов сети - число столбцов первой матрицы
            Assert.That.Value(Network.OutputsCount).AreEqual(W[layers_count - 1].GetLength(0));  // Число выходов сети - число строк последней матрицы

            Assert.That.Value(Network.Offests.Count).AreEqual(layers_count);                     // Количество векторов смещений равно числу слоёв сети
            Assert.That.Value(Network.OffsetWeights.Count).AreEqual(layers_count);               // Число векторов весовых коэффициентов смещений слоёв равно числу слоёв

            for (var i = 0; i < layers_count; i++)
            {
                Assert.That.Value(Network.Offests[i].Length).AreEqual(W[i].GetLength(0));        // Размер вектора смещений равен числу строк первой матрицы (числу нейронов входного слоя)
                CollectionAssert.That.Collection(Network.Offests[i]).AllEquals(1);               // По умолчанию смещения равны 1
                Assert.That.Value(Network.OffsetWeights[i].Length).AreEqual(W[i].GetLength(0));  // Размер вектора весовых коэффициентов смещений первого слоя равно числу строк матрицы первого слоя (числу нейронов)
                CollectionAssert.That.Collection(Network.OffsetWeights[i]).AllEquals(1);         // По умолчанию веса векторов смещений равны 1
            }

            Assert.That.Value(Network.HiddentOutputs.Count).AreEqual(layers_count - 1);
            for (var i = 0; i < layers_count - 1; i++)
            {
                var hidden_inputs = Network.HiddentOutputs[i];
                Assert.That.Value(hidden_inputs.Length).AreEqual(W[i].GetLength(0));
                CollectionAssert.That.Collection(hidden_inputs).AllEquals(0);
            }
        }

        /// <summary>Структура тестовой сети</summary>
        /// <remarks>
        ///         1.0 ┌──────┐
        ///  0 ----->>>>│ f(u) │>-
        ///     \ / 0.5 └──────┘  \ 1.5 ┌──────┐
        ///      Х                 ->>>>│ f(u) │>--- 1
        ///     / \-1.0 ┌──────┐  /-1.0 └──────┘
        ///  1 ----->>>>│ f(u) │>-
        ///         2.0 └──────┘
        ///
        ///  f(x) = 1 / (1 + e^-x)
        ///  df(x)/dx = x * (1 - x)
        /// </remarks>
        [NotNull]
        private static double[][,] GetNetworkStructure() => new[]
        {
            new[,] // Матрица коэффициентов передачи первого слоя
            {
                {1.0, 0.5}, // Число столбцов - число входов слоя (входов сети)
                {-1.0, 2.0} // Число строк - число нейронов слоя (число выходов слоя)
            },
            new[,] // Матрица коэффициентов передачи второго слоя
            {
                {1.5, -1.0} // В выходном слое один нейрон и два входа
            }
        };

        [TestMethod]
        public void Processing_Test()
        {
            var network_structure = GetNetworkStructure();
            var network = new MultilayerPerceptron(network_structure);
            CheckNetwork(network, network_structure);

            double[] input = { 0, 1 };                                                    // Входное воздействие
            double[] output = { 0 };                                                      // Вектор отклика сети
            double[] expected_output = { 1 };                                             // Ожидаемое значение оклика сети для процесса обучения

            const double rho = 0.5;
            var teacher = network.CreateTeacher();
            var error = teacher.Teach(input, output, expected_output, rho);               // Обработка входного воздействия сетью

            CollectionAssert.That.Collection(output).AreEqualValues(0.78139043094733129); // Проверка отклика сети
            Assert.That.Value(error).AreEqual(0.023895071840696763);                      // Проверка вычисленного значения ошибки обработки

            Assert.That.Value(network[0]).AreEqual(network_structure[0]);
            CollectionAssert.That.Collection(network[0]).AreEquals(new[,]
            {
                {  1, 0.50417715523146878 },
                { -1, 1.9991564893972078 }
            });
            Assert.That.Value(network[1]).AreEqual(network_structure[1]);
            CollectionAssert.That.Collection(network[1]).AreEquals(new[,]
            {
                { 1.5152652441182986, -0.982214126039723 }
            });

            CollectionAssert.That.Collection(network.Offests[0]).AreEqualValues(1, 1);
            CollectionAssert.That.Collection(network.Offests[1]).AreEqualValues(1);
            CollectionAssert.That.Collection(network.OffsetWeights[0]).AreEqualValues(1.0041771552314689, 0.99915648939720769);
            CollectionAssert.That.Collection(network.OffsetWeights[1]).AreEqualValues(1.0186713804831196);
        }

        [TestMethod]
        public void Process_Test()
        {
            var network_structure = GetNetworkStructure();
            var network = new MultilayerPerceptron(network_structure);

            CheckNetwork(network, network_structure);

            double[] input = { 0, 1 }; // Входное воздействие
            double[] output = { 0 };   // Вектор отклика сети

            network.Process(input, output);

            CollectionAssert.That.Collection(output).AreEqualValues(0.78139043094733129);
            CollectionAssert.That.Collection(network.HiddentOutputs[0]).AreEqualValues(0.81757447619364365, 0.95257412682243336);
        }

        [TestMethod]
        public void ProcessWithReturn_Test()
        {
            var network_structure = GetNetworkStructure();
            var network = new MultilayerPerceptron(network_structure);
            CheckNetwork(network, network_structure);

            double[] input = { 0, 1 };           // Входное воздействие

            var output = network.Process(input);

            CollectionAssert.That.Collection(output).AreEqualValues(0.78139043094733129);
            CollectionAssert.That.Collection(network.HiddentOutputs[0]).AreEqualValues(0.81757447619364365, 0.95257412682243336);
        }

        [TestMethod]
        public void ProcessWithErrorCheck_Test()
        {
            var network_structure = GetNetworkStructure();
            var network = new MultilayerPerceptron(network_structure);
            CheckNetwork(network, network_structure);

            double[] input = { 0, 1 };           // Входное воздействие
            double[] output = { 0 };             // Вектор отклика сети
            double[] expected_output = { 1 };    // Ожидаемое значение оклика сети для процесса обучения
            double[] errors = { 0 };

            network.Process(input, output, expected_output, errors);

            CollectionAssert.That.Collection(output).AreEqualValues(0.78139043094733129);
            CollectionAssert.That.Collection(network.HiddentOutputs[0]).AreEqualValues(0.81757447619364365, 0.95257412682243336);
            CollectionAssert.That.Collection(errors).AreEqualValues(0.023895071840696763);
        }

        [TestMethod]
        public void ProcessWithTeach_Test()
        {
            var network_structure = GetNetworkStructure();
            var network = new MultilayerPerceptron(network_structure);
            CheckNetwork(network, network_structure);

            double[] input = { 0, 1 };           // Входное воздействие
            double[] output = { 0 };             // Вектор отклика сети
            double[] expected_output = { 1 };    // Ожидаемое значение оклика сети для процесса обучения
            double[] errors = { 0 };

            network.Process(input, output, expected_output, errors);

            CollectionAssert.That.Collection(output).AreEqualValues(0.78139043094733129);
            CollectionAssert.That.Collection(network.HiddentOutputs[0]).AreEqualValues(0.81757447619364365, 0.95257412682243336);
            CollectionAssert.That.Collection(errors).AreEqualValues(0.023895071840696763);

            Assert.That.Value(network[0]).AreEqual(network_structure[0]);

            const double rho = 0.5;
            var teacher = network.CreateTeacher();
            teacher.Teach(input, output, expected_output, rho);

            CollectionAssert.That.Collection(network[0]).AreEquals(new[,]
            {
                {  1, 0.50417715523146878 },
                { -1, 1.9991564893972078 }
            });
            Assert.That.Value(network[1]).AreEqual(network_structure[1]);
            CollectionAssert.That.Collection(network[1]).AreEquals(new[,]
            {
                { 1.5152652441182986, -0.982214126039723 }
            });
        }

        [TestMethod, SuppressMessage("ReSharper", "PossibleMultipleEnumeration")]
        public void NeuralController_Test()
        {
            var rnd = new Random();
            var controller = new MultilayerPerceptron(4, new[] { 3, 4 }, (Layer, Neuron, Input) => rnd.NextDouble() - 0.5);
            //controller.Layer.Foreach(Layer => Layer.SetOffsets());

            Assert.That.Value(controller.InputsCount).AreEqual(4);
            Assert.That.Value(controller.OutputsCount).AreEqual(4);

            // Состояние персонажа
            // H - Health points    (Текущий уровень здоровья)
            // K - Knifes count     (количество ножей)
            // G - Guns count       (количество пистолетов)
            // E - Enemys count     (враг поблизости - количество)
            // Варианты действий
            // A - Atack enemy      (атаковать врага!)
            // R - Run              (бежать!!!)
            // W - Wander           (бродить, слоняться, искать приключений)
            // H - Hide             (прятаться...)
            // Должен быть выбран один из вариантов действйи - максимальное значение
            Example[] examples =
            {                    // H  K  G  E            A  R  W  H
                new Example(new []{ 2, 0, 0, 0 }, new []{ 0, 0, 1, 0 }), //  0 - здоровы как бык,   оружия нет,    врагов нет - бродить
                new Example(new []{ 2, 0, 0, 1 }, new []{ 0, 0, 1, 0 }), //  1 - здоровы как бык,   оружия нет,    враг 1     - уворачиваться
                new Example(new []{ 2, 0, 1, 1 }, new []{ 0, 0, 0, 0 }), //  2 - здоровы как бык,   есть пистолет, врага нет  - уворачиваться
                new Example(new []{ 2, 0, 1, 2 }, new []{ 1, 0, 0, 0 }), //  3 - здоровы как бык,   есть пистолет, врагов 2!! - атаковать!!!
                new Example(new []{ 2, 1, 0, 2 }, new []{ 0, 0, 0, 1 }), //  4 - здоровы как бык,   есть нож,      врагов 2!! - прятаться...
                new Example(new []{ 2, 1, 0, 1 }, new []{ 1, 0, 0, 0 }), //  5 - здоровы как бык,   есть нож,      враг 1
                                                                              
                new Example(new []{ 1, 0, 0, 0 }, new []{ 0, 0, 1, 0 }), //  6 - здоровье так себе, оружия нет,    врагов нет - бродить
                new Example(new []{ 1, 0, 0, 1 }, new []{ 0, 0, 0, 1 }), //  7 - здоровье так себе, оружия нет,    враг 1     - прятаться...
                new Example(new []{ 1, 0, 1, 1 }, new []{ 1, 0, 0, 0 }), //  8 - здоровье так себе, есть пистолет, враг 1     - атаковать!!!
                new Example(new []{ 1, 0, 1, 0 }, new []{ 0, 0, 0, 1 }), //  9 - здоровье так себе, есть пистолет, врагов нет - прятаться...
                new Example(new []{ 1, 1, 0, 2 }, new []{ 0, 0, 0, 1 }), // 10 - здоровье так себе, есть нож,      врагов 2!! - прятаться...
                new Example(new []{ 1, 1, 0, 1 }, new []{ 0, 0, 0, 1 }), // 11 - здоровье так себе, есть нож,      враг 1     - прятаться...
                                                                              
                new Example(new []{ 0, 0, 0, 0 }, new []{ 0, 0, 1, 0 }), // 12 - здоровья нет...,   оружия нет,    врагов нет - бродить
                new Example(new []{ 0, 0, 0, 1 }, new []{ 0, 0, 0, 1 }), // 13 - здоровья нет...,   оружия нет,    враг 1     - прятаться...
                new Example(new []{ 0, 0, 1, 1 }, new []{ 0, 0, 0, 1 }), // 14 - здоровья нет...,   есть пистолет, враг 1     - прятаться...
                new Example(new []{ 0, 0, 1, 2 }, new []{ 0, 1, 0, 0 }), // 15 - здоровья нет...,   есть пистолет, врагов 2!! - бежать!!!
                new Example(new []{ 0, 1, 0, 2 }, new []{ 0, 1, 0, 0 }), // 16 - здоровья нет...,   есть нож,      врагов 2!! - бежать!!!
                new Example(new []{ 0, 1, 0, 1 }, new []{ 0, 0, 0, 1 }), // 17 - здоровья нет...,   есть нож,      враг 1     - прятаться...
            };

            var teacher = controller.CreateTeacher();
            var epohs = Enumerable
                .Range(0, 100_000)
                .Select(I => teacher.Teach(0.2, examples))
                .ToArray();
            var errors = epohs.Select(e => e.ErrorAverage);

            var first_errors = errors.Take(2).ToArray();
            var last_errors = errors.ToArray().TakeLast(50).ToArray();


            CollectionAssert.That.Collection(first_errors).All(v => v > 0.4);
            CollectionAssert.That.Collection(last_errors).All(v => v < 0.095);
        }

        [TestMethod]
        public void SinApproximation_Test()
        {
            var rnd = new Random();
            var network = new MultilayerPerceptron(1, new[] { 200, 50, 1 }, (Layer, Neuron, Input) => rnd.NextUniform(0.5));
            network.Layer[2].Activation = ActivationFunction.Linear;
        }
    }
}