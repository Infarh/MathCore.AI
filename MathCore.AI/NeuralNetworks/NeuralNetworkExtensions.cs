﻿using System;
using System.CodeDom;
using System.Collections.Generic;
using System.Linq;
using MathCore.Annotations;

namespace MathCore.AI.NeuralNetworks
{
    public static class NeuralNetworkExtensions
    {
        [NotNull]
        public static Epoch TeachRandom(
            [NotNull] this NetworkTeacher Teacher,
            double Rho,
            [NotNull, ItemNotNull] Example[] Examples,
            [CanBeNull] Random rnd = null)
        {
            if (Teacher is null) throw new ArgumentNullException(nameof(Teacher));
            if (Examples is null) throw new ArgumentNullException(nameof(Examples));

            return Teacher.Teach(Rho, Examples.AsRandomEnumerable(rnd));
        }

        /// <summary>Выполнить обучение нейронной сети (одну эпоху) по заданному набору обучающих примеров</summary>
        /// <param name="Teacher">Обучаемая нейронная сеть</param>
        /// <param name="Rho">Коэффициент скорости обучения</param>
        /// <param name="Examples">Набор обучающих примеро</param>
        /// <returns>Результат выполнения обучения за эпоху</returns>
        [NotNull]
        public static Epoch Teach(
            [NotNull] this NetworkTeacher Teacher,
            double Rho,
            [NotNull, ItemNotNull] params Example[] Examples)
        {
            if (Teacher is null) throw new ArgumentNullException(nameof(Teacher));
            if (Examples is null) throw new ArgumentNullException(nameof(Examples));
            if (Teacher.Network.InputsCount == 0) throw new InvalidOperationException("Сеть не имеет входов");
            if (Teacher.Network.OutputsCount == 0) throw new InvalidOperationException("Сеть не имеет выходов");
            if (double.IsNaN(Rho)) throw new ArgumentException("Коэффициент скорости обучения - не-число!", nameof(Rho));
            if (Rho <= 0) throw new ArgumentOutOfRangeException(nameof(Rho), "Коэффициент скорости обучения должен быть больше 0");

            var inputs_count = Teacher.Network.InputsCount;
            var outputs_count = Teacher.Network.OutputsCount;
            for (var i = 0; i < Examples.Length; i++)
            {
                var example = Examples[i];
                if (example is null)
                    throw new InvalidOperationException($"Обучающий пример с индексом {i} отсутствует");
                if (example.Input.Length != inputs_count)
                    throw new InvalidOperationException($"Длина входного вектора примера №{i} ({example.Input.Length}) не равна количеству входов сети ({inputs_count})");
                if (example.ExpectedOutput.Length != outputs_count)
                    throw new InvalidOperationException($"Длина вектора ожидаемого результата №{i} ({example.ExpectedOutput.Length}) не равна количеству выходов сети ({outputs_count})");
            }

            return Teach(Teacher, Rho, (IEnumerable<Example>)Examples);
        }

        [NotNull]
        private static Epoch Teach(
            [NotNull] this NetworkTeacher Teacher,
            double Rho,
            [NotNull, ItemNotNull] IEnumerable<Example> Examples)
        {
            var outputs_count = Teacher.Network.OutputsCount;
            var examples = Examples as Example[] ?? Examples.ToArray();
            var results = new TeachResult[examples.Length];
            var max_error = 0d;
            var avg_error = 0d;
            var output = new double[outputs_count];
            for (var i = 0; i < examples.Length; i++)
            {
                var example = examples[i];
                var example_input = example.Input;
                var example_expected_output = example.ExpectedOutput;
                var error = Teacher.Teach(example_input, output, example_expected_output, Rho);
                if (error > max_error) max_error = error;
                avg_error += error;
                if (double.IsNaN(error)) throw new InvalidOperationException("Ошибка сети является \"не числом\" - возможно сеть нестабвльна.");
                if (double.IsInfinity(error)) throw new InvalidOperationException("Ошибка сети является \"бесконечностью\" - возможно сеть нестабильна.");
                results[i] = new TeachResult(example, output, error);
            }

            return new Epoch(results, max_error, avg_error / examples.Length);
        }

        [NotNull]
        public static double[] Process([NotNull] this INeuralNetwork Network, [NotNull] double[] Input)
        {
            if (Network is null) throw new ArgumentNullException(nameof(Network));
            if (Input is null) throw new ArgumentNullException(nameof(Input));

            var result = new double[Network.OutputsCount];
            Network.Process(Input, result);
            return result;
        }

        public static void Process(
            [NotNull] this INeuralNetwork Network,
            [NotNull] double[] Input,
            [NotNull] double[] Output,
            [NotNull] double[] ExpectedOutput,
            [NotNull] double[] Error)
        {
            Network.Process(Input, Output);

            for (var i = 0; i < Output.Length; i++)
            {
                var delta = ExpectedOutput[i] - Output[i];
                Error[i] = 0.5 * delta * delta;
            }
        }

        [NotNull]
        public static double[] Process(
            [NotNull] this INeuralNetwork Network,
            [NotNull] double[] Input,
            [NotNull] double[] Output,
            [NotNull] double[] ExpectedOutput
            )
        {
            var error = new double[Output.Length];
            Network.Process(Input, Output, ExpectedOutput, error);
            return error;
        }
    }
}