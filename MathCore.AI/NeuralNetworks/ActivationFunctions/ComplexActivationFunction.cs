﻿// ReSharper disable UnusedParameter.Global
// ReSharper disable UnusedMember.Global
// ReSharper disable InconsistentNaming
namespace MathCore.AI.NeuralNetworks.ActivationFunctions;

/// <summary>Комплексные экспоненциальные функции</summary>
public abstract class ComplexActivationFunction
{
    /// <summary>Комплексная экспонента</summary>
    public static ComplexActivationFunction Exponent => new ComplexExponent();

    /// <summary>Значение функции активации</summary>
    /// <param name="x">Взвешенная сумма входов нейронов</param>
    /// <returns>Значение выхода нейрона</returns>
    public abstract Complex Value(Complex x);

    /// <summary>Производная функции активации</summary>
    /// <param name="z">Значение на выходе нейрона</param>
    /// <returns>Значение производной функции активации</returns>
    public abstract Complex dValue(Complex z); 
}