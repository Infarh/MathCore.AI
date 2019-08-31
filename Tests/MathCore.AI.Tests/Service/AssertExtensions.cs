// ReSharper disable once CheckNamespace

using MathCore.Annotations;

namespace Microsoft.VisualStudio.TestTools.UnitTesting
{
    internal static class AssertExtensions
    {
        [NotNull] public static AssertEqualsChecker<T> Value<T>(this Assert that, T value) => new AssertEqualsChecker<T>(value);
        [NotNull] public static AssertDoubleEqualsChecker Value(this Assert that, double value) => new AssertDoubleEqualsChecker(value);
        [NotNull] public static AssertIntEqualsChecker Value(this Assert that, int value) => new AssertIntEqualsChecker(value);
    }
}
