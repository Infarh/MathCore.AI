using System.Collections.Generic;
using MathCore.Annotations;

// ReSharper disable once CheckNamespace
namespace Microsoft.VisualStudio.TestTools.UnitTesting
{
    internal static class CollectionAssertExtensions
    {
        //public static CollectionAssertChecker Collection(this CollectionAssert assert, ICollection ActualCollection) => new CollectionAssertChecker(ActualCollection);
        [NotNull] public static DoubleCollectionAssertChecker Collection(this CollectionAssert assert, ICollection<double> ActualCollection) => new DoubleCollectionAssertChecker(ActualCollection);

        [NotNull] public static DoubleDemensionArrayAssertChecker Collection(this CollectionAssert assert, double[,] array) => new DoubleDemensionArrayAssertChecker(array);

        [NotNull] public static CollectionAssertChecker<T> Collection<T>(this CollectionAssert assert, ICollection<T> ActualCollection) => new CollectionAssertChecker<T>(ActualCollection);
    }
}