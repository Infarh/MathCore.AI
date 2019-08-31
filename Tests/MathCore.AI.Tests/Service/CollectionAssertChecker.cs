using System.Collections;
using System.Collections.Generic;

// ReSharper disable once CheckNamespace
namespace Microsoft.VisualStudio.TestTools.UnitTesting
{
    public class CollectionAssertChecker<T>
    {
        private readonly ICollection<T> _ActualCollection;
        public CollectionAssertChecker(ICollection<T> ActualCollection) => _ActualCollection = ActualCollection;

        public void AreEquals(ICollection<T> ExpectedCollection) => CollectionAssert.AreEqual((ICollection)ExpectedCollection, (ICollection)_ActualCollection);
    }

    internal class CollectionAssertChecker
    {
        private readonly ICollection _ActualCollection;

        public CollectionAssertChecker(ICollection ActualCollection) => _ActualCollection = ActualCollection;

        public void AreEquals(ICollection ExpectedCollection) =>
            CollectionAssert.AreEqual(ExpectedCollection, _ActualCollection);
    }
}