using System.Collections;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MathCore.AI.Tests.Service
{
    internal class CollectionAssertChecker
    {
        private readonly ICollection _ActualCollection;

        public CollectionAssertChecker(ICollection ActualCollection) => _ActualCollection = ActualCollection;

        public void AreEquals(ICollection ExpectedCollection) => CollectionAssert.AreEqual(ExpectedCollection, _ActualCollection);
    }
}