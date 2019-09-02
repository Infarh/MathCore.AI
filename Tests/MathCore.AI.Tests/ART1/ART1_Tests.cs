using System;
using System.Linq;
using System.Text;
using MathCore.AI.ART1;
using MathCore.Annotations;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MathCore.AI.Tests.ART1
{
    [TestClass]
    public class ART1_Tests
    {
        /// <summary>Сущность, подлежащая классификации</summary>
        private class Item
        {
            /// <summary>Молоток</summary>
            public bool Humer { get; set; }
            /// <summary>Бумага</summary>
            public bool Paper { get; set; }
            /// <summary>Шоколадка</summary>
            public bool Snickers { get; set; }
            /// <summary>Отвёртка</summary>
            public bool Screwdriver { get; set; }
            /// <summary>Ручка</summary>
            public bool Pen { get; set; }
            /// <summary>Шоколадка</summary>
            public bool KitKat { get; set; }
            /// <summary>Гаечный ключ</summary>
            public bool Wrench { get; set; }
            /// <summary>Карандаш</summary>
            public bool Pencil { get; set; }
            /// <summary>Шоколадка</summary>
            public bool HeanthBar { get; set; }
            /// <summary>Счётчик ленты</summary>
            public bool TapeCounter { get; set; }
            /// <summary>Переплётная машина</summary>
            public bool BindingMachine { get; set; }

            public Item() { }

            public Item([NotNull] params int[] Values)
            {
                if (Values is null) throw new ArgumentNullException(nameof(Values));
                for (var i = 0; i < Values.Length; i++)
                    switch (i)
                    {
                        case 0: Humer = Values[i] > 0; break;
                        case 1: Paper = Values[i] > 0; break;
                        case 2: Snickers = Values[i] > 0; break;
                        case 3: Screwdriver = Values[i] > 0; break;
                        case 4: Pen = Values[i] > 0; break;
                        case 5: KitKat = Values[i] > 0; break;
                        case 6: Wrench = Values[i] > 0; break;
                        case 7: Pencil = Values[i] > 0; break;
                        case 8: HeanthBar = Values[i] > 0; break;
                        case 9: TapeCounter = Values[i] > 0; break;
                        case 10: BindingMachine = Values[i] > 0; break;
                    }
            }

            #region Overrides of Object

            public override string ToString()
            {
                var result = new StringBuilder();

                if (Humer) result.Append($"{nameof(Humer)}, ");
                if (Paper) result.Append($"{nameof(Paper)}, ");
                if (Snickers) result.Append($"{nameof(Snickers)}, ");
                if (Screwdriver) result.Append($"{nameof(Screwdriver)}, ");
                if (Pen) result.Append($"{nameof(Pen)}, ");
                if (KitKat) result.Append($"{nameof(KitKat)}, ");
                if (Wrench) result.Append($"{nameof(Wrench)}, ");
                if (Pencil) result.Append($"{nameof(Pencil)}, ");
                if (HeanthBar) result.Append($"{nameof(HeanthBar)}, ");
                if (TapeCounter) result.Append($"{nameof(TapeCounter)}, ");
                if (BindingMachine) result.Append($"{nameof(BindingMachine)}, ");

                if (result.Length > 0) result.Length -= 2;

                return $"[ {result} ]";
            }

            #endregion
        }

        [TestMethod]
        public void Algotithm_Base_Logic()
        {
            int[] p0 = { 1, 1, 1, 0, 0, 1, 0 };
            int[] p1 = { 1, 0, 0, 1, 1, 0, 1 };
            int[] p2 = { 1, 1, 0, 0, 0, 1, 0 };

            var classificator = new Classificator<int[]>
            {
                Vigilance = 0.9,
                Beta = 1,
            };
            for (var i = 0; i < p0.Length; i++)
            {
                var index = i;
                classificator.Criterias.Add(p => p[index]);
            }

            var claster0 = classificator.Add(p0);
            var claster1 = classificator.Add(p1);
            var claster2 = classificator.Add(p2);

            Assert.That.Value(claster0).AreReferenceEquals(claster2);
            Assert.That.Value(claster0).AreNotReferenceEquals(claster1);

            Assert.That.Value(claster0.ItemsCount).AreEqual(2);
            Assert.That.Value(claster1.ItemsCount).AreEqual(1);

            Assert.IsTrue(claster0.Contains(p0));
            Assert.IsTrue(claster0.Contains(p2));

            Assert.IsTrue(claster1.Contains(p1));
        }

        [TestMethod]
        public void Algorithm_Logic()
        {
            var classificator = new Classificator<Item>
            {
                Vigilance = 0.9,
                Beta = 1,
                Criterias =
                {
                    { "Молоток", Item => Item.Humer ? 1 : 0 },
                    { "Бумага", Item => Item.Paper ? 1 : 0 },
                    { "Snickers", Item => Item.Snickers ? 1 : 0 },
                    { "Отвёртка", Item => Item.Screwdriver ? 1 : 0 },
                    { "Ручка", Item => Item.Pen ? 1 : 0 },
                    { "Kit-Kat", Item => Item.KitKat ? 1 : 0 },
                    { "Гаечный ключ", Item => Item.Wrench ? 1 : 0 },
                    { "Карандаш", Item => Item.Pencil ? 1 : 0 },
                    { "Heanth Bar", Item => Item.HeanthBar ? 1 : 0 },
                    { "Счётчик ленты", Item => Item.TapeCounter ? 1 : 0 },
                    { "Переплётная машина", Item => Item.BindingMachine ? 1 : 0 },
                }
            };

            Item[] items =
            {        //  Hm Pp Sn Sc Pn KK Wr Pc HB TC BM
                new Item(0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0), //  0
                new Item(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1), //  1
                new Item(0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0), //  2
                new Item(0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1), //  3
                new Item(1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0), //  4
                new Item(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1), //  5
                new Item(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0), //  6
                new Item(0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0), //  7
                new Item(0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0), //  8
                new Item(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0), //  9
            };

            Assert.That.Value(classificator.Clusters.Count).AreEqual(0);

            var classification = classificator.Add(items);
            
            Assert.That.Value(classificator.Clusters.Count).GreaterThen(0);

            foreach (var current_class in classificator)
                foreach (var another_class in classificator.Except(current_class))
                    foreach (var item in current_class)
                        Assert.IsFalse(another_class.Contains(item));
        }
    }
}
