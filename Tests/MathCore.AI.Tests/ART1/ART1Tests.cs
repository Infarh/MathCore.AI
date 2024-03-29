﻿using System.Text;
using MathCore.AI.ART1;

// ReSharper disable MemberCanBePrivate.Local
// ReSharper disable AutoPropertyCanBeMadeGetOnly.Local
namespace MathCore.AI.Tests.ART1;

[TestClass]
public class ART1Tests
{
    /// <summary>Сущность, подлежащая классификации</summary>
    private record Item
    {
        /// <summary>Молоток</summary>
        public bool Hummer { get; init; }
        
        /// <summary>Бумага</summary>
        public bool Paper { get; init; }
        
        /// <summary>Шоколадка</summary>
        public bool Snickers { get; init; }
        
        /// <summary>Отвёртка</summary>
        public bool Screwdriver { get; init; }
        
        /// <summary>Ручка</summary>
        public bool Pen { get; init; }
        
        /// <summary>Шоколадка</summary>
        public bool KitKat { get; init; }
        
        /// <summary>Гаечный ключ</summary>
        public bool Wrench { get; init; }
        
        /// <summary>Карандаш</summary>
        public bool Pencil { get; init; }
        
        /// <summary>Шоколадка</summary>
        public bool ChocolateBar { get; init; }
        
        /// <summary>Счётчик ленты</summary>
        public bool TapeCounter { get; init; }
        
        /// <summary>Переплётная машина</summary>
        public bool BindingMachine { get; init; }

        public Item(params int[] Values)
        {
            if (Values is null) throw new ArgumentNullException(nameof(Values));
            for (var i = 0; i < Values.Length; i++)
                switch (i)
                {
                    case 0:  Hummer         = Values[i] > 0; break;
                    case 1:  Paper          = Values[i] > 0; break;
                    case 2:  Snickers       = Values[i] > 0; break;
                    case 3:  Screwdriver    = Values[i] > 0; break;
                    case 4:  Pen            = Values[i] > 0; break;
                    case 5:  KitKat         = Values[i] > 0; break;
                    case 6:  Wrench         = Values[i] > 0; break;
                    case 7:  Pencil         = Values[i] > 0; break;
                    case 8:  ChocolateBar   = Values[i] > 0; break;
                    case 9:  TapeCounter    = Values[i] > 0; break;
                    case 10: BindingMachine = Values[i] > 0; break;
                }
        }

        #region Overrides of Object

        public override string ToString()
        {
            var result = new StringBuilder();

            if (Hummer) result.Append($"{nameof(Hummer)}, ");
            if (Paper) result.Append($"{nameof(Paper)}, ");
            if (Snickers) result.Append($"{nameof(Snickers)}, ");
            if (Screwdriver) result.Append($"{nameof(Screwdriver)}, ");
            if (Pen) result.Append($"{nameof(Pen)}, ");
            if (KitKat) result.Append($"{nameof(KitKat)}, ");
            if (Wrench) result.Append($"{nameof(Wrench)}, ");
            if (Pencil) result.Append($"{nameof(Pencil)}, ");
            if (ChocolateBar) result.Append($"{nameof(ChocolateBar)}, ");
            if (TapeCounter) result.Append($"{nameof(TapeCounter)}, ");
            if (BindingMachine) result.Append($"{nameof(BindingMachine)}, ");

            if (result.Length > 0) result.Length -= 2;

            return $"[ {result} ]";
        }

        #endregion
    }

    [TestMethod]
    public void Algorithm_Base_Logic()
    {           
        int[] p0 = [1, 1, 1, 0, 0, 1, 0];
        int[] p1 = [1, 0, 0, 1, 1, 0, 1];
        int[] p2 = [1, 1, 0, 0, 0, 1, 0];

        var classificator = new Classificator<int[]>
        {
            Vigilance = 0.9,
            Beta      = 1,
        };
        for (var i = 0; i < p0.Length; i++)
        {
            var index = i;
            classificator.Criterias.Add(p => p[index]);
        }

        var cluster0 = classificator.Add(p0);
        var cluster1 = classificator.Add(p1);
        var cluster2 = classificator.Add(p2);

        Assert.That.Value(cluster0).IsReferenceEquals(cluster2);
        Assert.That.Value(cluster0).IsNotReferenceEquals(cluster1);

        Assert.That.Value(cluster0.ItemsCount).IsEqual(2);
        Assert.That.Value(cluster1.ItemsCount).IsEqual(1);

        Assert.IsTrue(cluster0.Contains(p0));
        Assert.IsTrue(cluster0.Contains(p2));

        Assert.IsTrue(cluster1.Contains(p1));
    }

    [TestMethod]
    public void Algorithm_Logic()
    {
        var classificator = new Classificator<Item>
        {
            Vigilance = 0.9,
            Beta      = 1,
            Criterias =
            {
                { "Молоток", Item => Item.Hummer ? 1 : 0 },
                { "Бумага", Item => Item.Paper ? 1 : 0 },
                { "Snickers", Item => Item.Snickers ? 1 : 0 },
                { "Отвёртка", Item => Item.Screwdriver ? 1 : 0 },
                { "Ручка", Item => Item.Pen ? 1 : 0 },
                { "Kit-Kat", Item => Item.KitKat ? 1 : 0 },
                { "Гаечный ключ", Item => Item.Wrench ? 1 : 0 },
                { "Карандаш", Item => Item.Pencil ? 1 : 0 },
                { "Heanth Bar", Item => Item.ChocolateBar ? 1 : 0 },
                { "Счётчик ленты", Item => Item.TapeCounter ? 1 : 0 },
                { "Переплётная машина", Item => Item.BindingMachine ? 1 : 0 },
            }
        };

        Item[] items =
        [                                         //  Hm Pp Sn Sc Pn KK Wr Pc HB TC BM
            new(0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0), //  0
            new(0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1), //  1
            new(0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0), //  2
            new(0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1), //  3
            new(1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0), //  4
            new(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1), //  5
            new(1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0), //  6
            new(0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0), //  7
            new(0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0), //  8
            new(0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0), //  9
        ];

        Assert.That.Value(classificator.Clusters.Count).IsEqual(0);

        // ReSharper disable once UnusedVariable
        var classification = classificator.Add(items);
            
        Assert.That.Value(classificator.Clusters.Count).GreaterThan(0);

        foreach (var current_class in classificator)
            foreach (var another_class in classificator.Except(current_class))
                foreach (var item in current_class)
                    Assert.IsFalse(another_class.Contains(item));
    }
}