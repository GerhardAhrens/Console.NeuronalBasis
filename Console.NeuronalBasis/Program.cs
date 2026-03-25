//-----------------------------------------------------------------------
// <copyright file="Program.cs" company="Lifeprojects.de">
//     Class: Program
//     Copyright © Lifeprojects.de 2026
// </copyright>
// <Template>
// 	Version 3.0.2026.1, 08.1.2026
// </Template>
//
// <author>Gerhard Ahrens - Lifeprojects.de</author>
// <email>developer@lifeprojects.de</email>
// <date>03.03.2026 14:26:39</date>
//
// <summary>
// Basis Implementierung zu einem Neuronalen Netz.
// Erzeugen eines Neurons
// </summary>
//-----------------------------------------------------------------------

namespace Console.NeuronalBasis
{
    /* Imports from NET Framework */
    using System;

    public class Program
    {
        public Program()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.CursorVisible = false;
        }
        private static void Main(string[] args)
        {
            CMenu unterMenu = new CMenu("Untermenü");
            unterMenu.AddItem("Untermenüpunkt 1", () => UnterMenuPoint("A"), "🖥");
            unterMenu.AddItem("Untermenüpunkt 2", () => UnterMenuPoint("B"), "🔊");

            CMenu mainMenu = new CMenu("Implementierung eines Neuronalen Netzes");
            mainMenu.AddItem("Ein Neuron erzeugen", MenuPoint1);
            //mainMenu.AddSubMenu("Einstellungen", unterMenu, "⚙");
            mainMenu.AddItem("Beenden", () => ApplicationExit());
            mainMenu.Show();
        }

        private static void ApplicationExit()
        {
            Environment.Exit(0);
        }

        private static void MenuPoint1()
        {
            Console.Clear();

            NeuralNetwork net = new NeuralNetwork(inputSize: 2, layerSizes: new int[] { 3, 1 } );

            double[] input = { 0.5, 0.8 };

            double[] result = net.Predict(input);

            Console.WriteLine(result[0]);

            CMenu.Wait();
        }

        private static void UnterMenuPoint(string param)
        {
            Console.Clear();

            CMenu.Wait(param);
        }
    }

    #region Implementierung Neuron
    internal class Neuron
    {
        public double[] Weights;
        public double Bias;

        public Neuron(int inputCount)
        {
            Random rand = new Random();

            this.Weights = new double[inputCount];

            for (int i = 0; i < inputCount; i++)
            {
                this.Weights[i] = rand.NextDouble() * 2 - 1;
            }

            this.Bias = rand.NextDouble() * 2 - 1;
        }

        public double Activate(double[] inputs)
        {
            double sum = this.Bias;

            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i] * this.Weights[i];
            }

            return Sigmoid(sum);
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }

    internal class Layer
    {
        public Neuron[] Neurons;

        public Layer(int neuronCount, int inputCount)
        {
            this.Neurons = new Neuron[neuronCount];

            for (int i = 0; i < neuronCount; i++)
            {
                this.Neurons[i] = new Neuron(inputCount);
            }
        }

        public double[] Forward(double[] inputs)
        {
            double[] outputs = new double[this.Neurons.Length];

            for (int i = 0; i < this.Neurons.Length; i++)
            {
                outputs[i] = this.Neurons[i].Activate(inputs);
            }

            return outputs;
        }
    }

    internal class NeuralNetwork
    {
        private Layer[] Layers;

        public NeuralNetwork(int inputSize, int[] layerSizes)
        {
            this.Layers = new Layer[layerSizes.Length];

            int previousSize = inputSize;

            for (int i = 0; i < layerSizes.Length; i++)
            {
                this.Layers[i] = new Layer(layerSizes[i], previousSize);
                previousSize = layerSizes[i];
            }
        }

        public double[] Predict(double[] inputs)
        {
            double[] output = inputs;

            foreach (var layer in this.Layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }
    }
    #endregion Implementierung Neuron

    #region Spam Feature Extractor Klasse
    public static class SpamFeatureExtractor
    {
        public static double[] Extract(string text)
        {
            text = text.ToLower();

            double containsFree = text.Contains("free") ? 1 : 0;
            double containsMoney = text.Contains("money") ? 1 : 0;
            double containsWin = text.Contains("win") ? 1 : 0;

            double linkCount = text.Split("http").Length - 1;

            double upperRatio =
                text.Count(char.IsUpper) /
                (double)Math.Max(1, text.Length);

            return new double[]
            {
            containsFree,
            containsMoney,
            containsWin,
            linkCount,
            upperRatio
            };
        }
    }
    #endregion Spam Feature Extractor Klasse
}
