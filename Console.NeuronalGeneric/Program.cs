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
// Konsolen Applikation mit Menü
// </summary>
//-----------------------------------------------------------------------

namespace Console.NeuronalGeneric
{
    /* Imports from NET Framework */
    using System;
    using System.Reflection.Emit;

    public class Program
    {
        public Program()
        {
            Console.OutputEncoding = System.Text.Encoding.UTF8;
            Console.CursorVisible = false;
        }
        private static void Main(string[] args)
        {
            CMenu mainMenu = new CMenu("Implementierung eines Neuronalen Netzes");
            mainMenu.AddItem("Spam Filter", MenuPoint1);
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

            NeuralNetwork net = new NeuralNetwork(6, new int[] { 6, 1 });

            var emails = new[]
            {
                ("Win money now!!! http://spam.com", 1),
                ("Free money waiting for you", 1),
                ("Congratulations you win", 1),
                ("Meeting tomorrow at 10", 0),
                ("Please review the document", 0),
                ("Lunch today?", 0),
                ("Deine Festplatte ist verschlüsselt. Diese geben für 1000 Bitcoins wieder frei.",1)
            };

            // Training
            for (int epoch = 0; epoch < 5000; epoch++)
            {
                foreach (var mail in emails)
                {
                    double[] input = SpamFeatureExtractor.Extract(mail.Item1);
                    double[] expected = { mail.Item2 };

                    net.Train(input, expected);
                }
            }

            Test(net);
            CMenu.Wait();
        }

        static void Test(NeuralNetwork net)
        {
            string[] tests =
            {
                "Free money now!!!",
                "Win big prizes",
                "Team meeting at 14",
                "Lunch tomorrow?",
                "Give me some Bitcoin",
            };

            foreach (var mail in tests)
            {
                var input = SpamFeatureExtractor.Extract(mail);
                var result = net.Predict(input);

                Console.WriteLine($"{mail} → Spam Wahrscheinlichkeit: {result[0]:F3}");
            }
        }
    }

    public static class SpamFeatureExtractor
    {
        public static double[] Extract(string text)
        {
            text = text.ToLower();

            double containsFree = text.Contains("free") ? 1 : 0;
            double containsMoney = text.Contains("money") ? 1 : 0;
            double containsWin = text.Contains("win") ? 1 : 0;
            double containsBitcoin = text.Contains("bitcoin") ? 1 : 0;

            double linkCount = text.Split("http").Length - 1;

            double upperRatio = text.Count(char.IsUpper) / (double)Math.Max(1, text.Length);

            return new double[]
            {
                containsFree,
                containsMoney,
                containsWin,
                containsBitcoin,
                linkCount,
                upperRatio
            };
        }
    }
    public class NeuralNetwork
    {
        private Layer[] layers;
        private double learningRate = 0.1;

        public NeuralNetwork(int inputSize, int[] layerSizes)
        {
            this.layers = new Layer[layerSizes.Length];

            int previousSize = inputSize;

            for (int i = 0; i < layerSizes.Length; i++)
            {
                this.layers[i] = new Layer(layerSizes[i], previousSize);
                previousSize = layerSizes[i];
            }
        }

        public double[] Predict(double[] input)
        {
            double[] output = input;

            foreach (var layer in this.layers)
            {
                output = layer.Forward(output);
            }

            return output;
        }

        public void Train(double[] input, double[] expected)
        {
            double[][] layerOutputs = new double[layers.Length + 1][];
            layerOutputs[0] = input;

            // Forward Pass
            for (int i = 0; i < layers.Length; i++)
            {
                layerOutputs[i + 1] = layers[i].Forward(layerOutputs[i]);
            }

            // Output Layer Delta
            Layer outputLayer = layers[layers.Length - 1];

            for (int i = 0; i < outputLayer.Neurons.Length; i++)
            {
                var neuron = outputLayer.Neurons[i];

                double error = expected[i] - neuron.Output;
                neuron.Delta = error * neuron.SigmoidDerivative();
            }

            // Hidden Layer Backpropagation
            for (int l = layers.Length - 2; l >= 0; l--)
            {
                Layer currentLayer = layers[l];
                Layer nextLayer = layers[l + 1];

                for (int i = 0; i < currentLayer.Neurons.Length; i++)
                {
                    double error = 0;

                    foreach (var nextNeuron in nextLayer.Neurons)
                    {
                        error += nextNeuron.Weights[i] * nextNeuron.Delta;
                    }

                    var neuron = currentLayer.Neurons[i];
                    neuron.Delta = error * neuron.SigmoidDerivative();
                }
            }

            // Gewichte aktualisieren
            for (int l = 0; l < layers.Length; l++)
            {
                var inputs = layerOutputs[l];

                foreach (var neuron in layers[l].Neurons)
                {
                    for (int w = 0; w < neuron.Weights.Length; w++)
                    {
                        neuron.Weights[w] += this.learningRate * neuron.Delta * inputs[w];
                    }

                    neuron.Bias += this.learningRate * neuron.Delta;
                }
            }
        }
    }

    public class Neuron
    {
        private static Random rnd = new Random();

        public Neuron(int inputCount)
        {
            this.Weights = new double[inputCount];

            for (int i = 0; i < inputCount; i++)
            {
                this.Weights[i] = rnd.NextDouble() * 2 - 1;
            }

            this.Bias = rnd.NextDouble() * 2 - 1;
        }

        public double[] Weights { get; set; }
        public double Bias { get; set; }
        public double Output { get; set; }
        public double Delta { get; set; }


        public double Activate(double[] inputs)
        {
            double sum = Bias;

            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i] * Weights[i];
            }

            this.Output = Sigmoid(sum);
            return this.Output;
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        public double SigmoidDerivative()
        {
            return Output * (1 - Output);
        }
    }

    public class Layer
    {
        public Layer(int neuronCount, int inputCount)
        {
            this.Neurons = new Neuron[neuronCount];

            for (int i = 0; i < neuronCount; i++)
            {
                this.Neurons[i] = new Neuron(inputCount);
            }
        }

        public Neuron[] Neurons { get; set; }

        public double[] Forward(double[] inputs)
        {
            double[] outputs = new double[Neurons.Length];

            for (int i = 0; i < this.Neurons.Length; i++)
            {
                outputs[i] = this.Neurons[i].Activate(inputs);
            }

            return outputs;
        }
    }
}
