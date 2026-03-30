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

namespace Console.NeuronalXOR
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
            mainMenu.AddItem("XOR Problem lösen", MenuPoint1);
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

            NeuralNetwork net = new NeuralNetwork();

            double[][] inputs =
            {
                new double[]{0,0},
                new double[]{0,1},
                new double[]{1,0},
                new double[]{1,1}
            };

            double[][] outputs =
            {
                new double[]{0},
                new double[]{1},
                new double[]{1},
                new double[]{0}
            };

            for (int epoch = 0; epoch < 10000; epoch++)
            {
                for (int i = 0; i < inputs.Length; i++)
                {
                    net.Train(inputs[i], outputs[i]);
                }
            }

            foreach (var input in inputs)
            {
                var result = net.Predict(input);

                Console.WriteLine($"{input[0]} XOR {input[1]} = {result[0]:F3}");
            }

            CMenu.Wait();
        }


        private static void UnterMenuPoint(string param)
        {
            Console.Clear();

            CMenu.Wait(param);
        }
    }

    #region Implementierung Neuron, XOR Lösung
    public class Neuron
    {
        public double[] Weights;
        public double Bias;
        public double Output;
        public double Delta;

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

        public double Activate(double[] inputs)
        {
            double sum = this.Bias;

            for (int i = 0; i < inputs.Length; i++)
            {
                sum += inputs[i] * this.Weights[i];
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
            return this.Output * (1 - this.Output);
        }
    }

    public class Layer
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

            for (int i = 0; i < Neurons.Length; i++)
            {
                outputs[i] = Neurons[i].Activate(inputs);
            }

            return outputs;
        }
    }

    public class NeuralNetwork
    {
        private Layer hiddenLayer;
        private Layer outputLayer;
        private double learningRate = 0.5;

        public NeuralNetwork()
        {
            hiddenLayer = new Layer(2, 2);
            outputLayer = new Layer(1, 2);
        }

        public double[] Predict(double[] input)
        {
            var hidden = hiddenLayer.Forward(input);
            var output = outputLayer.Forward(hidden);

            return output;
        }

        public void Train(double[] input, double[] expected)
        {
            // Forward Pass
            var hiddenOutputs = hiddenLayer.Forward(input);
            var finalOutputs = outputLayer.Forward(hiddenOutputs);

            // Output Layer Fehler
            for (int i = 0; i < outputLayer.Neurons.Length; i++)
            {
                var neuron = outputLayer.Neurons[i];

                double error = expected[i] - neuron.Output;

                neuron.Delta = error * neuron.SigmoidDerivative();
            }

            // Hidden Layer Fehler
            for (int i = 0; i < hiddenLayer.Neurons.Length; i++)
            {
                var neuron = hiddenLayer.Neurons[i];

                double error = 0;

                foreach (var outputNeuron in outputLayer.Neurons)
                {
                    error += outputNeuron.Weights[i] * outputNeuron.Delta;
                }

                neuron.Delta = error * neuron.SigmoidDerivative();
            }

            // Gewichte Output Layer aktualisieren
            foreach (var neuron in outputLayer.Neurons)
            {
                for (int i = 0; i < neuron.Weights.Length; i++)
                {
                    neuron.Weights[i] += learningRate *
                                         neuron.Delta *
                                         hiddenOutputs[i];
                }

                neuron.Bias += learningRate * neuron.Delta;
            }

            // Gewichte Hidden Layer aktualisieren
            foreach (var neuron in hiddenLayer.Neurons)
            {
                for (int i = 0; i < neuron.Weights.Length; i++)
                {
                    neuron.Weights[i] += learningRate *
                                         neuron.Delta *
                                         input[i];
                }

                neuron.Bias += learningRate * neuron.Delta;
            }
        }
    }
    #endregion Implementierung Neuron, XOR Lösung

}
