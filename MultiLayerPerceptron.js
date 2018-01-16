var ActivationFunction = {
	None: { value: 0 },
	Sigmoid: { value: 1},
	Linear: { value: 2},
	Gaussian: { value: 3},
	RationalSigmoid: { value: 4}
};

this.ActivationFunctions = function (aFunc, input){
	var activation = this;
	this.Evaluate = function (aFunc, input){
		switch (aFunc.value) {
			case 1:
				return activation.sigmoid(input);
			case 2:
				return activation.linear(input);
			case 3:
				return activation.gaussian(input);
			case 4:
				return activation.rationalsigmoid(input);
			case 0:
			default:
				return 0.0;
		}
	}

	this.EvaluateDerivative = function (aFunc, input) {
		switch (aFunc.value) {
			case 1:
				return activation.sigmoid_derivative(input);
			case 2:
				return activation.linear_derivative(input);
			case 3:
				return activation.gaussian_derivative(input);
			case 4:
				return activation.rationalsigmoid_derivative(input);
			case 0:
			default:
				return 0.0;
		}
	}

	this.sigmoid = function (x){
		return 1.0 / (1.0 + Math.exp(-x));
	}
	
	this.sigmoid_derivative = function (x) {
		return activation.sigmoid(x) * (1 - activation.sigmoid(x));
	}

	this.linear = function (x) {
		return x;
	}

	this.linear_derivative = function (x) {
		return 1;
	}

	this.gaussian = function (x) {
		return Math.exp(-Math.pow(x,2));
	}
	this.gaussian_derivative = function (x) {
		return -2.0 * x * activation.gaussian(x);
	}

	this.rationalsigmoid = function (x) {
		return x / (1.0 + Math.sqrt(1.0 + x * x));
	}

	this.rationalsigmoid_derivative = function (x) {
		var val = Math.sqrt(1.0 + x * x);
		return 1.0 / (val * ( 1+ val));
	}

}

this.Gaussian = function () {
	this.GetRandomGaussian = function () {
		var gauss = this;

		var u, v, s, t, val1, val2;

		do {
						u = 2 * Math.random() - 1;
						v = 2 * Math.random() - 1;
		} while ( u* u + v * v > 1 || ( u == 0 && v == 0));

		s = u * u + v * v;
		t = Math.sqrt((-2.0 * Math.log(s)) / s);

		val1 = u * t;
		val2 = v * t;

		return val1;
	}
}

this.MultiLayerPerceptron = function () {
	var Network = this;
	var layerCount;
	var inputSize;
	var layerSize;
	var activationFunction;

	var layerOutput;
	var layerInput;
	var bias;
	var delta;
	var previousBiasDelta;

	var weight;
	var previousWeightDelta;

		this.init = function (layerSizes, activationFunctions) {
			if ( activationFunctions.length != layerSizes.length || activationFunctions[0] != ActivationFunction.None) {
					console.log("Não é possivel construir a rede com estes parametros");
					return;
			}

			layerCount = layerSizes.length - 1;
			inputSize = layerSizes[0];
			layerSize = [];

			for (var i = 0; i < layerCount; i++){
				layerSize[i] = layerSizes[i + 1];
			}

			activationFunction = [];
			for (var i=0; i < layerCount; i++){
				activationFunction[i] = activationFunctions[i + 1]; 
			}

			//bidimensinal
			bias = [];
			previousBiasDelta = [];
			delta = [];
			layerOutput = [];
			layerInput = [];

			//tridimensinal
			weight = [];
			previousWeightDelta = [];

			for ( var l = 0; l < layerCount; l++) {
				bias[l] = [];
				previousBiasDelta[l] = [];
				delta[l] = [];

				layerOutput[l] = [];
				layerInput[l] = [];

				weight[l] = [];
				previousWeightDelta[l] = [];

				 for( var i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]) ; i++) {
				 	weight[l][i] = [];
				 	previousWeightDelta[l][i] = [];
				 }
			}

				//iniciar os pesos

				for ( var l = 0; l < layerCount; l++) {
					for ( var j = 0; j < layerSize[l]; j++) {
						bias[l][j] = new Gaussian().GetRandomGaussian();
						previousBiasDelta[l][j] = 0.0;
						layerOutput[l][j] = 0.0;
						layerInput[l][j] = 0.0;
						delta[l][j] = 0.0;
					}

					for (var i = 0; i < ( l == 0 ? inputSize : layerSize[l - 1]) ; i++ ){
						for (var j = 0; j < layerSize[l]; j++){
							weight[l][i][j] = new Gaussian().GetRandomGaussian();
							previousWeightDelta[l][i][j] = 0.0;
						}
					}
				}

		}

		this.Run = function (input) {
			if ( input.length != inputSize ){
				console.log("Dados inconsistentes");
			}

			var output = [];

			for ( var l = 0; l < layerCount; l++){
				for ( var j = 0; j < layerSize[l]; j++){
					var sum = 0;
					for (var i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]); i++){
						sum += weight[l][i][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);
					}

					sum += bias[l][j];
					layerInput[l][j] = sum;
					layerOutput[l][j] = new ActivationFunctions().Evaluate(activationFunction[l], sum);
				}
			}

			for ( var i = 0; i < layerSize[layerCount - 1]; i++){
				output[i] = layerOutput[layerCount - 1][i];
			}

			return output;
		}

		this.TrainCase = function (input, desired, TrainingRate, Momentum){
			if (input.length != inputSize){
				console.log("Entrada invalida");
				return;
			}

			if (desired.length != layerSize[layerCount - 1]){
				console.log("Entrada inconsistente");
			}
			var error = 0.0, sum = 0.0, weightDelta = 0.0, biasDelta = 0.0;

			var output = [];
			output = Network.Run(input);

			for ( var l = layerCount - 1; l >= 0; l--)
			{
				if ( l == layerCount - 1){
					for ( var k = 0; k < layerSize[l]; k++){
						delta[l][k] = output[k] - desired[k];
						error += Math.pow(delta[l][k], 2);
						delta[l][k] *= new ActivationFunctions().EvaluateDerivative(activationFunction[l], layerInput[l][k]);
					}
				}
					else // Camadas ocultas
					{
						for ( var i = 0; i < layerSize[l]; i++){
							sum = 0.0;
							for ( var j = 0; j < layerSize[l + 1]; j++){
								sum += weight[l + 1][i][j] * delta[l][j];
							}
							sum *= new ActivationFunctions().EvaluateDerivative(activationFunction[l], layerInput[l][i]);
							delta[l][i] = sum;
						}
					}
			}

			for ( var l = 0; l < layerCount; l++){
				for( var i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]) ; i++){
					for(var j = 0; j < layerSize[l]; j++){
						weightDelta = TrainingRate * delta[l][j] * (l == 0 ? input[i] : layerOutput[l - 1][i]);
						weight[l][i][j] -= weightDelta + Momentum * previousWeightDelta[l][i][j];
						previousWeightDelta[l][i][j] = weightDelta;
					}
				}
			}

			for ( var l = 0;l < layerCount; l++ ){
				for ( var i = 0; i < layerSize[l]; i++){
					biasDelta = TrainingRate * delta[l][i];
					bias[l][i] -= biasDelta + Momentum * previousBiasDelta[l][i];

					previousBiasDelta[l][i] = biasDelta;
				}
			}

			return error;
		}

		this.Train = function(training_sets, max_count) {
			var input, output;

			input = [];
			output = [];
			for (var i = 0; i < training_sets.length; i++){
				input[i] = [];

				for(var j = 0; j < training_sets[i].input.length; j++){
					input[i][j] = training_sets[i].input[j];
				}
			}

			for (var i = 0; i < training_sets.length; i++){
				output[i] = [];

				for ( var j = 0; j < training_sets[i].output.length; j++){
					output[i][j] = training_sets[i].output[j];
				}
			}
			var error = 0.0;
			var count = 0;

			do{
				count++;
				error = 0.0;

				for(var i = 0; i < training_sets.length; i++){
					error += Network.TrainCase(input[i], output[i], 0.15, 0.10);
				}
				if (count % 100 == 0) console.log("Época " + count + " completada com erro " + error);
			
			} while (error > 0.0001 && count <= max_count);
		}

		this.ToJson = function (o) {
			var writer = {
				NeuralNetwork: { Type: "MultiLayerPerceptron" },
				Parameters: { inputSize: inputSize, layerCount: layerCount, Layers: [] },
				Weights: [],
			}

			for( var l = 0; l< layerCount; l++){
				writer.Parameters.Layers[l] = { Index: l, Size: layerSize[l], Type: activationFunction[l]};
			}

			for( var l = 0; l< layerCount; l++){
				writer.Weights[l] = {Layer: { Index: l, Node: [] } }

				for ( var j = 0; j< layerCount[l]; j++){
					writer.Weights[l].Layer.Node[j] = { Index: j, Bias: bias[l][j], Axon: [] };

					for ( var i = 0; i < (l == 0 ? inputSize : layerSize[l - 1]) ; i++){
						writer.Weights[l].Layer.Node[j].Axon[i] = { Index: i, Value: weight[l][i][j] }
					}
				}
			}
			document.getElementById(o).value = JSON.stringify(writer);
		}

		this.Load = function ( file, callback) {
				$.getJSON(file, function(o){
					layerCount = o.Parameters.layerCount;
					inputSize =  o.Parameters.inputSize;
					layerSize = [];

					for ( var i = 0; i < layerCount; i++){
						layerSize[i] = o.Parameters.Layers[i].Size;
					}

					activationFunction = [];
					for ( var i = 0; i < layerCount; i++){
						activationFunction[i] = o.Parameters.Layers[i].Type;
					}

					//bidimensinal
					bias = [];
					previousBiasDelta = [];
					delta = [];
					layerOutput = [];
					layerInput = [];

					//tridimensinal
					weight = [];
					preveiousWeightDelta = [];

					for ( var l = 0; l < layerCount; l++){
						bias[l] = [];
						previousBiasDelta[l] = [];
						delta[l] = [];

						layerOutput[l] = [];
						layerInput[l] = [];

						weight[l] = [];
						previousWeightDelta[l] = [];

						for ( var i = 0 ; i < (l == 0 ? inputSize : layerSize[l - 1]); i++){
							weight[l][i] = [];
							previousWeightDelta[l][i] = [];
						}
					}

					//carregar pesos
					for ( var l = 0; l < layerCount; l++){
						for ( var j = 0; j < layerSize[l]; j++){
							bias[l][j] = o.Weights[l].Layer.Node[j].Bias;
						}

						for ( var i = 0; i < ( l == 0 ? inputSize : layerSize[l - 1]) ; i++){
							for( var j = 0; j < layerSize[l]; j++){
								weight[l][i][j] = o.Weights[l].Layer.Node[j].Axon[i].Value;
							}
						}
					}
					if (callback) callback();
				});
		}
}