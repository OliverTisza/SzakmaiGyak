import java.io.*;
import java.util.Arrays;

public class Perceptron {
	public static void main(String[] args) throws IOException {

		DataReader reader = new DataReader();
		reader.init(args);

		Instance[] train_data = reader.getInstances(args[0]);
		Instance[] test_data = reader.getInstances(args[1]);

		PerceptronModel model = learn(train_data, 5);

		System.out.println("\n\nRegular Perceptron\n======================================================");
		System.out.println("Final Training Accuracy: " + model.accuracy(train_data));
		System.out.println("Final Testing Accuracy: " + model.accuracy(test_data));
		System.out.println("======================================================");

		model.setToAvgWeights();
		System.out.println("\n\nAveraged Perceptron\n======================================================");
		System.out.println("Final Training Accuracy: " + model.accuracy(train_data));
		System.out.println("Final Testing Accuracy: " + model.accuracy(test_data));
		System.out.println("======================================================");
	}

	public static PerceptronModel learn(Instance[] train_data, int num_iters) {
		PerceptronModel model = new PerceptronModel();

		/*
		 * 
		 * Fill in the main body of the perceptron algorithm
		 * 
		 */

		for (int k = 0; k < num_iters; k++) {

			for (int i = 0; i < train_data.length; i++) {
				

				int guess = model.predict(train_data[i]);

				if (guess != train_data[i].real_label) {

					double[] helyes = train_data[i].getFeatures(train_data[i].real_label);
					double[] tipp = train_data[i].getFeatures(guess);
					double update[] = tipp;
					Arrays.fill(update, 0);

					for (int j = 0; j < tipp.length; j++) {

						update[j] = helyes[j] - tipp[j];

					}

					model.update(update);
				}

				for (int j = 0; j < model.weights.length; j++) {

					model.avg_weights[j] += model.weights[j];

				}

			}

		}

		for (int j = 0; j < model.weights.length; j++) {

			model.avg_weights[j] = model.avg_weights[j] / (num_iters * train_data.length);

		}

		return model; // Returns a model with all weights = 0;
	}

}
