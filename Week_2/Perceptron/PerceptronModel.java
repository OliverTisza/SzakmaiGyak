public class PerceptronModel {
    public static DataReader reader;
    public double[] weights;
    public double[] avg_weights;
    public int num_updates = 0;

    public PerceptronModel() {
	this.weights = new double[reader.num_features * reader.num_labels];
	this.avg_weights = new double[reader.num_features * reader.num_labels];
    }

    // Given an Instance, return the class label
    public int predict(Instance inst) { //Tippeles
	/*
	  Predict a label given an instance inst
	*/
    	double max = 0;
    	int maxI = 0;
    	
    	for(int i = 0; i <3 ; i++) {
    		double currMax = inst.getScore(weights, i);
    		if(currMax > max) {
    			max = currMax;
    			maxI = i;
    		}
    		
    		//return maxI;
    	}
    	
    	
	return maxI; // dummy to make it compile, returns the label 0
    }

    // Given a set of instances, calculate labeling accuracy
    public double accuracy(Instance[] instances) {
	int correct = 0;
	int total = 0;
	for(int i = 0; i < instances.length; i++) {
	    int predicted_label = predict(instances[i]);
	    if(predicted_label == instances[i].real_label)
		correct++;
	    total++;
	}
	return (double)correct/total;
    }

    // Method to be called at the end of training to set weights to avg_weights
    public void setToAvgWeights() {
	/*
	  Needed for second part of the assignment
	*/
    	
    	weights = avg_weights;
    	
    	
    }

    // Update parameters by adding the update vector
    public void update(double[] updates) {
	/*
	  Need to fill in for both parts
	*/
    	for (int i=0; i< updates.length;i++) {
    		
    		weights[i] = weights[i] + updates[i];
    		
    	}
    	
    	
    }

}
