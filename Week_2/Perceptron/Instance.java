public class Instance {
    public static DataReader reader;
    public int real_label = 0;
    public double[] feature_vals;

    public Instance(int real_label, double[] feature_vals) {
	this.real_label = real_label;
	this.feature_vals = feature_vals;
    }

    /*
      Given a set of weights and a possible label, return the score
      w * f(x,y), where x is this instance and y = label
     */
    public double getScore(double[] weights, int label) {
	double[] block_feature_vals = getFeatures(label); //[1.0,1.0,1.0,1.0...]

	double score = 0.0;
	for(int i = 0; i < weights.length; i++)
	    score += weights[i] * block_feature_vals[i]; //[1.0]*[1.0] = 1.0 .... ???

	return score;
    }

    /*
      Get a block feature representation f(x,y) = [1.0,3.0]  7:1.0
     */
    public double[] getFeatures(int label) {
	double[] block_feature_vals = new double[reader.num_features * reader.num_labels];
	for(int i = 0; i < feature_vals.length; i++) {
	    block_feature_vals[i + (label*reader.num_features)] = feature_vals[i]; // [0 + (0*4)] {0} = 1.0			[1 + (0*4)] {1} = 1.0		[2 + (0*4)] {2} = 1.0
	}																		   // [0 + (1*4)] {4} = 1.0			[1 + (1*4)] {5} = 1.0		[2 + (1*4)] {6} = 1.0					
																			   // [0 + (2*4)] {8} = 1.0			[1 + (2*4)] {9} = 1.0		[2 + (2*4)] {10} = 1.0
	return block_feature_vals;
    }

}
