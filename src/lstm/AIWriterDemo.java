package lstm;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AIWriterDemo {

	
	
	public static void main(String[] args) throws ClassNotFoundException, IOException {

		int miniBatchSize =12;// 32;	//Size of mini batch to use when  training
		int examplesPerEpoch = 15*miniBatchSize;//50 * miniBatchSize;	//i.e., how many examples to learn on between generating samples
		int exampleLength = 40;//  100;					//Length of each training example
        
		//Get Network
		String modelfile = "data/model.txt";
		//modelfile = "C:\\data\\model.txt";
		//File mf = new File(modelfile);
		LSTMWordSeqModel m = new LSTMWordSeqModel();
		MultiLayerNetwork net = m.ReadModelFromFile(modelfile);
		
		
		//Get CharacterIterator
		
    	String fileLocation ="data/jaylyrics_all.txt";
		char[] validCharacters = CharacterIterator.getChineseCharSet(fileLocation);
		CharacterIterator iter = new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
				miniBatchSize, exampleLength, examplesPerEpoch, validCharacters, new Random(12345),true);
		
		//Prepare writing 
		int numSamples = 1;
		int charToSample = 300;
		String initStr ="美的空气";
		long rnd = System.currentTimeMillis();
		
		String[] samples = sampleCharactersFromNetwork(initStr , net,
				iter, new Random(rnd), charToSample, numSamples );
		//writing
		
		for(String s : samples){
			s = s.replace('|', '\n');
			System.out.println(s);
		}
		
	}
	/** Generate a sample from the network, given an (optional, possibly null) initialization. Initialization
	 * can be used to 'prime' the RNN with a sequence you want to extend/continue.<br>
	 * Note that the initalization is used for all samples
	 * @param initialization String, may be null. If null, select a random character as initialization for all samples
	 * @param charactersToSample Number of characters to sample from network (excluding initialization)
	 * @param net MultiLayerNetwork with one or more GravesLSTM/RNN layers and a softmax output layer
	 * @param iter CharacterIterator. Used for going from indexes back to characters
	 */
	private static String[] sampleCharactersFromNetwork( String initialization, MultiLayerNetwork net,
			CharacterIterator iter, Random rng, int charactersToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		if( initialization == null ){
			initialization = String.valueOf(iter.getRandomCharacter());
		}
		
		//Create input for initialization
		//numSamples, numCharacters, inputLength
		//shape -> dimensions
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
		//initializationInput = Nd4j.zeros(numSamples, iter.inputColumns());
		//INDArray i2 = Nd4j.zeros(numSamples, iter.inputColumns());
	
		
		//INDArray ii = Nd4j.zeros(2, 3, 4);
		char[] init = initialization.toCharArray();
		for( int len=0; len<init.length; len++ ){
			
			int cidx = iter.convertCharacterToIndex(init[len]);
			for( int s=0; s<numSamples; s++ ){
				
				//insert 1.0 at [numSample, charIdx, all_initStr]
				//multi word input --> single word output
				initializationInput.putScalar(new int[]{s,cidx,len}, 1.0f); 
				//initializationInput.putScalar(new int[]{j,idx}, 1.0f);
			}
		}
		
		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) sb[i] = new StringBuilder(initialization);
		
		//Sample from network (and feed samples back into input) one character at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		//(dimension 2 size)-1 , 1 ,0  --> 
		//?
		//index : initialization.length() -1 , dim : 1, 0 
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output
		//
		for( int i=0; i<charactersToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				//number of char
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				//取得所有char 的 output prob. distribution.
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);
				//System.out.println( "-->"+  iter.convertIndexToCharacter(sampledCharacterIdx));
				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToCharacter(sampledCharacterIdx));	//Add sampled character to StringBuilder (human readable output)
			}
			
			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}
		
		String[] out = new String[numSamples];
		for( int i=0; i<numSamples; i++ ) out[i] = sb[i].toString();
		return out;
	}
	/** Given a probability distribution over discrete classes, sample from the distribution
	 * and return the generated class index.
	 * @param distribution Probability distribution over classes. Must sum to 1.0
	 */
	private static int sampleFromDistribution( double[] distribution, Random rng ){
		double d = rng.nextDouble(); //0~1.0
		//if(d < 0.5) d *= 2 ;
		
		double sum = 0.0;
		double maxp = 0.0;
		int maxi = 0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			
			if( d <= sum ) {
				
				//输出差异值，确认显著性
				double dist = distribution[i] - sum/(i+1) ;
				if (distribution[i] > maxi){
					maxi = i;
					maxp = distribution[i] ;
				}
				
				if(dist > 0.3){
					System.out.println("Dist distance:" + sum + ", select distingushed");
					return i;
				}
				
			}
		}
		System.out.println("Max Prob:" + maxp + ", select Max Prob.");
		return maxi ; // no
		
		//Should never happen if distribution is a valid probability distribution
		//throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
}
