package lstm;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class AIWordsWriter {
	//int senLen = rng.nextInt((5 - 3) + 1) + 3;
	static int senLen = 8;
	static int wCount = 0;
	public static void main(String[] args) throws ClassNotFoundException, IOException {

		int miniBatchSize =12;// 32;	//Size of mini batch to use when  training
		int examplesPerEpoch = 15*miniBatchSize;//50 * miniBatchSize;	//i.e., how many examples to learn on between generating samples
		int exampleLength = 40;//  100;					//Length of each training example
        
		//Get Network
		String filepath = "data/segres_patent_jay.txt";
		String modelfile = "data/model_patent_jay.txt";
		//modelfile = "C:\\data\\model.txt";
		//File mf = new File(modelfile);
		LSTMWordSeqModel m = new LSTMWordSeqModel();
		MultiLayerNetwork net = m.ReadModelFromFile(modelfile);
		
		//Get CharacterIterator
		
		WordIterator iter = GetWordIterbyFile(miniBatchSize,exampleLength,examplesPerEpoch,filepath);
		
		int nOut = iter.totalOutcomes();
		
		//Prepare writing 
		int numSamples = 1;
		int wordToSample = 150;
		String initStr ="空调";
		long rnd = System.currentTimeMillis();
		
		String[] samples = sampleWordsFromNetwork(initStr , net,
				iter, new Random(rnd), wordToSample, numSamples );
		//writing
		
		for(String s : samples){
			//s = s.replace("|", ",");
			String[] Sentence= s.split("\\|");
			int snum = Sentence.length ; // Sentence.length-1 
			for(int i = 0 ; i < snum ;i++){  //ignore last line
			
				 System.out.println(Sentence[i]);
				
			}
			
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
	private static String[] sampleWordsFromNetwork(String initialization, MultiLayerNetwork net,
			WordIterator iter, Random rng, int charactersToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		if( initialization == null){
			initialization = String.valueOf(iter.getRandomWord());
		}
		
		String[] init = initialization.split(" ");
		
		
		
		//Create input for initialization
		//numSamples, numCharacters, inputLength
		//shape -> dimensions
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
		//initializationInput = Nd4j.zeros(numSamples, iter.inputColumns());
		//INDArray i2 = Nd4j.zeros(numSamples, iter.inputColumns());
	
		
		//INDArray ii = Nd4j.zeros(2, 3, 4);
		
		for( int len=0; len<init.length; len++ ){
			
			int widx =  0; 
			try{
				widx = iter.convertWordToIndex(init[len]);
			}catch(Exception ex){
				widx = ((int) (rng.nextDouble() * iter.numWords())) ;
			}
	       
			
			for( int s=0; s<numSamples; s++ ){
				
				//insert 1.0 at [numSample, charIdx, all_initStr]
				//multi word input --> single word output
				initializationInput.putScalar(new int[]{s,widx,len}, 1.0f); 
				//initializationInput.putScalar(new int[]{j,idx}, 1.0f);
			}
		}
		
		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i<numSamples; i++ ) {
			
			sb[i] = new StringBuilder();
			for(String s : init) sb[i].append(s);
		
		}
		
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
				
				//int sampledCharacterIdx = sampleFromDistributionv0(outputProbDistribution,rng);
				int sampledCharacterIdx = sampleFromDistribution(outputProbDistribution,rng);
				/*
				Boolean maxProb = false ;
				
				if (sampledCharacterIdx < 0){
					
					sampledCharacterIdx = 0 -sampledCharacterIdx ;
					maxProb = true ;
				}
				*/
				//System.out.println( "-->"+  iter.convertIndexToCharacter(sampledCharacterIdx));
				nextInput.putScalar(new int[]{s,sampledCharacterIdx}, 1.0f);		//Prepare next time step input
				String word = iter.convertIndexToWord(sampledCharacterIdx);
				
				
			
				
				if(word.equals("|")){
					if(wCount >= senLen){
						sb[s].append(word);
						wCount=0;
					}else{
						sb[s].append(" ");
					}
				
				}else{
					
					if(wCount >= senLen){
						sb[s].append("|");
						wCount = 0;
					}
					
					sb[s].append(word);
					wCount++;
				}
				//sb[s].append(word);	//Add sampled character to StringBuilder (human readable output)
			   // wCount++;
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
	private static int sampleFromDistribution( double[] distribution, Random rng  ){
		
		int resi = -1;
		
		//double d = rng.nextDouble(); //0~1.0
		
		double middle =  rng.nextDouble(); // 0.0~1.0
		double r1 = 0.0 + (middle * rng.nextDouble()); // 0.0~mid
	    double r2 = middle + ( (1-middle) * rng.nextDouble()); // mid~1.0 
		double sum = 0.0;
		double maxp = 0.0;
		int maxi = 0;
       	
		sum = 0.0 ;
		
		for( int i=0; i<distribution.length; i++ ){
			
			sum += distribution[i];
			//fine a probability that distinguished from others.
			double dist = distribution[i] - sum/(i+1) ;
			if(dist > 0.33){ //
			//System.out.println("Dist distance:" + dist + ", select distingushed");
				resi = i ;
		
			}
			
			
			if( (sum >= r1) && (sum <= r2)) {  //Get max Prob. with random range;
				
				if (distribution[i] > maxi){
					maxi = i;
					maxp = distribution[i] ;
				}
							
			}
			
			
		}
		
		if(resi == -1){ // no distinguished Prob. 
			
			//System.out.println("Max Prob:" + maxp + ", select Max Prob.");
			resi = maxi ;
		}
		
		return resi ;
		//Should never happen if distribution is a valid probability distribution
		//throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
	
	private static int sampleFromDistributionv0( double[] distribution, Random rng ){
		double d = rng.nextDouble();
		double sum = 0.0;
		for( int i=0; i<distribution.length; i++ ){
			sum += distribution[i];
			if( d <= sum ) {
				
				return i;
			
			}
		}
		//Should never happen if distribution is a valid probability distribution
		throw new IllegalArgumentException("Distribution is invalid? d="+d+", sum="+sum);
	}
	public static WordIterator  GetWordIterbyFile(int miniBatchSize, int exampleLength, int examplesPerEpoch,String filepath) throws IOException{
		List<String> validWords = WordIterator.getChineseWordSet(filepath);
		
		return new WordIterator(filepath, Charset.forName("UTF-8"),
				miniBatchSize, exampleLength, examplesPerEpoch, validWords, new Random(12345),true);
	}

}
