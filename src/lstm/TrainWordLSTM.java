package lstm;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class TrainWordLSTM {

	public static void main(String[] args) throws IOException, ClassNotFoundException {
		
		double start = System.currentTimeMillis();
		int lstmLayerSize = 20;//200;				//Number of units in each GravesLSTM layer
		int miniBatchSize =40;// 32;				//Size of mini batch to use when training, subset of samples.
		int examplesPerEpoch = 20*miniBatchSize;//50 * miniBatchSize;	//i.e., how many examples to learn on between generating samples
		int exampleLength =  9;//  100;					//Length of each training example
		int numEpochs = 500;//30;							//Total number of training + sample generation epochs
		int nSamplesToGenerate = 1;					//Number of samples to generate after each training epoch
		int nCharactersToSample = 50;				//Length of each sample to generate
		
		//Example ("中文 中文 中文") separate by space 
		String generationInitialization = "家电"; //null;		//Optional character initialization; a random character is used if null
	
		Random rng = new Random(12345);
		
		String filepath = "data/segres_patent_jay.txt";
		String modelfile = "data/model_patent_jay.txt";
		String resfile = "data/res_patent.txt" ;
		
		WordIterator iter = GetWordIterbyFile(miniBatchSize,exampleLength,examplesPerEpoch,filepath);
		
		int nOut = iter.totalOutcomes();
		
		//Set up network configuration:
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
			.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
			.learningRate(0.2)
			.rmsDecay(0.95)
			.seed(12345)
			.regularization(true)
			.l2(0.001)
			.list(5)
			.layer(0, new GravesLSTM.Builder().nIn(iter.inputColumns()).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(2, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(3, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
					.updater(Updater.RMSPROP)
					.activation("tanh").weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.layer(4, new RnnOutputLayer.Builder(LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
					.updater(Updater.RMSPROP)
					.nIn(lstmLayerSize).nOut(nOut).weightInit(WeightInit.DISTRIBUTION)
					.dist(new UniformDistribution(-0.08, 0.08)).build())
			.pretrain(false).backprop(true)
			.build();
		
		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		net.setListeners(new ScoreIterationListener(1));
		
		//Print the  number of parameters in the network (and for each layer)
		Layer[] layers = net.getLayers();
		int totalNumParams = 0;
		for( int i=0; i<layers.length; i++ ){
			int nParams = layers[i].numParams();
			System.out.println("Number of parameters in layer " + i + ": " + nParams);
			totalNumParams += nParams;
		}
		System.out.println("Total number of network parameters: " + totalNumParams);
		
		
		//Do training, and then generate and print samples from network
		
		LSTMWordSeqModel m = new LSTMWordSeqModel();
		
		for( int i=0; i<numEpochs; i++ ){			
					
			File mf = new File(modelfile);
			     
			if(mf.exists()){    //existed pre-trained file
				net = m.ReadModelFromFile(modelfile);
				System.out.println("Load previous model: " + modelfile);
			}
			
			net.fit(iter);   //training 
					
			System.out.println("--------------------");
			System.out.println("Completed epoch " + i );
			System.out.println("Sampling characters from network given initialization \"" + ("") + "\"");
					
			//sampling
			List<String> samples = sampleWordsFromNetwork(generationInitialization,net,iter,rng,nCharactersToSample,nSamplesToGenerate);
			
			for( int j=0; j<samples.size(); j++ ){
				System.out.println("----- Sample " + j + " -----");
						System.out.println(samples.get(j));
						
						//将结果输出
			        	double dur = (System.currentTimeMillis() - start) / 1000 ;
			            String oStr = samples.get(j).trim() + "-" + dur ;
						StoreString2File(oStr, resfile);
						//System.out.println();
			}
					
			iter.reset();	//Reset iterator for another epoch
			try{
						m.StoreModeltoFile(net, modelfile);
			}catch(Exception ex){
						System.out.println(ex.toString());
			}
		}
				
		System.out.println("\n\nExample complete");
	}
	
	public static WordIterator  GetWordIterbyFile(int miniBatchSize, int exampleLength, int examplesPerEpoch,String filepath) throws IOException{
		List<String> validWords = WordIterator.getChineseWordSet(filepath);
		
		return new WordIterator(filepath, Charset.forName("UTF-8"),
				miniBatchSize, exampleLength, examplesPerEpoch, validWords, new Random(12345),true);
	}
	
	private static List<String> sampleWordsFromNetwork( String initialization, MultiLayerNetwork net,
			WordIterator iter, Random rng, int wordsToSample, int numSamples ){
		//Set up initialization. If no initialization: use a random character
		
		if( initialization == null ){
			initialization = String.valueOf(iter.getRandomWord());
		}
		//
		
		//Create input for initialization
		INDArray initializationInput = Nd4j.zeros(numSamples, iter.inputColumns(), initialization.length());
		
		// How to split user input to word sequence ? auto segmentation ?
		String[] init = initialization.split(" ");
		
		for( int i=0; i<init.length; i++ ){
			
			int widx = 0 ;
			
			try{
				widx = iter.convertWordToIndex(init[i]);//idx may be null
			}
			catch(Exception ex){
				widx = ((int) (rng.nextDouble() * iter.numWords())) ;
			}
			for( int j=0; j<numSamples; j++ ){
				initializationInput.putScalar(new int[]{j,widx,i}, 1.0f);
			}
		}
		
		StringBuilder[] sb = new StringBuilder[numSamples];
		for( int i=0; i< numSamples; i++ ) sb[i] = new StringBuilder(initialization);
		
		//Sample from network (and feed samples back into input) one character at a time (for all samples)
		//Sampling is done in parallel here
		net.rnnClearPreviousState();
		INDArray output = net.rnnTimeStep(initializationInput);
		output = output.tensorAlongDimension(output.size(2)-1,1,0);	//Gets the last time step output
		
		for( int i=0; i<wordsToSample; i++ ){
			//Set up next input (single time step) by sampling from previous output
			INDArray nextInput = Nd4j.zeros(numSamples,iter.inputColumns());
			//Output is a probability distribution. Sample from this for each example we want to generate, and add it to the new input
			for( int s=0; s<numSamples; s++ ){
				
				double[] outputProbDistribution = new double[iter.totalOutcomes()];
				for( int j=0; j<outputProbDistribution.length; j++ ) outputProbDistribution[j] = output.getDouble(s,j);
				int sampledWordIdx = sampleFromDistribution(outputProbDistribution,rng);
				
				nextInput.putScalar(new int[]{s,sampledWordIdx}, 1.0f);		//Prepare next time step input
				sb[s].append(iter.convertIndexToWord(sampledWordIdx));	//Add sampled character to StringBuilder (human readable output)
			}
			
			output = net.rnnTimeStep(nextInput);	//Do one time step of forward pass
		}
		
		//String[] out = new String[numSamples];
		List<String> outlist =  new ArrayList<String>();
		for( int i=0; i<numSamples; i++ ) {
			outlist.add(sb[i].toString());
						
		}
		return outlist;
	}
	
	private static int sampleFromDistribution( double[] distribution, Random rng ){
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
	
	public static void StoreString2File(String s, String fn) {
		try{
			FileWriter fw = new FileWriter(fn, true); //append
			//PrintWriter out = new PrintWriter(fn);
			PrintWriter out = new PrintWriter(fw);
			out.println(s);
			out.close();
			
		}catch(Exception ex){
			System.out.println(ex.toString());
		}
		
	}

}
