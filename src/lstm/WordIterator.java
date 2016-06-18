package lstm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

public class WordIterator implements DataSetIterator{

	private static final long serialVersionUID = -7287833919126626356L;
	private static final int MAX_SCAN_LENGTH = 200; 
	

	
	private List<String> validWords = new ArrayList<String>();
	private Map<String,Integer> Word2IdxMap;
	private int numWords =0 ;
	private List<String> fileWords =  new ArrayList<String>();
	

	private int exampleLength;
	private int miniBatchSize;
	private int numExamplesToFetch;
	private int examplesSoFar = 0;
	private Random rng;
	
	private final boolean alwaysStartAtNewLine;
	

	

	public WordIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
			int numExamplesToFetch, List<String> validWords, Random rng, boolean alwaysStartAtNewLine ) throws IOException {
		
		if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
		if(numExamplesToFetch % miniBatchSize != 0 ) throw new IllegalArgumentException("numExamplesToFetch must be a multiple of miniBatchSize");
		if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
		//this.validCharacters = validCharacters;
		
		this.validWords = validWords;
		
		this.exampleLength = exampleLength;
		this.miniBatchSize = miniBatchSize;
		this.numExamplesToFetch = numExamplesToFetch;
		this.rng = rng;
		this.alwaysStartAtNewLine = alwaysStartAtNewLine;
		
		//Store valid characters is a map for later use in vectorization
		
		Word2IdxMap = new HashMap<>();
		for( int i=0; i<validWords.size(); i++ ) Word2IdxMap.put(validWords.get(i), i);
	
		numWords = validWords.size();
		
		//Load file and convert contents to a List<String> 
		boolean newLineValid = Word2IdxMap.containsKey('\n');
		
		List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
		
				
		List<String> strList = new ArrayList<String>();
		
		int currIdx = 0;
		int remove = 0;
		for( String s : lines ){
			
			s += " |" ; // line signal
			String[] lineWords = s.split(" ");
			
			for (String lw : lineWords) {
				
				if (! Word2IdxMap.containsKey(lw))  { 
					remove++;
					continue;
				}
				strList.add(lw);
				
			}
			
			if(newLineValid) {
				
				strList.add("\n");
				
			}else{
				//characters[currIdx++] = ' ';
			}
			
		}
		
	
		fileWords = strList;
	
		if( exampleLength >= fileWords.size() ) 
			throw new IllegalArgumentException("exampleLength="+exampleLength
				+" cannot exceed number of valid Words in file ("+fileWords.size()+")");
		
		
		System.out.println("Loaded and converted file: " + fileWords.size() + " valid words "
		 + "(" + remove + " removed)");
	}
	
	
	public static char[] getChineseCharSet(String filename) throws IOException{
		
		String resStr ="";
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		while( (line = br.readLine()) != null) {  //
			
		  // line = new String(line.getBytes("iso-8859-1"), "utf-8"); 
			
			resStr += line ; 
		}
		br.close();
		resStr += " " ; // add sentiment separator
		resStr += "|" ; // add line separator
		return resStr.toCharArray();
		
	}
	public static List<String> getChineseWordSet(String filename) throws IOException{
		
		List<String> resList = new ArrayList<String>();
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		String line = null;
		while( (line = br.readLine()) != null) {  //
			
		  //line = new String(line.getBytes("iso-8859-1"), "utf-8"); 
			String[] words = line.split(" ");
			for(String w : words){
				resList.add(w); 
			}
		}
		br.close();
		
		resList.add(" "); // add sentiment separator
		resList.add("|"); // add line separator
		return resList;
		
	}
	/** A minimal character set, with a-z, A-Z, 0-9 and common punctuation etc */
	public static char[] getMinimalCharacterSet(){
		
		List<Character> validChars = new java.util.LinkedList<>();
		for(char c='a'; c<='z'; c++) validChars.add(c);
		for(char c='A'; c<='Z'; c++) validChars.add(c);
		for(char c='0'; c<='9'; c++) validChars.add(c);
		char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
		for( char c : temp ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}
	/** As per getMinimalCharacterSet(), but with a few extra characters */
	public static char[] getDefaultCharacterSet(){
		List<Character> validChars = new LinkedList<>();
		for(char c : getMinimalCharacterSet() ) validChars.add(c);
		char[] additionalChars = {'@', '#', '$', '%', '^', '*', '{', '}', '[', ']', '/', '+', '_',
				'\\', '|', '<', '>'};
		for( char c : additionalChars ) validChars.add(c);
		char[] out = new char[validChars.size()];
		int i=0;
		for( Character c : validChars ) out[i++] = c;
		return out;
	}
	
	public String convertIndexToWord( int idx ){
		return validWords.get(idx);
	}
	
	public int convertWordToIndex( String c ){
		
		
				
		return Word2IdxMap.get(c);
	}
	
	public String getRandomWord(){
		return validWords.get((int)(rng.nextDouble()*validWords.size()));
	}

	public boolean hasNext() {
		
		return examplesSoFar + miniBatchSize <= numExamplesToFetch;
		
	}
	
	public DataSet next() {
		return next(miniBatchSize);
	}

	public DataSet next(int num) {
		
		if( examplesSoFar+num > numExamplesToFetch ) throw new NoSuchElementException();
		
		
		//Allocate space:
		//mini-batch , num Characters, Length of each training example
		INDArray input = Nd4j.zeros(num,numWords,exampleLength);
		INDArray labels = Nd4j.zeros(num,numWords,exampleLength);
		
		int maxStartIdx = fileWords.size() - exampleLength;
		
		
		//Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
		// of the file in the same minibatch
		for( int i=0; i < num; i++ ){
			
			
			int startIdx = (int) (rng.nextDouble()*maxStartIdx);
			int endIdx = startIdx + exampleLength;
			int scanLength = 0;
			
			//default is true; samples start at sentence head
			if(alwaysStartAtNewLine){
				while(startIdx >= 1 && !fileWords.get(startIdx-1).equals("\n") && scanLength++ < MAX_SCAN_LENGTH ){
					startIdx--;
					endIdx--;
				}
			}
			
			int currWordIdx = Word2IdxMap.get(fileWords.get(startIdx));	//Current input
			int c = 0; // example length index
			
			for(int j=startIdx+1; j<=endIdx; j++, c++ ){
				
				int nextWordIdx = Word2IdxMap.get(fileWords.get(j));		//Next character to predict
				
				input.putScalar(new int[]{i,currWordIdx,c}, 1.0);
				labels.putScalar(new int[]{i,nextWordIdx,c}, 1.0);
				
				currWordIdx = nextWordIdx;
			}
		}
		
		examplesSoFar += num;
		return new DataSet(input,labels);
	}

	public int totalExamples() {
		return numExamplesToFetch;
	}

	public int inputColumns() {
		return numWords;
	}

	public int numWords() {
		return numWords;
	}

	public void reset() {
		examplesSoFar = 0;
	}

	public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return examplesSoFar;
	}

	public int numExamples() {
		return numExamplesToFetch;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

	@Override
	public int totalOutcomes() {
		// TODO Auto-generated method stub
		return numWords;
	}
}
