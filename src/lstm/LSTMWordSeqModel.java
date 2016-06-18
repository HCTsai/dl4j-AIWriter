package lstm;



import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import java.io.UnsupportedEncodingException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.jblas.DoubleMatrix;
public class LSTMWordSeqModel implements java.io.Serializable {

	
	public MultiLayerNetwork net ;
	//public CharacterIterator ci ;
	
	public void StoreModeltoB64File(MultiLayerNetwork net, String fn) {
     	//this.net = net;
	   try{
			String NetStr = ObjtoB64((Serializable)net);
			PrintWriter out = new PrintWriter(fn);
			out.println(NetStr);
			out.close();
			
		}catch(Exception ex){
			System.out.println(ex.toString());
		}
		
	}
	
	public MultiLayerNetwork ReadModelFromB64File(String fn) {
		MultiLayerNetwork m = null;
		try{
			
		//BufferedReader in = new BufferedReader(new FileReader(fn));  
		//String Bas64EnStr = in.readLine() ;
			
		String Bas64EnStr = readFileStr(fn);
		m = (MultiLayerNetwork) B64toObj(Bas64EnStr);	    
	
		
		//in.close();
		}catch(Exception ex){
			
		}
		
		return m;
	}
	
	public void StoreModeltoFile(MultiLayerNetwork net, String fn) throws IOException {
     
		FileOutputStream fs = new FileOutputStream(fn);
		ObjectOutputStream os = new ObjectOutputStream(fs);
		os.writeObject(net);
		os.close();
	}
	
	public MultiLayerNetwork ReadModelFromFile(String fn) throws IOException, ClassNotFoundException{
		 
		FileInputStream fi=new FileInputStream(fn);
		ObjectInputStream oi=new ObjectInputStream(fi);
		MultiLayerNetwork m=(MultiLayerNetwork)oi.readObject();
		oi.close();
		
		return m;
	}
	
	public String readFileStr(String filePath) {
		String fileContent = "";
                //目标地址
		File file = new File(filePath);
		if (file.isFile() && file.exists()) {
			try {
				InputStreamReader read = new InputStreamReader(
						new FileInputStream(file), "UTF-8");
				BufferedReader reader = new BufferedReader(read);
				String line;
				try {
                                        //循环，每次读一行
					while ((line = reader.readLine()) != null) {
						fileContent += line;
					}
					reader.close();
					read.close();
				} catch (IOException e) {
					e.printStackTrace();
				}

			} catch (UnsupportedEncodingException e) {
				e.printStackTrace();
			} catch (FileNotFoundException e) {
				e.printStackTrace();
			}
		}
		return fileContent;
	}

	
    /** Read the object from Base64 string. */
	private static Object B64toObj(String s) {
		
		Object o = null;
		try{
        byte [] data = Base64Coder.decode( s );
        ObjectInputStream ois = new ObjectInputStream( 
                                        new ByteArrayInputStream(  data ) );
        o = ois.readObject();
        ois.close();
		}catch(Exception ex){
			
		}
        return o;
   }

    /** Write the object to a Base64 string. */
    private static String ObjtoB64( Serializable o )  {
    	String str = "";
    	
    	try{
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ObjectOutputStream oos = new ObjectOutputStream( baos );
        oos.writeObject( o );
        oos.close();
        str = new String(Base64Coder.encode(baos.toByteArray()));
        
    	}catch (Exception ex){
    		
    	}
        return str ;
    }
}
