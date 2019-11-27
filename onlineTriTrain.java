import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Random;

import moa.classifiers.HoeffdingTree;
import moa.core.Measurement;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class onlineTriTrain extends moa.classifiers.AbstractClassifier{
		
	  static PrintWriter pw;
	  private Classifier m_baseClassifier = null;
	  private Classifier[] m_classifiers;
	  final int numClassifier = 3;

	  private Random m_rand = new Random(1);

	  public onlineTriTrain(){}

	  public void setClassifier(Classifier c){
	    m_baseClassifier = c;
	  }
//	  public void setClassifier(AbstractClassifier c){
//		    m_baseClassifier = c;
//		  }

	  public void setRandomObject(Random random){
	    m_rand = random;
	  }
	  
	  public void buildIncrementalClassifier(Instances labeled, Instances unlabeled, double[] err_prime,
			  double[] s_prime) throws Exception{
		    double[] err = new double[numClassifier];             // e_i
		    Instances[] labeleds = new Instances[numClassifier];
		    
		    for(int i = 0; i < numClassifier; i++){
			      labeleds[i] = new Instances(labeled.resampleWithWeights(m_rand));     //L_i <-- Bootstrap(L)
			      m_classifiers[i].buildClassifier(labeleds[i]);                        //h_i <-- Learn(L_i)
		    }
		    
//		    boolean bChanged = true;

		    /** repeat until none of h_i ( i \in {1...numClassifier} ) changes */
//		    while(bChanged){
//		      bChanged = false;
		      boolean[] bUpdate = new boolean[m_classifiers.length];
		      Instances[] Li = new Instances[numClassifier];

		      /** for i \in {1...numClassifier} do */
		      for(int i = 0; i < numClassifier; i++){
		        Li[i] = new Instances(labeled, 0);         //L_i <-- \phi
		        err[i] = measureError(labeled, i);         //e_i <-- MeasureError(h_j & h_k) (j, k \ne i)

		        /** if (e_i < e'_i) */
		        if(err[i] < err_prime[i]){
//		        if(err[i] < 0.5){
		          /** for every x \in U do */
		          for(int j = 0; j < unlabeled.numInstances(); j++){
		            Instance curInst = new Instance(unlabeled.instance(j));
		            curInst.setDataset(Li[i]);
//		            System.out.println(j);
		            double classval = m_classifiers[(i+1)%numClassifier].classifyInstance(curInst);

		            /** if h_j(x) = h_k(x) (j,k \ne i) */
		            if(classval == m_classifiers[(i+2)%numClassifier].classifyInstance(curInst)){
		              curInst.setClassValue(classval);
		              Li[i].add(curInst);                //L_i <-- L_i \cup {(x, h_j(x))}
		            }
		          }// end of for j

		          /** if (l'_i == 0 ) */
		          if(s_prime[i] == 0){
		            s_prime[i] = Math.floor(err[i] / (err_prime[i] - err[i]) + 1);   //l'_i <-- floor(e_i/(e'_i-e_i) +1)
		          }
		          /** if (l'_i < |L_i| ) */
		          if(s_prime[i] < Li[i].numInstances()){
//		        	  System.out.println("hello l'=0");
		            /** if ( e_i * |L_i| < e'_i * l'_i) */
		            if(err[i] * Li[i].numInstances() < err_prime[i] * s_prime[i])
		              bUpdate[i] = true;                                          // update_i <-- TURE

		            /** else if (l'_i > (e_i / (e'_i - e_i))) */
		            else if (s_prime[i] > (err[i] / (err_prime[i] - err[i]))){
		            	System.out.println("subsample");
		              int numInstAfterSubsample = (int) Math.ceil(err_prime[i] * s_prime[i] / err[i] - 1);
		              Li[i].randomize(m_rand);
		              Li[i] = new Instances(Li[i], 0, numInstAfterSubsample);         //L_i <-- Subsample(L_i, ceilling(e'_i*l'_i/e_i-1)
		              bUpdate[i] = true;                                              //update_i <-- TRUE
		            }
		            else{
		            	System.out.println("aaaaaaaaaaaaaaaaaaaa");
		            }
		          }
		        }
		      }//end for i = 1...numClassifier

		      //update
		      for(int i = 0; i < numClassifier; i++){
		        /** if update_i = TRUE */
		        if(bUpdate[i]){
		          int size = Li[i].numInstances();
//		          bChanged = true;
		          m_classifiers[i].buildClassifier(combine(labeled, Li[i]));        //h_i <-- Learn(L \cup L_i)
		          err_prime[i] = err[i];                                            //e'_i <-- e_i
		          s_prime[i] = size;                                                //l'_i <-- |L_i|
		        }
		      }// end fo for
//		    } //end of repeat
	  }

	  public void buildClassifier(Instances dataset, int window, double p) throws Exception {
	    double[] err_prime = new double[numClassifier];       // e'_i
	    double[] s_prime = new double[numClassifier];         // l'_i

	    if (m_baseClassifier == null)
	      throw new Exception("Base classifier should be set before the building process");

	    if (!dataset.classAttribute().isNominal())
	      throw new Exception("The class value should be nominal");

	    m_classifiers = Classifier.makeCopies(m_baseClassifier, numClassifier);
	    
	    for(int i = 0; i < numClassifier; i++){
		      err_prime[i] = 0.5;                                                   //e'_i <-- .5
		      s_prime[i] = 0;                                                       //l'_i <-- 0
		      }
	    
	    //make window on dataset. then divide it into labeled and unlabeled data.
	    //while there are some window on data
	    int numberWindow = 0; 
	    boolean continu = true;
	    double totalError = 0.0;
	    while(continu){
	    	 Instances windowData = makeWindow(dataset, window, numberWindow);
	    	 makeSemiDataset mkds = new makeSemiDataset();
	    	 mkds.makeSSDataset(windowData, p);
	    	 Instances labeled = mkds.getLabeledInst();
	    	 Instances unlabeled = mkds.getUnlabeledInst();
	    	 Instances test = mkds.getLabeledInst();
//	    	 Instances test = windowData;
	    	 test.setClassIndex(test.numAttributes()-1);
	    	 
	    	 if(numberWindow>=3){
	    		 double err = 0;
			      for(int i = 0; i < test.numInstances(); i++){
//			    	  pw.write(test.instance(i).classValue()+" "+classifyInstance(test.instance(i))+"\n");
			        if (classifyInstance(test.instance(i)) != test.instance(i).classValue()){
				    	  pw.write(test.instance(i).classValue()+" "+classifyInstance(test.instance(i))+"\n");
				    	  err += 1.0;
			        }
			      }
			      err /= test.numInstances();
			      totalError += err;

			      System.out.println(numberWindow+"  " +(1-err)*100+" "+(1-(totalError/(numberWindow-2)))*100);
	    	 }
	    	 
	    	 
	 	     buildIncrementalClassifier(labeled, unlabeled, err_prime, s_prime);
	    	 
	 	     if((numberWindow+1)*window >= dataset.numInstances())
	 	    	 continu = false;
	 	     else
	 	    	 numberWindow++;
	    	 
	    }
	    System.out.println("* "+(1-(totalError/(numberWindow-2)))*100); 
	    
	  }
	  
	  public Instances makeWindow(Instances dataset, int window, int number){
		  Instances output = new Instances(dataset);
		  output.delete();
		  int start = number*window;
		  int end = (number+1)*window;
		  if(dataset.numInstances() < end)
			  end = dataset.numInstances();
		  for(int i=start; i<end; i++)
			  output.add(dataset.instance(i));
		  
		return output;
	  }
	  
	  public double[] distributionForInstance(Instance inst) throws Exception {
//		  System.out.println(inst.numClasses());
	    double[] res = new double[inst.numClasses()];
	    for(int i = 0; i < m_classifiers.length; i++){
	      double[] distr1 = m_classifiers[i].distributionForInstance(inst);
	      double[] distr = new double[inst.numClasses()];
	      for(int k=0;k<distr1.length; k++)
	    	  distr[k] = distr1[k];
//	      System.out.println(distr.length);
	      for(int j = 0; j < res.length; j++)
	        res[j] += distr[j];
	    }
	    Utils.normalize(res);
	    return res;
	  }

	  public double classifyInstance(Instance inst) throws Exception {
	    double[] distr = distributionForInstance(inst);	    
	    return Utils.maxIndex(distr);
	  }

	  /**
	   * Adds the instances in initial training set L to the newly labeled set Li
	   *
	   * @param L Instances The initial training set
	   * @param Li Instances The newly labeled set
	   * @return Instances The combined data set
	   */
	  private Instances combine(Instances L, Instances Li){
	    for(int i = 0; i < L.numInstances(); i++)
	      Li.add(L.instance(i));

	    return Li;
	  }

	  /**
	   * Measure combined error excluded the classifier 'id' on the given data set
	   *
	   * @param data Instances The data set
	   * @param id int The id of classifier to be excluded
	   * @return double The error
	   * @throws Exception Some Exception
	   */
	  protected double measureError(Instances data, int id) throws Exception{
		  Classifier[] c = new Classifier[numClassifier-1];
		  for(int i=0; i<c.length;i++)
			  c[i] = m_classifiers[(id+i+1)%numClassifier];
	    double err = 0;
	    int count = 0;

	    for(int i = 0; i < data.numInstances(); i++){
	      double[] prediction = new double[numClassifier-1]; 
	      ArrayList<Double> predArray = new ArrayList<Double>();
	      ArrayList<Integer> predNums = new ArrayList<Integer>();
	      
	      for(int k=0;k<prediction.length;k++){
	    	  prediction[k] = c[k].classifyInstance(data.instance(i));
	    	  if(predArray.contains(prediction[k])){
	    		  int ind = predArray.indexOf(prediction[k]);
	    		  predNums.set(ind, predNums.get(ind)+1);
	    	  }else{
	    		  predArray.add(prediction[k]);
	    		  predNums.add(1);
	    	  }
	      }
	      
	      int max = predNums.get(0);
	      int indmax = 0;
	      for(int k=0;k<predNums.size();k++){
	    	  if(max < predNums.get(k)){
	    		  max = predNums.get(k);
	    		  indmax = k;
	    	  }
	      }
	      if(max >= Math.floor(numClassifier/2)+1){
	    	  count++;
	    	  if(predArray.get(indmax) != data.instance(i).classValue())
	    		  err += 1;
	      }
	    }

	    err /= count;
	    return err;
	  }

	  public static void main(String[] args){
	    try {
			//building a semi-supervised dataset
			
			
			//building the classifier
//			TriTrain tri = new TriTrain();
//			HoeffdingTree hoftree = new HoeffdingTree();
//			setClassifier(hoftree);
//	    	makeSemiDataset mkds = new makeSemiDataset();
//	    	mkds.makeSSDataset(new Instances(new BufferedReader(new FileReader("datasets/usenet1.arff"))), p);
//	    	Instances labeled = mkds.getLabeledInst();
//	    	Instances unlabeled = mkds.getUnlabeledInst();
//	    	Instances test = mkds.getUnlabeledInst();
//	    	test.setClassIndex(test.numAttributes()-1);

	      J48 dt = new J48();
	      dt.setUnpruned(true);
//	      TriTrain tri = new TriTrain();
//	      tri.setClassifier(dt);
//	      tri.buildClassifier(labeled, unlabeled);
	    	
	    	pw = new PrintWriter(new FileWriter("kddOut.txt"));
	    	onlineTriTrain ott = new onlineTriTrain();
	    	Instances ins = new Instances(new BufferedReader(new FileReader(args[0])));
	    	ins.setClassIndex(ins.numAttributes()-1);
	    	IncrementalClassifier inHoftree = new IncrementalClassifier(); 
//	    	ott.setClassifier(inHoftree);
	    	ott.setClassifier(dt);
	    	ott.buildClassifier(ins, new Integer(args[1]), new Double(args[2]));
//	      double err = 0;
//	      for(int i = 0; i < test.numInstances(); i++){
//	        if (tri.classifyInstance(test.instance(i)) != test.instance(i).classValue())
//	          err += 1.0;
//	      }
//	      err /= test.numInstances();
//
//	      System.out.println("Error rate = " + err);
	    }
	    catch(Exception e){
	      e.printStackTrace();
	    }
	    pw.close();
	  }

	 
	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void resetLearningImpl() {
		// TODO Auto-generated method stub
//		tritrain.setClassifier(m_baseClassifier);
//		try {
//			m_classifiers = Classifier.makeCopies(m_baseClassifier, numClassifier);
//		} catch (Exception e) {
//			// TODO Auto-generated catch block
//			System.out.println("Unable to make copy of base classifiers!");
//			e.printStackTrace();
//		}
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public boolean isRandomizable() {
		return false;
	}	
}
