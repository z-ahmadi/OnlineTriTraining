import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import weka.core.Instances;

public class makeSemiDataset {
	static Instances labeledInst, unlabeledInst;
	//labeled dataset --> unlabeled dataset + labeled dataset
	//args[0] = name of input dataset
	//args[1] = probability of labeled data 
	public static void makeSSDataset(Instances inst, double p) throws IOException {
//		BufferedReader in = new BufferedReader(new FileReader(args[0]));
//		Instances instances = new Instances(in);
		Instances instances = new Instances(inst); 
		instances.setClassIndex(instances.numAttributes()-1);
//		in.close();
		
//		in = new BufferedReader(new FileReader(args[0]));
		labeledInst = new Instances(inst);
		labeledInst.delete();
//		System.out.println(labeledInst.numInstances()+" "+labeledInst.numAttributes());
//		in.close();
		
//		in = new BufferedReader(new FileReader(args[0]));
		unlabeledInst = new Instances(inst);
		unlabeledInst.delete();
//		System.out.println(unlabeledInst.numInstances()+" "+unlabeledInst.numAttributes());
//	    in.close();
		
//	    double labeledProb = new Double(args[1]);
		double labeledProb = p;
		int numLabeled = (int) Math.floor(labeledProb*instances.numInstances());
		int numUnlabeled = (int) Math.floor((1-labeledProb)*instances.numInstances());
		//random sampling 
		Random rand = new Random();
		int counter = 0;
		boolean[] flagSelected = new boolean[instances.numInstances()];
		
		while(counter < numLabeled){
			double d = rand.nextDouble();
			int indx = (int) Math.floor(d*instances.numInstances());
			if(!flagSelected[indx]){
				labeledInst.add(instances.instance(indx));
				flagSelected[indx] = true;
				counter++;
			}
		}
		for(int i=0;i<instances.numInstances();i++){
			if(!flagSelected[i]){
				unlabeledInst.add(instances.instance(i));
			}
		}

		labeledInst.setClassIndex(instances.numAttributes()-1);
		unlabeledInst.setClassIndex(instances.numAttributes()-1);
//		unlabeledInst.deleteAttributeAt(instances.numAttributes()-1);

//		System.out.println(instances.numInstances()+" "+instances.numAttributes());
//		System.out.println(labeledInst.numInstances()+" "+labeledInst.numAttributes());
//		System.out.println(unlabeledInst.numInstances()+" "+unlabeledInst.numAttributes());
	}
	public static Instances getLabeledInst() {
		return labeledInst;
	}
	public static Instances getUnlabeledInst() {
		return unlabeledInst;
	}

}
