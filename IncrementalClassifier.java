import moa.classifiers.HoeffdingTree;
import moa.classifiers.NaiveBayes;
import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class IncrementalClassifier extends Classifier{
	HoeffdingTree hoftree = new HoeffdingTree();
//	NaiveBayes hoftree = new NaiveBayes();

	@Override
	public void buildClassifier(Instances inst) throws Exception {
		// TODO Auto-generated method stub
		for(int i=0; i<inst.numInstances(); i++){
//			System.out.print(inst.instance(i).classValue()+" ");
			hoftree.trainOnInstanceImpl(inst.instance(i));			
		}
//		System.out.println();
	}
	
	public double classifyInstance(Instance inst){
		double[] votes = hoftree.getVotesForInstance(inst);
		double max = votes[0];
		for(int i=0;i<votes.length;i++){
			if(max > votes[i])
				max = votes[i];
		}
		
		return max;
	}
	
	public double[] distributionForInstance(Instance inst){
		double[] votes = hoftree.getVotesForInstance(inst);
		return votes;
	}

}
