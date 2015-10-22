package cz.wikkiw.ruggedness;

import cz.wikkiw.fitnessfunctions.FitnessFunction;
import cz.wikkiw.fitnessfunctions.Rosenbrock;
import cz.wikkiw.fitnessfunctions.objects.Boundary;
import cz.wikkiw.fitnessfunctions.objects.Individual;
import cz.wikkiw.prwm.PRWm;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 *
 * @author adam
 */
public class RuggednessEstimation {

    private FitnessFunction ffunction;
    private int dimension;
    private int walkSteps;
    private double stepBoundary;
    
    private Boundary boundary;

    public RuggednessEstimation(FitnessFunction ffunction, int dimension, int walkSteps, double stepBoundary) {
        this.ffunction = ffunction;
        this.dimension = dimension;
        this.walkSteps = walkSteps;
        this.stepBoundary = stepBoundary;
        
        this.boundary = this.ffunction.getBoundary();
    }

    /**
     * 
     * @return 
     */
    public double estimateRuggedness(){
        
        /**
         * First determine stability measure
         */
        Random rnd = new Random();
        int[] startZone = new int[this.dimension];
        for(int i=0; i<this.dimension; i++){
            startZone[i] = rnd.nextInt(2);
        }
        double changeProbability = 0.05;
        PRWm prwm = new PRWm(this.dimension, this.walkSteps, this.stepBoundary, startZone, this.ffunction, changeProbability);
        prwm.walk();
        
        List<Individual> walk = prwm.getWalkIndividuals();
        Individual bestInd = prwm.getBest();
        
        System.out.println(bestInd);
        
        double e = this.countStabilityMeasure(walk);
 
        double[] eTable = new double[]{0, e/128, e/64, e/32, e/16, e/8, e/4, e/2, e};
        double[] entropyTable = new double[9];
        
        /**
         * Code to time series, group counts, count entropy
         */
        int[] timeSerie;
        int[] groupCounts;
        double entropy;
        double maxEntropy = 0;
        
        for(int i=0; i<eTable.length; i++){
            timeSerie = new int[walk.size()-1];
            timeSerie = this.codeToTimeSeries(walk, eTable[i]);
            
//            System.out.println(Arrays.toString(timeSerie));
            
            groupCounts = new int[6];            
            groupCounts = this.getGroupCounts(timeSerie);
            
//            System.out.println(Arrays.toString(groupCounts));
            
            entropy = this.countEntropy(groupCounts, timeSerie.length-1);
            entropyTable[i] = entropy;
            if(entropy > maxEntropy){
                maxEntropy = entropy;
            }

        }
        
        System.out.println(Arrays.toString(entropyTable));
        
        /**
         * Max of entropyTable
         */
        
        
        return maxEntropy; 
    }
    
    /**
     * 
     * @param runCount
     * @return 
     */
    public double estimateRuggedness(int runCount){
    
        double sum = 0;
        
        for(int i=0; i<runCount; i++){
            sum += this.estimateRuggedness();
        }
        
        return sum/(double)runCount;
        
    }
    
    /**
     * 
     * @param groupCounts
     * @param doubles
     * @return 
     */
    private double countEntropy(int[] groupCounts, int doubles){
        
        double entropy = 0;
        double entropyPart;
        double probability;
        
        for(int i=0; i<groupCounts.length; i++){
            
            if(groupCounts[i] > 0){
                probability = groupCounts[i]/(double) doubles;
                entropyPart = (probability * (Math.log(probability)/Math.log(6)));
                entropy += entropyPart;
            }
            
        }
        
        return -entropy;
        
    }
    
    /**
     * 
     * @param timeSerie
     * @return 
     */
    private int[] getGroupCounts(int[] timeSerie){
        
        int[] groupCounts = new int[]{0,0,0,0,0,0};
        
        for(int i=0; i<timeSerie.length-1; i++){
            
            if(timeSerie[i]==0 && timeSerie[i+1]==1){
                groupCounts[0]++;
            }
            else if(timeSerie[i]==0 && timeSerie[i+1]==-1){
                groupCounts[1]++;
            }
            else if(timeSerie[i]==1 && timeSerie[i+1]==0){
                groupCounts[2]++;
            }
            else if(timeSerie[i]==1 && timeSerie[i+1]==-1){
                groupCounts[3]++;
            }
            else if(timeSerie[i]==-1 && timeSerie[i+1]==0){
                groupCounts[4]++;
            }
            else {
                groupCounts[5]++;
            }
            
        }
        
        return groupCounts;
        
    }
    
    /**
     * 
     * @param indList
     * @param e
     * @return 
     */
    private int[] codeToTimeSeries(List<Individual> indList, double e){
        
        int[] timeSerie = new int[indList.size()-1];
        
        for(int i=0; i<indList.size()-1; i++){
            
            if((indList.get(i+1).getFitness() - indList.get(i).getFitness()) < -e){
                timeSerie[i] = -1;
            }
            if(Math.abs(indList.get(i+1).getFitness() - indList.get(i).getFitness()) <= e){
                timeSerie[i] = 0;
            }
            if((indList.get(i+1).getFitness() - indList.get(i).getFitness()) > e){
                timeSerie[i] = 1;
            }
        }
        
        return timeSerie;
        
    }
    
    /**
     * 
     * @param indList
     * @return 
     */
    private double countStabilityMeasure(List<Individual> indList){
        
        double e = Double.MIN_VALUE;
        double dif;
        
        for(int i=0; i<indList.size()-1; i++){
            
            dif = Math.abs(indList.get(i).getFitness() - indList.get(i+1).getFitness());
            
            if(dif > e){
                e = dif;
            }
            
        }
        
        return e;
        
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        FitnessFunction ffunction = new Rosenbrock();
        int dimension = 2;
        int walkSteps = 10000;
        double stepBoundary = 0.1;
        int walkCount = 10;
        
        RuggednessEstimation re = new RuggednessEstimation(ffunction, dimension, walkSteps, stepBoundary);
        System.out.println(re.estimateRuggedness(walkCount));
        
    }
    
}
