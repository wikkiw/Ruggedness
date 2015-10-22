package cz.wikkiw.ruggedness;

import cz.wikkiw.fitnessfunctions.Ackley;
import cz.wikkiw.fitnessfunctions.FitnessFunction;
import cz.wikkiw.fitnessfunctions.Griewank;
import cz.wikkiw.fitnessfunctions.Quadric;
import cz.wikkiw.fitnessfunctions.Rosenbrock;
import cz.wikkiw.fitnessfunctions.Salomon;
import cz.wikkiw.fitnessfunctions.Schwefel;
import cz.wikkiw.fitnessfunctions.objects.Boundary;
import cz.wikkiw.fitnessfunctions.objects.Individual;
import cz.wikkiw.prwm.PRWm;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;

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

    public RuggednessEstimation() {
    }

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
    private double estimateSingleRuggedness(){
        
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
        
//        for(Individual ind : walk){
//            System.out.println(ind);
//        }
        
//        System.out.println("=============================\nBEST: " + bestInd);
        
//        double e = this.countStabilityMeasureV2(walk);
        double e = this.countStabilityMeasure(walk);
        
//        System.out.println("=============================\nE: " + e);
 
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
            
//            System.out.println("======================");
            
            timeSerie = new int[walk.size()-1];
            timeSerie = this.codeToTimeSeries(walk, eTable[i]);
            
//            System.out.println("TIME SERIE:\n" + Arrays.toString(timeSerie));
            
            groupCounts = new int[6];            
            groupCounts = this.getGroupCounts(timeSerie);
            
//            System.out.println("GROUP COUNTS:\n" + Arrays.toString(groupCounts));
            
            entropy = this.countEntropy(groupCounts, timeSerie.length-1);
            entropyTable[i] = entropy;
            if(entropy > maxEntropy){
                maxEntropy = entropy;
            }

        }
        
//        System.out.println("ENTROPY TABLE:\n" + Arrays.toString(entropyTable));
        
        /**
         * Max of entropyTable
         */
        
        
        return maxEntropy; 
    }
    
    /**
     * 
     * @return 
     */
    private double estimateDimRuggedness(){
        
        double sum = 0;
        
        for(int i=0; i<this.dimension; i++){
            
            sum += this.estimateSingleRuggedness();
            
        }
        
        return sum/(double) this.dimension;
        
    }
    
    /**
     * 
     * @param runCount
     * @return 
     */
    public double[] estimateRuggedness(int runCount){
    
        double sum = 0;
        double[] result = new double[runCount];
        
        for(int i=0; i<runCount; i++){
            result[i] = this.estimateDimRuggedness();
        }
        
        return result;
        
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
     * 
     * @param indList
     * @return 
     */
    private double countStabilityMeasureV2(List<Individual> indList){

        double min = Double.MAX_VALUE;
        double max = Double.MIN_VALUE;
        
        for (Individual indList1 : indList) {
            if (indList1.getFitness() < min) {
                min = indList1.getFitness();
            }
            if (indList1.getFitness() > max) {
                max = indList1.getFitness();
            }
        }
        
        return Math.abs(max-min);
        
    }
    
    /**
     * 
     * @param ff
     * @param walkSteps
     * @param stepBoundary
     * @param walkCount
     * @return 
     */
    public double[] printOutRuggedness(FitnessFunction ff, int walkSteps, double stepBoundary, int walkCount){
        
        
        double[] ruggedness = new double[5];
        this.ffunction = ff;
        this.walkSteps = walkSteps;
        this.stepBoundary = stepBoundary;
        this.boundary = this.ffunction.getBoundary();
        double[] ruggednessTable;
        
        this.dimension = 2;
        ruggednessTable = this.estimateRuggedness(walkCount);
        ruggedness[0] = this.mean(ruggednessTable);
        
        this.dimension = 5;
        ruggednessTable = this.estimateRuggedness(walkCount);
        ruggedness[1] = this.mean(ruggednessTable);
        
        this.dimension = 10;
        ruggednessTable = this.estimateRuggedness(walkCount);
        ruggedness[2] = this.mean(ruggednessTable);
        
        this.dimension = 20;
        ruggednessTable = this.estimateRuggedness(walkCount);
        ruggedness[3] = this.mean(ruggednessTable);
        
        this.dimension = 30;
        ruggednessTable = this.estimateRuggedness(walkCount);
        ruggedness[4] = this.mean(ruggednessTable);
        
        PrintWriter writer;
        try {
            writer = new PrintWriter(this.ffunction.getName()+"-ruggedness.txt", "UTF-8");

                System.out.println(this.ffunction.getName() + " Ruggedness: " + Arrays.toString(ruggedness));

                writer.println(Arrays.toString(ruggedness));
        
            writer.close();
        
        } catch (FileNotFoundException | UnsupportedEncodingException ex) {
            Logger.getLogger(RuggednessEstimation.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return ruggedness;
        
    }
    
    /**
     * 
     * @param array
     * @return 
     */
    private double mean(double[] array){
        
        double sum = 0;
        
        for(int i=0; i<array.length; i++){
            sum += array[i];
        }
        
        return sum/(double)array.length;
    }
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        FitnessFunction ffunction;
        RuggednessEstimation re;
        int walkSteps = 1000;
        double stepBoundary = 0.1;
        int walkCount = 30;
        
        ffunction = new Ackley();
        re = new RuggednessEstimation();
        re.printOutRuggedness(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Griewank();
        re = new RuggednessEstimation();
        re.printOutRuggedness(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Quadric();
        re = new RuggednessEstimation();
        re.printOutRuggedness(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Rosenbrock();
        re = new RuggednessEstimation();
        re.printOutRuggedness(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Salomon();
        re = new RuggednessEstimation();
        re.printOutRuggedness(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Schwefel();
        re = new RuggednessEstimation();
        re.printOutRuggedness(ffunction, walkSteps, stepBoundary, walkCount);
        
    }
    
}
