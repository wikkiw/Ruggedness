package cz.wikkiw.fla.gradient;

import cz.wikkiw.fitnessfunctions.Ackley;
import cz.wikkiw.fitnessfunctions.FitnessFunction;
import cz.wikkiw.fitnessfunctions.Griewank;
import cz.wikkiw.fitnessfunctions.Quadric;
import cz.wikkiw.fitnessfunctions.Rosenbrock;
import cz.wikkiw.fitnessfunctions.Salomon;
import cz.wikkiw.fitnessfunctions.Schwefel;
import cz.wikkiw.fitnessfunctions.f1;
import cz.wikkiw.fitnessfunctions.f10;
import cz.wikkiw.fitnessfunctions.f11;
import cz.wikkiw.fitnessfunctions.f12;
import cz.wikkiw.fitnessfunctions.f13;
import cz.wikkiw.fitnessfunctions.f14;
import cz.wikkiw.fitnessfunctions.f15;
import cz.wikkiw.fitnessfunctions.f2;
import cz.wikkiw.fitnessfunctions.f3;
import cz.wikkiw.fitnessfunctions.f4;
import cz.wikkiw.fitnessfunctions.f5;
import cz.wikkiw.fitnessfunctions.f6;
import cz.wikkiw.fitnessfunctions.f7;
import cz.wikkiw.fitnessfunctions.f8;
import cz.wikkiw.fitnessfunctions.f9;
import cz.wikkiw.fitnessfunctions.objects.Boundary;
import cz.wikkiw.fitnessfunctions.objects.Individual;
import cz.wikkiw.prwm.ManhattanPRWm;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;

/**
 *
 * @author adam
 */
public class GradientEstimation {

    private FitnessFunction ffunction;
    private int dimension;
    private int walkSteps;
    private double stepBoundary;
    private Individual best;
    
    private Boundary boundary;

    public GradientEstimation() {
    }

    public GradientEstimation(FitnessFunction ffunction, int dimension, int walkSteps, double stepBoundary) {
        this.ffunction = ffunction;
        this.dimension = dimension;
        this.walkSteps = walkSteps;
        this.stepBoundary = stepBoundary;
        
        this.boundary = this.ffunction.getBoundary();
        this.best = new Individual(new double[dimension], Double.MAX_VALUE);
    }

    /**
     * 
     * @return 
     */
    private double[] estimateSingleGradient(){
        
        /**
         * First determine stability measure
         */
        Random rnd = new Random();
        int[] startZone = new int[this.dimension];
        for(int i=0; i<this.dimension; i++){
            startZone[i] = rnd.nextInt(2);
        }
        double changeProbability = 0.05;
        ManhattanPRWm prwm = new ManhattanPRWm(this.dimension, this.walkSteps, this.stepBoundary, startZone, this.ffunction, changeProbability);
        prwm.walk();
        
        List<Individual> walk = prwm.getWalkIndividuals();
        Individual bestInd = prwm.getBest();
        if(bestInd.getFitness() < this.best.getFitness()){
            this.best = bestInd;
        }
        
        double[] grads = new double[walk.size()-1];
        double fmax = Double.MIN_VALUE;
        double fmin = Double.MAX_VALUE;
        
        for(Individual ind : walk){
            if(ind.getFitness()<fmin){
                fmin = ind.getFitness();
            }
            if(ind.getFitness()>fmax){
                fmax = ind.getFitness();
            }
        }
        
        for(int i = 0; i < walk.size()-1; i++){
            
            grads[i] = (Math.abs(walk.get(i+1).getFitness() - walk.get(i).getFitness())/(fmax-fmin))/(this.stepBoundary/this.ffunction.getBoundary().getRange());
            
        }
        
        double gavg;
        double gdev;
        
        gavg = DoubleStream.of(grads).average().getAsDouble();
        
        double sum = 0;
        for(int i = 0; i < grads.length; i++){
            sum += Math.pow(gavg - grads[i], 2); 
        }
        
        gdev = Math.sqrt(sum/grads.length);
        
        double[] ret = new double[]{gavg,gdev};
        
        return ret; 
    }
    
    /**
     * 
     * @return 
     */
    private double[] estimateDimGradient(){
        
        this.best = new Individual(new double[dimension], Double.MAX_VALUE);
        
        double sumavg = 0;
        double sumdev = 0;
        double[] res;
        
        for(int i=0; i<this.dimension; i++){
            
            res = this.estimateSingleGradient();
            sumavg+=res[0];
            sumdev+=res[1];
            
        }
        
        double[] ret = new double[]{sumavg/(double)this.dimension, sumdev/(double)this.dimension};
        
        return ret;
        
    }
    
    /**
     * 
     * @param runCount
     * @return 
     */
    public List<double[]> estimateGradient(int runCount){
    
        double sum = 0;
        double[] result = new double[runCount];
        List<double[]> ret = new ArrayList<>();
        
        for(int i=0; i<runCount; i++){
            ret.add(this.estimateDimGradient());
        }
        
        return ret;
        
    }
    
    /**
     * 
     * @param ff
     * @param walkSteps
     * @param stepBoundary
     * @param walkCount
     * @return 
     */
    public double[] printOutGradient(FitnessFunction ff, int walkSteps, double stepBoundary, int walkCount){
        
        
        double[] gradient;
        this.ffunction = ff;
        this.walkSteps = walkSteps;
        this.stepBoundary = stepBoundary;
        this.boundary = this.ffunction.getBoundary();
        List<double[]> gradientTable;
        
//        this.dimension = 2;
//        ruggednessTable = this.estimateRuggedness(walkCount);
//        ruggedness[0] = this.mean(ruggednessTable);
//        bestArray.add(this.best);
        
        this.dimension = 10;
        this.ffunction.init(dimension);
        gradientTable = this.estimateGradient(walkCount);
        
        double avgSum = 0;
        double devSum = 0;
        
        for(double[] single : gradientTable){
            
            avgSum += single[0];
            devSum += single[1];
            
        }
        
        gradient = new double[]{avgSum/(double)gradientTable.size(), devSum/(double)gradientTable.size()};
        
        PrintWriter writer;
        try {
            writer = new PrintWriter(this.ffunction.getName()+"-gradient.txt", "UTF-8");

            writer.println("{" + gradient[0] + ", " + gradient[1] + "}");
            
            System.out.println(this.ffunction.getName() + " Gradient: " + Arrays.toString(gradient));
        
            writer.close();
        
        } catch (FileNotFoundException | UnsupportedEncodingException ex) {
            Logger.getLogger(GradientEstimation.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return gradient;
        
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
        GradientEstimation re;
        int walkSteps = 10;
        int dimension = 5;
        double stepBoundary;
        int walkCount = 30;
        
        ffunction = new Ackley();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Griewank();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Rosenbrock();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Schwefel();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Salomon();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new Quadric();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f1();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f2();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f3();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f4();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f5();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f6();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f7();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f8();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f9();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f10();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f11();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f12();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f13();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f14();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
        ffunction = new f15();
        stepBoundary = (ffunction.getBoundary().getRange()*dimension)/1000.0;
        re = new GradientEstimation();
        re.printOutGradient(ffunction, walkSteps, stepBoundary, walkCount);
        
    }
    
}
