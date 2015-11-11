package cz.wikkiw.fla.peakness;

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
import cz.wikkiw.fla.gradient.GradientEstimation;
import cz.wikkiw.prwm.EdgeWalk;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;

/**
 *
 * @author adam on 11/11/2015
 */
public class EdgePeakness {

    private FitnessFunction ffunction;
    private int dimension;
    private int walkSteps;
    private Individual best;
    
    private Boundary boundary;

    public EdgePeakness() {
    }

    public EdgePeakness(FitnessFunction ffunction, int dimension, int walkSteps) {
        this.ffunction = ffunction;
        this.dimension = dimension;
        this.walkSteps = walkSteps;
        
        this.boundary = this.ffunction.getBoundary();
        this.best = new Individual(new double[dimension], Double.MAX_VALUE);
    }

    /**
     * 
     * @return 
     */
    private double estimateSinglePeakness(){
        

        Random rnd = new Random();

        EdgeWalk edge = new EdgeWalk(this.dimension, this.walkSteps,this.ffunction);
        edge.walk();
        
        List<Individual> walk = edge.getWalkIndividuals();
        Individual bestInd = edge.getBest();
        
        if(this.best == null){
            this.best = walk.get(0);
        }
        
        if(bestInd.getFitness() < this.best.getFitness()){
            this.best = bestInd;
        }
        
        double peaks = 0;
        int direction = -1;
        double prev_fitness = walk.get(0).getFitness();
        
        for(Individual ind : walk){
            
            if(ind.getFitness() < prev_fitness && direction == 1){
                peaks += 1;
                direction = 0;
            }
            else if(ind.getFitness() > prev_fitness && direction == 0){
                peaks += 1;
                direction = 1;
            }
            else if(ind.getFitness() < prev_fitness && direction == -1){
                direction = 0;
            }
            else if(ind.getFitness() > prev_fitness && direction == -1){
                direction = 1;
            }
            
        }
        
        return peaks; 
    }
    
    /**
     * 
     * @param runCount
     * @return 
     */
    public double[] estimatePeakness(int runCount){

        double[] ret = new double[runCount];
        
        for(int i=0; i<runCount; i++){
            ret[i] = this.estimateSinglePeakness();
        }
        
        return ret;
        
    }
    
    /**
     * 
     * @param ff
     * @param walkSteps
     * @param walkCount
     * @param dimension
     * @return 
     */
    public double[] printOutPeakness(FitnessFunction ff, int walkSteps, int walkCount, int dimension){

        this.ffunction = ff;
        this.walkSteps = walkSteps;
        this.boundary = this.ffunction.getBoundary();
        
        this.dimension = dimension;
        this.ffunction.init(this.dimension);
        double[] peaknessTable = this.estimatePeakness(walkCount);
        
        double avgPeakness = DoubleStream.of(peaknessTable).average().getAsDouble();
        double stdPeakness;
        double sum = 0;
        
        for(int i=0; i< peaknessTable.length; i++){
            sum += Math.pow(peaknessTable[i] - avgPeakness, 2);
        }
        
        stdPeakness = Math.sqrt((1/(double)peaknessTable.length)*sum);

        PrintWriter writer;
        try {
            writer = new PrintWriter(this.ffunction.getName()+"-peakness.txt", "UTF-8");

            writer.println("{" + avgPeakness + ", " + stdPeakness + "}");
            
            System.out.println(this.ffunction.getName() + " Peakness: " + Arrays.toString(peaknessTable));
            System.out.println("Average: " + avgPeakness + ", STD: " + stdPeakness);
        
            writer.close();
        
        } catch (FileNotFoundException | UnsupportedEncodingException ex) {
            Logger.getLogger(GradientEstimation.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        return peaknessTable;
        
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
        EdgePeakness re;
        int dimension = 10;
        int walkSteps = 100 * dimension;
        int walkCount = 10;
        
        ffunction = new Ackley();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new Griewank();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new Rosenbrock();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new Schwefel();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new Salomon();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new Quadric();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f1();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f2();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f3();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f4();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f5();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f6();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f7();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f8();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f9();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f10();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f11();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f12();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f13();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f14();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
        ffunction = new f15();
        
        re = new EdgePeakness();
        re.printOutPeakness(ffunction, walkSteps, walkCount, dimension);
        
    }
    
}
