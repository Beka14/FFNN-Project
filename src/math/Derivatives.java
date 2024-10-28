package math;

public class Derivatives {
    public static float D_ReLU(float input){
        return (input <= 0) ? (float) 0.1 : 1;
    }

}
