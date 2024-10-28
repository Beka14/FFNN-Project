package math;

public class ActivationFunctions {

    public static float ReLU(float input){
        return (input <= 0) ? 0 : input;
    }
}
