public class ActivationFunctions {

    public static float ReLU(float input){
        return (input <= 0) ? 0 : input;
    }

    public static float LeakyReLU(float input){
        float alfa = 0.01f;
        return (input <= 0) ? input * alfa : input;
    }
}
