public class Derivatives {
    public static float D_ReLU(float input){
        return (input <= 0) ? 0f : 1f;
    }

    public static float D_LeakyReLU(float input){
        float alfa = 0.01f;
        return (input <= 0) ? alfa : 1f;
    }

}
