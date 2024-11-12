package layers;

public abstract class Layer {

    public Layer getNextLayer() {
        return nextLayer;
    }

    public void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public Layer getPrevLayer() {
        return prevLayer;
    }

    public void setPrevLayer(Layer prevLayer) {
        this.prevLayer = prevLayer;
    }

    public Layer nextLayer;
    public Layer prevLayer;

    public abstract float[] backProp(float[] dloss_doutput);
}
