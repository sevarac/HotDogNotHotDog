package hotdog;

import deepnetts.net.ConvolutionalNetwork;
import deepnetts.util.FileIO;

import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ri.ml.classification.ImageClassifierNetwork;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

/**
 * Example how to use the trained hot dog image classifier in Java using VisRec API JSR381
 *
 * Scene from TV show Silicon Valley https://www.youtube.com/watch?v=vIci3C4JkL0
 * Data set: https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog
 *
 * Visual Recognition API JSR381 https://jcp.org/en/jsr/detail?id=381
 * JSR project on GitHub https://github.com/JavaVisRec
 */
public class UseTrainedHotDogClassifier {

    public static void main(String[] args) throws IOException, ClassNotFoundException {

        // load a trained model/neural network
        ConvolutionalNetwork convNet =  FileIO.createFromFile("hotdog.dnet", ConvolutionalNetwork.class);
        // create an image classifier using trained model
        ImageClassifier<BufferedImage> classifier = new ImageClassifierNetwork(convNet);

        // load image to classify
        BufferedImage image = ImageIO.read(new File("src/main/resources/pizza.jpg"));
        // feed image into a classifier to recognize it
        Map<String, Float> results = classifier.classify(image);

        // interpret the classification result / probability
        float hotDogProbability = results.get("hot_dog");
        if (hotDogProbability > 0.5) {
            System.out.println("There is a high probability that this is a hot dog");
        } else {
            System.out.println("Most likely this is not a hot dog");
        }
    }
}
