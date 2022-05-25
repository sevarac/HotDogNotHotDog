package hotdog;

import javax.imageio.ImageIO;
import javax.visrec.ml.classification.ImageClassifier;
import javax.visrec.ml.classification.NeuralNetImageClassifier;
import javax.visrec.ml.model.ModelCreationException;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.file.Paths;
import java.util.Map;

/**
 * Example how to train the hot dog image classifier in Java using VisRec API JSR381
 * See UseTrainedHotDogClassifier for how to use trained hot dog classifier.
 *
 * Scene from TV show Silicon Valley https://www.youtube.com/watch?v=vIci3C4JkL0
 * Data set: https://www.kaggle.com/datasets/dansbecker/hot-dog-not-hot-dog
 *
 * Visual Recognition API JSR381 https://jcp.org/en/jsr/detail?id=381
 * JSR project on GitHub https://github.com/JavaVisRec
 *
 */
public class TrainHotDogClassifier {

	public static void main(String[] args) {
		try {
			ImageClassifier<BufferedImage> classifier =
					NeuralNetImageClassifier.builder()
							.inputClass(BufferedImage.class) // input class for classifier
							.imageWidth(128) // width of the input image
							.imageHeight(128) // height of the input image
							.labelsFile(Paths.get("images/labels.txt"))// list of image labels
							.trainingFile(Paths.get("images/index.txt")) // index of images with corresponding labels
							.networkArchitecture(Paths.get("src/main/resources/hot_dog.json"))// architecture of the convolutional neural network in json
							.maxError(0.03f) // error level to stop the training (maximum acceptable error)
							.maxEpochs(1000) // maximum number of training iterations (epochs)
							.learningRate(0.01f)// amount of error to use for adjusting internal parameters in each training iteration
							.exportModel(Paths.get("hotdog.dnet")) // name of the file to save trained model
							.build();

		} catch (ModelCreationException e) { // if something goes wrong an exception is thrown
			System.out.println("Model creation failed! " + e.getMessage());
		}
	}
}