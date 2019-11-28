package org.deeplearning4j.transferlearning.vgg16;

import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.*;
import javafx.scene.paint.Color;
import org.slf4j.Logger;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Objects;

public class NoisedImages {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(NoisedImages.class);

    private static void addNoiseAndSaveImagesFromSource() throws IOException {
        final String PATH_TO_IMAGES_DIR = "C:\\Users\\Piastres\\dl4jDataDir\\symbol_photo\\4";
        final String IMAGE_FORMAT = "jpg";
        final String UPLOADED_IMAGE_SUFFIX = "_noised.jpg";
        File folder = new File(PATH_TO_IMAGES_DIR);

        for (File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            String pathToImage = PATH_TO_IMAGES_DIR + "\\" + fileEntry.getName();
            FileInputStream inputStream = new FileInputStream(pathToImage);
            Image sourceImage = new Image(inputStream);
            Image noisedImage = addNoiseToImage(sourceImage);
            BufferedImage bufferedImage = SwingFXUtils.fromFXImage(noisedImage, null);
            String noisedImageName = pathToImage + UPLOADED_IMAGE_SUFFIX;
            try {
                ImageIO.write(bufferedImage, IMAGE_FORMAT, new File(noisedImageName));
            } catch (IOException e) {
                log.error(e.getMessage());
            }
        }
    }

    private static Image addNoiseToImage(Image sourceImage){
        PixelReader pixelReader = sourceImage.getPixelReader();
        int width = (int) sourceImage.getWidth();
        int height = (int) sourceImage.getHeight();
        WritableImage noisedImage = new WritableImage(width, height);
        PixelWriter pixelWriter = noisedImage.getPixelWriter();

        for (int i= 0 ; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Color color = pixelReader.getColor(j, i);
                double noise = Math.random();
                Color changedColor = new Color(Math.min(color.getRed() + noise, 1), Math.min(color.getGreen() + noise, 1),
                        Math.min(color.getBlue() + noise, 1), 1);
                pixelWriter.setColor(j, i, changedColor);
            }
        }

        return noisedImage;
    }

    public static void main(String[] args) throws IOException {
        addNoiseAndSaveImagesFromSource();
    }
}
