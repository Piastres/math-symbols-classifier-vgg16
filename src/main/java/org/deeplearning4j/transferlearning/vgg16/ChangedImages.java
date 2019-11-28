package org.deeplearning4j.transferlearning.vgg16;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;

public class ChangedImages {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(ChangedImages.class);
    private static int i = 5871;
    private static Path path;
    private static String dir = "ch";

    private static void resizeAndSaveImagesFromSource() throws IOException {
        final String UPLOAD_FOLDER_PATH = "C:\\Users\\Piastres\\Desktop\\srt";
        final String SOURCE_FOLDER_PATH = "C:\\Users\\Piastres\\Desktop\\dst\\";
        File folder = new File(UPLOAD_FOLDER_PATH);
        i = 5871;
        path = Paths.get(SOURCE_FOLDER_PATH + dir + "\\");
        Files.createDirectories(path);

        for (File fileEntry : Objects.requireNonNull(folder.listFiles())) {
            opencv_core.IplImage image = cvLoadImage(UPLOAD_FOLDER_PATH + fileEntry.getName());
            resizeAndSaveImage(image);
            i++;
        }
    }

    private static void resizeAndSaveImage(opencv_core.IplImage sourceImage) {
        final int IMAGE_WIDTH = 224;
        final int IMAGE_HEIGHT = 224;
        final String IMAGE_FORMAT = ".jpg";

        opencv_core.Mat resultNew = new opencv_core.Mat(sourceImage);
        opencv_core.Mat resultEnd = new opencv_core.Mat();
        opencv_imgproc.resize(resultNew, resultEnd, new opencv_core.Size(IMAGE_WIDTH, IMAGE_HEIGHT));
        opencv_imgcodecs.imwrite(path + "\\" + dir + "_" + i + IMAGE_FORMAT, resultEnd);
    }

    public static void main(String[] args) {
        try {
            resizeAndSaveImagesFromSource();
        } catch (IOException e) {
            log.error(e.getMessage());
        }
    }
}
