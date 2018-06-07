package org.deeplearning4j.transferlearning.vgg16;


import org.apache.commons.io.FileUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgcodecs;
import org.bytedeco.javacpp.opencv_imgproc;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.bytedeco.javacpp.opencv_highgui.cvWaitKey;
import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
/**
 * Created by Piastres on 02.06.2018.
 */
public class ChangeImgs {
    private static int i;
    private static Path path;
    private static String dir = "ch";
    public static void listFilesForFolder(final File folder) {
        i = 5871;
        path = Paths.get("C:\\Users\\Piastres\\Desktop\\dst\\" + dir + "\\");
        System.out.println(path);
        try {
            Files.createDirectories(path);
        } catch (IOException e) {
            e.printStackTrace();
        }
        for (final File fileEntry : folder.listFiles()) {
            opencv_core.IplImage image = cvLoadImage("C:\\Users\\Piastres\\Desktop\\srt\\"+fileEntry.getName());
            opencv_core.Mat img = new opencv_core.Mat(image);
            //System.out.println(img.channels());
            //System.out.println(image.width() + " "+ image.height());
            imgResize(image);
            fileEntry.delete();
            i++;
        }
    }
    public static void imgResize(opencv_core.IplImage orgImg){
        //opencv_core.IplImage image = opencv_core.IplImage.create(cvGetSize(orgImg), IPL_DEPTH_64F, 3);
        //opencv_imgproc.cvCvtColor(orgImg, image, opencv_imgproc.CV_GRAY2BGR);
        opencv_core.Mat resultNew = new opencv_core.Mat(orgImg);
        opencv_core.Mat resultEnd = new opencv_core.Mat();
        opencv_imgproc.resize(resultNew, resultEnd, new opencv_core.Size(224,224));
        opencv_imgcodecs.imwrite(path +"\\"+ dir +"_"+i+".jpg", resultEnd);
    }

    public static void open(String filename) {
        opencv_core.IplImage image = cvLoadImage(filename);
        if (image != null) {
            cvWaitKey();
        }
    }
    public static void main(String[] args) {
        final File folder = new File("C:\\Users\\Piastres\\Desktop\\srt");
        listFilesForFolder(folder);
    }
}
