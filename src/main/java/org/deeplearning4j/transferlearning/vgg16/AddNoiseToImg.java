package sample;

import javafx.embed.swing.SwingFXUtils;
import javafx.scene.image.*;
import javafx.scene.paint.Color;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;


/**
 * Created by Piastres on 07.06.2018.
 */
public class AddNoiseToImg{

    public static void getDirPath(String pathDir) throws IOException {
        File folder = new File(pathDir);
        for (final File fileEntry : folder.listFiles()) {
            System.out.println(pathDir);
            System.out.println(fileEntry.getName());
            String str = pathDir+"\\"+fileEntry.getName();
            System.out.println(str);
            FileInputStream inputstream = new FileInputStream(str);
            Image image = new Image(inputstream);
            Image dst = AddNoiseToImg(image);
            BufferedImage bImage = SwingFXUtils.fromFXImage(dst, null);
            System.out.println(bImage.getHeight());
            try {
                String out = str + "NOISE.jpg";
                ImageIO.write(bImage, "jpg", new File(out));
            } catch (IOException e) {
            }
        }
    }

    public static Image AddNoiseToImg(Image img){
        PixelReader pixelReader = img.getPixelReader();
        int w = (int) img.getWidth();
        int h = (int) img.getHeight();

        WritableImage image= new WritableImage(w,h);
        PixelWriter pixelWriter = image.getPixelWriter();
        for (int i=0; i<h; i++){
            for (int j=0; j<w; j++){
                Color color = pixelReader.getColor(j , i);
                double noise = Math.random() / 1;
                Color changeColor = new Color(Math.min(color.getRed() + noise, 1), Math.min(color.getGreen()+noise, 1),
                        Math.min(color.getBlue() + noise, 1), 1);
                pixelWriter.setColor(j, i, changeColor);
            }
        }
        return image;
    }

    public static void main(String[] args) throws IOException {
        getDirPath("C:\\Users\\Piastres\\dl4jDataDir\\symbol_photo\\4");
    }

}
