import javafx.embed.swing.SwingFXUtils;
import javafx.fxml.FXML;
import javafx.scene.chart.BarChart;
import javafx.scene.chart.CategoryAxis;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.image.Image;
import javafx.scene.layout.Pane;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import javax.imageio.ImageIO;
import javafx.scene.image.ImageView;
import org.deeplearning4j.transferlearning.vgg16.SymbolsClassifier;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Map;

public class Controller {
    @FXML
    private ImageView loadedImg;
    @FXML
    private Pane diagramPane;

    public void onClick() throws IOException {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Загрузить изображение");
        fileChooser.setInitialDirectory(
                new File(System.getProperty("user.home"))
        );
        fileChooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("All Images", "*.*"),
                new FileChooser.ExtensionFilter("JPG", "*.jpg"),
                new FileChooser.ExtensionFilter("JPEG", "*.jpeg"),
                new FileChooser.ExtensionFilter("PNG", "*.png"),
                new FileChooser.ExtensionFilter("TIF", "*.tif"),
                new FileChooser.ExtensionFilter("BMP", "*.bmp")
        );
        File file = fileChooser.showOpenDialog(new Stage());
        if (file != null) {
            createBarChart(SymbolsClassifier.loadModel(file.getPath()));
            BufferedImage bufferedImage = ImageIO.read(file);
            Image image = SwingFXUtils.toFXImage(bufferedImage, null);
            loadedImg.setImage(image);
        }
    }
    private void createBarChart(Map<String, Float> map) {
        CategoryAxis xAxis = new CategoryAxis();
        xAxis.setLabel("Символ");
        NumberAxis yAxis = new NumberAxis(0, 100, 10);
        yAxis.setLabel("Процент");
        BarChart<Number, String> barChart = new BarChart<>(yAxis, xAxis);
        // Series
        XYChart.Series<Number, String> dataSeries = new XYChart.Series<>();
        for (Map.Entry entry : map.entrySet()){
            if ((float) entry.getValue() < 1){
                dataSeries.getData().add(new XYChart.Data<>( (float) 1, entry.getKey().toString() ));
            } else {
                dataSeries.getData().add(new XYChart.Data<>((float) entry.getValue(), entry.getKey().toString()));
            }
        }
        // Add Series to BarChart.
        barChart.getData().add(dataSeries);
        barChart.setTitle("Топ-5 результатов классификации");
        barChart.setLegendVisible(false);
        barChart.setStyle("-fx-font-family: \"Segoe UI\", Helvetica, Arial, sans-serif; -fx-text-fill: #4A4A4A");
        diagramPane.getChildren().addAll(barChart);
    }
}
