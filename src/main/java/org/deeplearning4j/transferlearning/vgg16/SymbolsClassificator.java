package org.deeplearning4j.transferlearning.vgg16;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;

/**
 * Created by Piastres on 06.06.2018.
 */
public class SymbolsClassificator {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);

    //load exist model (after train)
    public static Map<String, Float> loadModel(String filePath) throws IOException {
        log.info("Model load started ...");
        //String modelFile = "C:\\Users\\Piastres\\IdeaProjects\\dl4jcorebenchmark\\src\\main\\resources\\MyComputationGraph2.zip";
        File modelFile = new File("C:\\Users\\Piastres\\IdeaProjects\\dl4jcorebenchmark\\src\\main\\resources\\MyComputationGraph2.zip");
        ComputationGraph conf = ModelSerializer.restoreComputationGraph(modelFile);
        log.info(conf.summary());
        conf.init();
        log.info("Model load completed");
        //File folder = new File("C:\\Users\\Piastres\\dl4jDataDir\\symbol_photo\\0");
        File fileImg = new File(filePath);
        //for (final File fileEntry : folder.listFiles()) {
            //INDArray[] output = conf.output(false,loadFile(fileEntry.getPath()));
            INDArray[] output = conf.output(false,loadFile(filePath));
            // Sort output for top 5
            INDArray[] sorted = Nd4j.sortWithIndices(output[0], 1, false);
            // sorted map for results
            HashMap <String, Float>  mapResult = new HashMap<String, Float>();
            log.info(String.valueOf(sorted[0].data()));
            log.info(String.valueOf(sorted[1].data()));
            for (int i=0; i<sorted[0].length(); i++){
                mapResult.put(String.valueOf(sorted[0].getInt(i)),sorted[1].getFloat(i) * 100);
            }
            Map result = mapResult.entrySet().stream()
                    .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue,
                            (oldValue, newValue) -> oldValue, LinkedHashMap::new));
        List keys = new ArrayList();
        keys.addAll(result.keySet());
        for (int i = 5; i<keys.size(); i++){
            result.remove(keys.get(i));
        }
            log.info(mapResult.toString());
            log.info(result.toString());
        //}
        //log.info(output.toString());
        // Get top 5
        // extract label for prediction

        return result;
    }

    //load file
    private static INDArray loadFile(String filePath) throws IOException {
        // Convert file to INDArray
        log.info("Convert file to INDArray...");
        //NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        //INDArray image = loader.asMatrix(new File(filePath));
        log.info("-------------------------------------" + filePath);
        String srt = "C:\\Users\\Piastres\\dl4jDataDir\\symbol_photo\\6\\6_444.jpg";
        //log.info("=========================" + srt);
        //log.info("=========================" + filePath);
        //INDArray image = new NativeImageLoader().asMatrix(new File("C:\\Users\\Piastres\\dl4jDataDir\\symbol_photo\\6\\6_444.jpg"));
        //INDArray image = new NativeImageLoader().asMatrix(new File(srt));
        INDArray image = new NativeImageLoader().asMatrix(new File(filePath));
        //file.delete();
        log.info("Mean subtraction pre-processing...");
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);
        return image;
    }

    /*
    public static void main(String[] args) throws IOException {
        loadModel();
    }
    */
}
