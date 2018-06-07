package org.deeplearning4j.transferlearning.vgg16;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.*;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.distribution.NormalDistribution;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.exceptions.InvalidKerasConfigurationException;
import org.deeplearning4j.nn.modelimport.keras.exceptions.UnsupportedKerasConfigurationException;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import static java.lang.Math.toIntExact;

/**
 * @author susaneraly on 3/9/17.
 *
 * We use the transfer learning API to construct a new model based of org.deeplearning4j.transferlearning.vgg16
 * We will hold all layers but the very last one frozen and change the number of outputs in the last layer to
 * match our classification task.
 * In other words we go from where fc2 and predictions are vertex names in org.deeplearning4j.transferlearning.vgg16
 *  fc2 -> predictions (1000 classes)
 *  to
 *  fc2 -> predictions (5 classes)
 * The class "FitFromFeaturized" attempts to train this same architecture the difference being the outputs from the last
 * frozen layer is presaved and the fit is carried out on this featurized dataset.
 * When running multiple epochs this can save on computation time.
 */
public class EditLastLayerOthersFrozen {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(EditLastLayerOthersFrozen.class);

    //protected static final int numClasses = 5;
    //protected static final int numClasses = 96;
    //protected static final int numClasses = 55;
    //protected static final int numClasses = 140;
    private static final Random rng  = new Random(13);
    protected static final int numClasses = 10;
    //protected static final int numClasses = 4;
    protected static final long seed = 12345;

    private static final int trainPerc = 80;
    //private static final int batchSize = 15;
    //private static final int batchSize = 8;
    private static final int batchSize = 4;
    private static final int nEpochs = 20;
    private static final String featureExtractionLayer = "fc2";

    public static void main(String [] args) throws UnsupportedKerasConfigurationException, IOException, InvalidKerasConfigurationException {

        //Import vgg
        //Note that the model imported does not have an output layer (check printed summary)
        //  nor any training related configs (model from keras was imported with only weights and json)
        log.info("\n\nLoading org.deeplearning4j.transferlearning.vgg16...\n\n");
        //ZooModel zooModel = VGG16.builder().build();
        ZooModel zooModel = new VGG16();
        //Model net = zooModel.initPretrained(PretrainedType.IMAGENET);
        //ComputationGraph vgg16 = (ComputationGraph) net;
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        log.info(vgg16.summary());

        //Decide on a fine tune configuration to use.
        //In cases where there already exists a setting the fine tune setting will
        //  override the setting for all layers that are not "frozen".
        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                //.updater(new Nesterovs(5e-5))
                //.updater(new Nesterovs(3e-5, 0.9))
                .updater(new Nesterovs(1e-5))
                .seed(seed)
                .build();

        //Construct a new model with the intended architecture and print summary
        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor(featureExtractionLayer) //the specified layer and below are "frozen"
                .removeVertexKeepConnections("predictions") //replace the functionality of the final vertex
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(numClasses)
                                .weightInit(WeightInit.DISTRIBUTION)
                                .dist(new NormalDistribution(0,0.2*(2.0/(4096+numClasses)))) //This weight init dist gave better results than Xavier
                                .activation(Activation.SOFTMAX).build(),
                        "fc2")
                .build();
        log.info(vgg16Transfer.summary());

        //Dataset iterators
        /*
        SymbolsDataSetIterator.setup(batchSize,trainPerc);
        DataSetIterator trainIter = SymbolsDataSetIterator.trainIterator();
        DataSetIterator testIter = SymbolsDataSetIterator.testIterator();
        */
        /*
        String DATA_DIR = new File(System.getProperty("user.home")) + "/dl4jDataDir";
        File rootFile = new File(DATA_DIR);
        if (!rootFile.exists()) {
            rootFile.mkdir();
        }
        File tarFile = new File(DATA_DIR, "doc108308742_467339969.zip");
        if (!tarFile.isFile()) {
            log.info("Downloading the symbol dataset from https://vk.com/doc108308742_467339969...");
            FileUtils.copyURLToFile(
                    new URL("https://vk.com/doc108308742_467339969"),
                    tarFile);
        }
        ArchiveUtils.unzipFileTo(tarFile.getAbsolutePath(), rootFile.getAbsolutePath());
        log.info("Data set download completed");

        String FLOWER_DIR = DATA_DIR + "/symbol_photos";
        */
        //File mainPath = new File("C:\\Users\\Piastres\\IdeaProjects\\dl4jcorebenchmark\\src\\main\\resources\\animals");
        File mainPath = new File("C:\\Users\\Piastres\\dl4jDataDir\\symbol_photo");
        FileSplit filesInDir = new FileSplit(mainPath, NativeImageLoader.ALLOWED_FORMATS, rng);
        //File parentDir = new File(FLOWER_DIR);
        //String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        //FileSplit filesInDir = new FileSplit(parentDir, allowedExtensions, rng);
        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        //BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        //int numExamples = toIntExact(filesInDir.length());
        //int numLabels = filesInDir.getRootDir().listFiles(File::isDirectory).length;
        String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        //BalancedPathFilter pathFilter = new BalancedPathFilter(rng, labelMaker, numExamples, numLabels, batchSize);

        log.info("numExamples: " + toIntExact(filesInDir.length()));
        log.info("numClasses: " + filesInDir.getRootDir().listFiles(File::isDirectory).length);
        //InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPerc, 100-trainPerc);
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, 0.8, 0.2);
        InputSplit trainData = filesInDirSplit[0];
        log.info("trainData " + trainData.length());
        InputSplit testData = filesInDirSplit[1];
        log.info("testData " + testData.length());



        ImageTransform flipTransform1 = new FlipImageTransform(rng);
        ImageTransform flipTransform2 = new FlipImageTransform(new Random(123));
        ImageTransform warpTransform = new WarpImageTransform(rng, 42);
        List<ImageTransform> transforms = Arrays.asList(new ImageTransform[]{flipTransform1, warpTransform, flipTransform2});
        int height = 224;
        int width = 224;
        int channels = 3;
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        //log.info("recordReader size " + recordReader.numLabels());
        recordReader.initialize(trainData, warpTransform);
        //recordReader.initialize(trainData);
        //log.info("recordReader size " + recordReader.numLabels());
        /*
        int o = 0;
        for (ImageTransform transform : transforms) {
            o++;
            if (o==1){log.info("START TRANSFORM");}
            else {log.info("ADD TRANSFORM");}
            recordReader.initialize(trainData, transform);
        }
        */

        //recordReader.initialize(trainData, null);
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        //log.info("vuvvl " + trainIter.getLabels().size());
        trainIter.setPreProcessor(new VGG16ImagePreProcessor());

        recordReader.initialize(testData);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
        testIter.setPreProcessor(new VGG16ImagePreProcessor());

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        StatsStorage statsStorage = new InMemoryStatsStorage();
        int listenerFrequency = 1;
        vgg16Transfer.setListeners(new StatsListener(statsStorage, listenerFrequency));

/*
        DataSetIterator dataIter = null;
        for (ImageTransform transform : transforms) {
            recordReader.initialize(trainData, transform);
            dataIter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
            dataIter.setPreProcessor(new VGG16ImagePreProcessor());
            trainIter = new MultipleEpochsIterator(50, dataIter);
            //scaler.fit(dataIter);
            //dataIter.setPreProcessor(scaler);
            //network.fit(trainIter);
        }
        */
        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(vgg16Transfer);
        log.info(transferLearningHelper.unfrozenGraph().summary());

        uiServer.attach(statsStorage);
        for (int epoch = 0; epoch < nEpochs; epoch++) {
            if (epoch == 0) {

                //Evaluation eval = transferLearningHelper.unfrozenGraph().evaluate(testIter);
                Evaluation eval;
                eval = vgg16Transfer.evaluate(testIter);
                log.info("Eval stats BEFORE fit.....");
                log.info(eval.stats() + "\n");
                testIter.reset();
            }
            int iter = 0;
            while (trainIter.hasNext()) {
                //while(dataIter.hasNext()) {
                //transferLearningHelper.fitFeaturized(trainIter.next());
                vgg16Transfer.fit(trainIter.next());
                if (iter % 10 == 0) {
                    log.info("Evaluate model at iter " + iter + " ....");

                    //Evaluation eval = transferLearningHelper.unfrozenGraph().evaluate(testIter);
                    Evaluation eval = vgg16Transfer.evaluate(testIter);
                    log.info(eval.stats());
                    testIter.reset();
                }
                iter++;
            }
            trainIter.reset();
            log.info("Epoch #"+epoch+" complete");

        }

        log.info("Model build complete");

        File locationToSave = new File("MyComputationGraph3.zip");
        //10 classes 0,97 epochs: 20
        //File locationToSave = new File("MyComputationGraph2.zip");
        //10 classes 0,8 epochs: 5
        //File locationToSave = new File("MyComputationGraph1.zip");
        //10 classes 0,6667 epochs: 5
        //File locationToSave = new File("MyComputationGraphNew.zip");
        boolean saveUpdater = false;
        ModelSerializer.writeModel(vgg16Transfer, locationToSave, saveUpdater);

        log.info("Model saved");
    }
}