package org.example;

import org.deeplearning4j.models.word2vec.Word2Vec;
//import org.deeplearning4j.text.tokenization.tokenizer.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
//import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;

public class ChatbotModel {
    private MultiLayerNetwork model;
    private final TokenizerFactory tokenizerFactory;
    private final Word2Vec word2Vec;

    public ChatbotModel(Word2Vec word2Vec) {
        this.word2Vec = word2Vec;
        this.tokenizerFactory = new DefaultTokenizerFactory();
        this.tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }

    public void buildModel(int inputSize, int outputSize, int hiddenLayerSize, int numHiddenLayers) {
        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(org.nd4j.linalg.learning.config.Adam.builder().build())
                .l2(1e-4)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(inputSize)
                        .nOut(hiddenLayerSize)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder()
                        .nIn(hiddenLayerSize)
                        .nOut(outputSize)
                        .activation(Activation.SOFTMAX)
                        .lossFunction(org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction.MCXENT)
                        .build())
                .backpropType(BackpropType.Standard)
                .build();

        model = new MultiLayerNetwork(config);
        model.init();
        model.setListeners(new ScoreIterationListener(100)); // Print the score every 100 iterations
    }

    public void trainModel(INDArray questionVectors, INDArray answerLabels, int numEpochs) {
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            model.fit(questionVectors, answerLabels);
        }
    }

    public String chat(String userQuestion) {
        INDArray questionVector = transformTextToVector(userQuestion);
        INDArray predictedAnswer = model.output(questionVector);
        return getMostProbableAnswer(predictedAnswer);
    }

    private INDArray transformTextToVector(String text) {
        String[] tokens = tokenizerFactory.create(text).getTokens().toArray(new String[0]);
        //INDArray vectorSum = Nd4j.zeros(word2Vec.getWordVector(word2Vec.vocab().wordAtIndex(0)).length());
        INDArray vectorSum = Nd4j.zeros(word2Vec.getLayerSize());
        for (String token : tokens) {
            if (word2Vec.hasWord(token)) {
                vectorSum.addi(word2Vec.getWordVectorMatrix(token));
            }
        }

        return vectorSum;
    }

    private String getMostProbableAnswer(INDArray predictedAnswer) {
        int idx = Nd4j.argMax(predictedAnswer, 1).getInt(0);
        return word2Vec.vocab().wordAtIndex(idx);
    }
}
