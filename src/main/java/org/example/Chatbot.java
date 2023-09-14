package org.example;



import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.TokenPreProcess;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.Arrays;
import java.util.List;

public class Chatbot {

    public static void main(String[] args)  {
        String dataPath = "E:\\Developement\\SpringBootAPP\\ML\\ChatBot_Model\\src\\main\\resources\\Data - Sheet1.csv"; // Path to your dataset
        String modelPath = "E:\\Developement\\SpringBootAPP\\ML\\ChatBot_Model\\src\\main\\resources\\word2vec_model.txt"; // Path to save Word2Vec model

        // Define tokenizer factory and preprocessor
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        TokenPreProcess preProcessor = new CommonPreprocessor();

        // Initialize SentenceIterator
        SentenceIterator iter = new LabelAwareFileSentenceIterator(new File(dataPath));
//        iter.setPreProcessor((SentencePreProcessor) preProcessor);
       // TokenPreProcess preProcessor = new CommonPreprocessor();

// Preprocess the text before tokenization
        String preprocessedText = preProcessor.preProcess(dataPath);
        iter.setPreProcessor(preprocessedText);

// Tokenize the preprocessed text
        List<String> tokens = tokenizerFactory.create(preprocessedText).getTokens();

        // Build Word2Vec model
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .learningRate(0.025)
                .iterate((SentenceIterator) tokens)
                .tokenizerFactory(tokenizerFactory)
                .build();

        vec.fit();

        // Save Word2Vec model
        WordVectorSerializer.writeWord2VecModel(vec, new File(modelPath));

        // Load Word2Vec model
        Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(new File(modelPath));


        String[] questions = {
                "What is the capital of France?",
                "How does photosynthesis work?",
                "Tell me a joke."
        };

        String[] answerLabels = {
                "Paris",
                "Photosynthesis is the process by which plants convert light energy into chemical energy to produce food.",
                "Why don't scientists trust atoms? Because they make up everything."
        };

        String[] allLabels= {"paris" , "bangaore", "rameshwaram", " good","ujjwal" };

        //Word2Vec word2Vec = WordVectorSerializer.readWord2VecModel(new File("path/to/word2vec_model.txt"));
        DataConverter dataConverter = new DataConverter(word2Vec);
        // Convert questions to vectors using Word2Vec
        INDArray questionVectors = dataConverter.convertQuestionsToVectors(questions);

        // Convert answer labels to one-hot encoded vectors
        INDArray oneHotLabels = dataConverter.convertAnswerLabelsToOneHot(answerLabels, allLabels );




        // Initialize and build the chatbot model
        ChatbotModel chatbotModel = new ChatbotModel(word2Vec);
       // chatbotModel.buildModel(word2Vec.getWordVector(word2Vec.vocab().wordAtIndex(0)).length(), word2Vec.vocab().numWords(), 64, 1);
        chatbotModel.buildModel(word2Vec.getLayerSize(), word2Vec.vocab().numWords(), 64, 1);

        // Prepare your dataset as INDArray (question vectors) and labels (answer labels)


        // Train the chatbot model
        chatbotModel.trainModel(questionVectors, oneHotLabels, 10); // Train for 100 epochs



        System.out.println("User: " + Arrays.toString(questions));
        System.out.println("Chatbot: " + oneHotLabels);








    }
}

