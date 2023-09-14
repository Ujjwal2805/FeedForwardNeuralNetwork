package org.example;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class DataConverter {
    private final Word2Vec word2Vec;

    public DataConverter(Word2Vec word2Vec) {
        this.word2Vec = word2Vec;
    }

    public INDArray convertQuestionsToVectors(String[] questions) {
        int vectorSize = word2Vec.getLayerSize();
        INDArray questionVectors = Nd4j.zeros(questions.length, vectorSize);

        for (int i = 0; i < questions.length; i++) {
            String[] tokens = questions[i].split("\\s+");
            for (String token : tokens) {
                if (word2Vec.hasWord(token)) {
                    questionVectors.getRow(i).addi(word2Vec.getWordVectorMatrix(token));
                }
            }
        }

        return questionVectors;
    }

    public INDArray convertAnswerLabelsToOneHot(String[] answerLabels, String[] allLabels) {
        int numClasses = 5;
        INDArray oneHotLabels = Nd4j.zeros(answerLabels.length, numClasses);

        for (int i = 0; i < answerLabels.length; i++) {
            String label = answerLabels[i];
            int classIdx = getClassIndex(label, allLabels);
            oneHotLabels.putScalar(i, classIdx, 1.0);
        }

        return oneHotLabels;
    }

    private int getClassIndex(String label, String[] allLabels) {
        for (int i = 0; i < allLabels.length; i++) {
            if (label.equals(allLabels[i])) {
                return i;
            }
        }
        return -1; // Label not found, handle this case accordingly
    }
}
