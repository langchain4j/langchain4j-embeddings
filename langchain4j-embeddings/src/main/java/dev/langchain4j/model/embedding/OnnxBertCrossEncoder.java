package dev.langchain4j.model.embedding;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static ai.onnxruntime.OnnxTensor.createTensor;
import static java.nio.LongBuffer.wrap;

public class OnnxBertCrossEncoder {

    public static void main(String[] args) throws IOException {

        Path path = Paths.get("C:\\dev\\ai\\onnx\\ms-marco-MiniLM-L-6-v2\\model.onnx");
        OnnxBertCrossEncoder crossEncoder = new OnnxBertCrossEncoder(Files.newInputStream(path));

        float score = crossEncoder.encode(
                "How many people live in Berlin?",
                "Berlin had a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers."
        );
        System.out.println(score);
    }

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final BertTokenizer tokenizer;

    public OnnxBertCrossEncoder(InputStream modelInputStream) {
        try {
            this.environment = OrtEnvironment.getEnvironment();
            this.session = environment.createSession(loadModel(modelInputStream));
            this.tokenizer = new BertTokenizer();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public float encode(String first, String second) {
        try (Result result = runModel(first, second)) {
            return ((float[][]) result.get(0).getValue())[0][0];
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private Result runModel(String first, String second) throws OrtException {

        List<String> firstTokens = tokenizer.tokenize(first);
        List<String> secondTokens = tokenizer.tokenize(second);

        List<String> stringTokens = new ArrayList<>();
        stringTokens.add("[CLS]");
        if (firstTokens.size() <= 250) {
            stringTokens.addAll(firstTokens);
        } else {
            stringTokens.addAll(firstTokens.subList(0, 250));
        }
        stringTokens.add("[SEP]");
        if (secondTokens.size() <= 250) {
            stringTokens.addAll(secondTokens);
        } else {
            stringTokens.addAll(secondTokens.subList(0, 250));
        }
        stringTokens.add("[SEP]");

        // TODO reusable buffers
        long[] tokens = stringTokens.stream()
                .mapToLong(tokenizer::tokenId)
                .toArray();

        long[] attentionMasks = new long[stringTokens.size()];
        for (int i = 0; i < stringTokens.size(); i++) {
            attentionMasks[i] = 1L;
        }

        long[] tokenTypeIds = new long[stringTokens.size()];
        int tokenTypeIdx = 0;
        tokenTypeIds[tokenTypeIdx++] = 0L; // CLS
        for (int i = 0; i < Math.min(firstTokens.size(), 250); i++) { // TODO
            tokenTypeIds[tokenTypeIdx++] = 0L; // first
        }
        tokenTypeIds[tokenTypeIdx++] = 0L; // SEP
        for (int i = 0; i < Math.min(secondTokens.size(), 250); i++) { // TODO
            tokenTypeIds[tokenTypeIdx++] = 1L; // second
        }
        tokenTypeIds[tokenTypeIdx] = 1L; // SEP

        long[] shape = {1, tokens.length};

        try (
                OnnxTensor tokensTensor = createTensor(environment, wrap(tokens), shape);
                OnnxTensor attentionMasksTensor = createTensor(environment, wrap(attentionMasks), shape);
                OnnxTensor tokenTypeIdsTensor = createTensor(environment, wrap(tokenTypeIds), shape)
        ) {
            Map<String, OnnxTensor> inputs = new HashMap<>();
            inputs.put("input_ids", tokensTensor);
            inputs.put("token_type_ids", tokenTypeIdsTensor);
            inputs.put("attention_mask", attentionMasksTensor);

            return session.run(inputs);
        }
    }

    private byte[] loadModel(InputStream modelInputStream) {
        try (
                InputStream inputStream = modelInputStream;
                ByteArrayOutputStream buffer = new ByteArrayOutputStream()
        ) {
            int nRead;
            byte[] data = new byte[1024];

            while ((nRead = inputStream.read(data, 0, data.length)) != -1) {
                buffer.write(data, 0, nRead);
            }

            buffer.flush();
            return buffer.toByteArray();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private static float[] toEmbedding(Result result) throws OrtException {
        float[][] vectors = ((float[][][]) result.get(0).getValue())[0];
        return normalize(meanPool(vectors));
    }

    private static float[] meanPool(float[][] vectors) {

        int numVectors = vectors.length;
        int vectorLength = vectors[0].length;

        float[] averagedVector = new float[vectorLength];

        for (int i = 0; i < numVectors; i++) {
            for (int j = 0; j < vectorLength; j++) {
                averagedVector[j] += vectors[i][j];
            }
        }

        for (int j = 0; j < vectorLength; j++) {
            averagedVector[j] /= numVectors;
        }

        return averagedVector;
    }

    private static float[] normalize(float[] vector) {

        float sumSquare = 0;
        for (float v : vector) {
            sumSquare += v * v;
        }
        float norm = (float) Math.sqrt(sumSquare);

        float[] normalizedVector = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            normalizedVector[i] = vector[i] / norm;
        }

        return normalizedVector;
    }

    int countTokens(String text) {
        return tokenizer.tokenize(text).size();
    }
}
