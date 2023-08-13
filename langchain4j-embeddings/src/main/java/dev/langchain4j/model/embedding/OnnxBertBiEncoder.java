package dev.langchain4j.model.embedding;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static ai.onnxruntime.OnnxTensor.createTensor;
import static dev.langchain4j.internal.Exceptions.illegalArgument;
import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;
import static java.nio.LongBuffer.wrap;

public class OnnxBertBiEncoder {

    private static final String CLS = "[CLS]";
    private static final String SEP = "[SEP]";
    private static final int MAX_SEQUENCE_LENGTH = 510; // 512 - 2 (special tokens [CLS] and [SEP])

    private final OrtEnvironment environment;
    private final OrtSession session;
    private final BertTokenizer tokenizer;
    private final PoolingMode poolingMode;

    public OnnxBertBiEncoder(InputStream modelInputStream, URL vocabularyFile, PoolingMode poolingMode) {
        try {
            this.environment = OrtEnvironment.getEnvironment();
            this.session = environment.createSession(loadModel(modelInputStream));
            this.tokenizer = new BertTokenizer(vocabularyFile);
            this.poolingMode = ensureNotNull(poolingMode, "poolingMode");
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public float[] embed(String text) {
        int tokenCount = tokenizer.estimateTokenCountInText(text);
        if (tokenCount > MAX_SEQUENCE_LENGTH) {
            throw illegalArgument("Cannot embed text longer than %s tokens. " +
                    "The following text is %s tokens long: %s", MAX_SEQUENCE_LENGTH, tokenCount, text);
        }
        try (Result result = encode(text)) {
            return toEmbedding(result);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private Result encode(String text) throws OrtException {

        List<String> stringTokens = new ArrayList<>();
        stringTokens.add(CLS);
        stringTokens.addAll(tokenizer.tokenize(text));
        stringTokens.add(SEP);

        // TODO reusable buffers
        long[] tokens = stringTokens.stream()
                .mapToLong(tokenizer::tokenId)
                .toArray();

        long[] attentionMasks = new long[stringTokens.size()];
        for (int i = 0; i < stringTokens.size(); i++) {
            attentionMasks[i] = 1L;
        }

        long[] tokenTypeIds = new long[stringTokens.size()];
        for (int i = 0; i < stringTokens.size(); i++) {
            tokenTypeIds[i] = 0L;
        }

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

    private float[] toEmbedding(Result result) throws OrtException {
        float[][] vectors = ((float[][][]) result.get(0).getValue())[0];
        return normalize(pool(vectors));
    }

    private float[] pool(float[][] vectors) {
        switch (poolingMode) {
            case CLS:
                return clsPool(vectors);
            case MEAN:
                return meanPool(vectors);
            default:
                throw illegalArgument("Unknown pooling mode: " + poolingMode);
        }
    }

    private static float[] clsPool(float[][] vectors) {
        return vectors[0];
    }

    private static float[] meanPool(float[][] vectors) {

        int numVectors = vectors.length;
        int vectorLength = vectors[0].length;

        float[] averagedVector = new float[vectorLength];

        for (float[] vector : vectors) {
            for (int j = 0; j < vectorLength; j++) {
                averagedVector[j] += vector[j];
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
