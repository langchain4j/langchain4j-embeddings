package dev.langchain4j.model.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.OnnxBertBiEncoder.EmbeddingAndTokenCount;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import static java.nio.file.Files.newInputStream;

public abstract class AbstractInProcessEmbeddingModel implements EmbeddingModel, TokenCountEstimator {

    protected static OnnxBertBiEncoder loadFromJar(String modelFileName, String tokenizerFileName, PoolingMode poolingMode) {
        InputStream model = Thread.currentThread().getContextClassLoader().getResourceAsStream(modelFileName);
        InputStream tokenizer = Thread.currentThread().getContextClassLoader().getResourceAsStream(tokenizerFileName);
        return new OnnxBertBiEncoder(model, tokenizer, poolingMode);
    }

    static OnnxBertBiEncoder loadFromFileSystem(Path pathToModel, Path pathToTokenizer, PoolingMode poolingMode) {
        try {
            return new OnnxBertBiEncoder(newInputStream(pathToModel), newInputStream(pathToTokenizer), poolingMode);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static OnnxBertBiEncoder loadFromFileSystem(Path pathToModel, InputStream tokenizer, PoolingMode poolingMode) {
        try {
            return new OnnxBertBiEncoder(newInputStream(pathToModel), tokenizer, poolingMode);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    protected abstract OnnxBertBiEncoder model();

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {

        int inputTokenCount = 0;

        List<Embedding> embeddings = new ArrayList<>();
        for (TextSegment segment : segments) {
            EmbeddingAndTokenCount embeddingAndTokenCount = model().embed(segment.text());
            embeddings.add(Embedding.from(embeddingAndTokenCount.embedding));
            inputTokenCount += embeddingAndTokenCount.tokenCount - 2; // do not count special tokens [CLS] and [SEP]
        }

        return Response.from(embeddings, new TokenUsage(inputTokenCount));
    }

    @Override
    public int estimateTokenCount(String text) {
        return model().countTokens(text);
    }
}
