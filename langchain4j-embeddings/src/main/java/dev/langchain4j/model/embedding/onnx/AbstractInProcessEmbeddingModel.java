package dev.langchain4j.model.embedding.onnx;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.DimensionAwareEmbeddingModel;
import dev.langchain4j.model.embedding.TokenCountEstimator;
import dev.langchain4j.model.embedding.onnx.OnnxBertBiEncoder.EmbeddingAndTokenCount;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.model.output.TokenUsage;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;

import static java.nio.file.Files.newInputStream;
import static java.util.concurrent.CompletableFuture.supplyAsync;
import static java.util.stream.Collectors.toList;

public abstract class AbstractInProcessEmbeddingModel extends DimensionAwareEmbeddingModel implements TokenCountEstimator {

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

    protected abstract Executor executor();

    @Override
    public Response<List<Embedding>> embedAll(List<TextSegment> segments) {

        Executor executor = executor();

        List<CompletableFuture<EmbeddingAndTokenCount>> futures = segments.stream()
                .map(segment -> supplyAsync(() -> model().embed(segment.text()), executor))
                .collect(toList());

        int inputTokenCount = 0;
        List<Embedding> embeddings = new ArrayList<>();

        for (CompletableFuture<EmbeddingAndTokenCount> future : futures) {
            try {
                EmbeddingAndTokenCount embeddingAndTokenCount = future.get();
                embeddings.add(Embedding.from(embeddingAndTokenCount.embedding));
                inputTokenCount += embeddingAndTokenCount.tokenCount - 2; // do not count special tokens [CLS] and [SEP]
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }

        return Response.from(embeddings, new TokenUsage(inputTokenCount));
    }

    @Override
    public int estimateTokenCount(String text) {
        return model().countTokens(text);
    }
}
