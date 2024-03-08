package dev.langchain4j.model.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Path;
import java.util.List;

import static java.nio.file.Files.newInputStream;
import static java.util.stream.Collectors.toList;

public abstract class AbstractInProcessEmbeddingModel implements EmbeddingModel, TokenCountEstimator {

    static OnnxBertBiEncoder loadFromJar(String modelFileName, String tokenizerFileName, PoolingMode poolingMode) {
        InputStream model = AbstractInProcessEmbeddingModel.class.getResourceAsStream("/" + modelFileName);
        InputStream tokenizer = AbstractInProcessEmbeddingModel.class.getResourceAsStream("/" + tokenizerFileName);
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

        List<Embedding> embeddings = segments.stream()
                .map(segment -> Embedding.from(model().embed(segment.text())))
                .collect(toList());

        return Response.from(embeddings);
    }

    @Override
    public int estimateTokenCount(String text) {
        return model().countTokens(text);
    }
}
