package dev.langchain4j.model.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import static java.util.stream.Collectors.toList;

public abstract class AbstractInProcessEmbeddingModel implements EmbeddingModel, TokenCountEstimator {

    static OnnxBertBiEncoder loadFromJar(String modelFileName, String vocabularyFileName, PoolingMode poolingMode) {
        InputStream inputStream = AbstractInProcessEmbeddingModel.class.getResourceAsStream("/" + modelFileName);
        return new OnnxBertBiEncoder(
                inputStream,
                AbstractInProcessEmbeddingModel.class.getResource("/" + vocabularyFileName),
                poolingMode
        );
    }

    static OnnxBertBiEncoder loadFromFileSystem(Path pathToModel) {
        try {
            return new OnnxBertBiEncoder(
                    Files.newInputStream(pathToModel),
                    AbstractInProcessEmbeddingModel.class.getResource("/bert-vocabulary-en.txt"),
                    PoolingMode.MEAN
            );
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    protected abstract OnnxBertBiEncoder model();

    @Override
    public List<Embedding> embedAll(List<TextSegment> segments) {
        return segments.stream()
                .map(segment -> Embedding.from(model().embed(segment.text())))
                .collect(toList());
    }

    @Override
    public int estimateTokenCount(String text) {
        return model().countTokens(text);
    }
}
