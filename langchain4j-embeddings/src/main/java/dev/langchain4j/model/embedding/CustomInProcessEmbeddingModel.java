package dev.langchain4j.model.embedding;

import dev.langchain4j.data.segment.TextSegment;

import java.nio.file.Path;
import java.util.List;

public class CustomInProcessEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private final OnnxBertEmbeddingModel model;

    public CustomInProcessEmbeddingModel(Path pathToModel) {
        model = load(pathToModel);
    }

    @Override
    protected OnnxBertEmbeddingModel model() {
        return model;
    }

    @Override
    // TODO remove before merging and once 0.19.0 was published
    public int estimateTokenCount(List<TextSegment> textSegments) {
        int tokenCount = 0;
        for (TextSegment textSegment : textSegments) {
            tokenCount += estimateTokenCount(textSegment);
        }
        return tokenCount;
    }
}
