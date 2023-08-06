package dev.langchain4j.model.embedding;

import dev.langchain4j.data.segment.TextSegment;

import java.util.List;

public class E5_SMALL_V2_EmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertEmbeddingModel MODEL = load("/e5-small-v2.onnx");

    @Override
    protected OnnxBertEmbeddingModel model() {
        return MODEL;
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
