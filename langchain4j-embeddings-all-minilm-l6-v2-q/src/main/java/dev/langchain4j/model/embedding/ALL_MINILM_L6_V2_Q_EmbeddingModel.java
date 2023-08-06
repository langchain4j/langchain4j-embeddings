package dev.langchain4j.model.embedding;

import dev.langchain4j.data.segment.TextSegment;

import java.util.List;

public class ALL_MINILM_L6_V2_Q_EmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertEmbeddingModel MODEL = load("/all-minilm-l6-v2-q.onnx");

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