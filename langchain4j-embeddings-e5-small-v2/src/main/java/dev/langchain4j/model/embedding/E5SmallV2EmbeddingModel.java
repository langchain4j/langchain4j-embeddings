package dev.langchain4j.model.embedding;

public class E5SmallV2EmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "e5-small-v2.onnx",
            "bert-vocabulary-en.txt",
            PoolingMode.MEAN
    );

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }
}
