package dev.langchain4j.model.embedding;

public class AllMiniLmL6V2EmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "all-minilm-l6-v2.onnx",
            "bert-vocabulary-en.txt",
            PoolingMode.MEAN
    );

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }
}
