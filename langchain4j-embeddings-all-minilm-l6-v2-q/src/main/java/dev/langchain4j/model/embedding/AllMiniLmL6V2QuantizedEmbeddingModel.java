package dev.langchain4j.model.embedding;

public class AllMiniLmL6V2QuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "all-minilm-l6-v2-q.onnx",
            "bert-vocabulary-en.txt",
            PoolingMode.MEAN
    );

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }
}