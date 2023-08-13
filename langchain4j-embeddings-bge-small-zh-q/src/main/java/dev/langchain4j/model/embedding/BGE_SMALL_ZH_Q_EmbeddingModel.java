package dev.langchain4j.model.embedding;

public class BGE_SMALL_ZH_Q_EmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "bge-small-zh-q.onnx",
            "bge-small-zh-vocabulary.txt",
            PoolingMode.CLS
    );

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }
}
