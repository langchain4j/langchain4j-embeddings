package dev.langchain4j.model.embedding;

/**
 * Model: BAAI bge-small-en
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Dimensions: 384
 * <p>
 * It is recommended to add "Represent this sentence for searching relevant passages:" prefix to a query.
 * <p>
 * More details <a href="https://huggingface.co/BAAI/bge-small-en">here</a>
 */
public class BgeSmallEnEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "bge-small-en.onnx",
            "bert-vocabulary-en.txt",
            PoolingMode.CLS
    );

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }
}
