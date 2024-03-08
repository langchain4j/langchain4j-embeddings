package dev.langchain4j.model.embedding;

/**
 * BAAI bge-small-zh embedding model that runs within your Java application's process.
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Embedding dimensions: 512
 * <p>
 * It is recommended to add "为这个句子生成表示以用于检索相关文章：" prefix to a query.
 * <p>
 * More details <a href="https://huggingface.co/BAAI/bge-small-zh">here</a>
 */
public class BgeSmallZhEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "bge-small-zh.onnx",
            "tokenizer.json",
            PoolingMode.CLS
    );

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }
}
