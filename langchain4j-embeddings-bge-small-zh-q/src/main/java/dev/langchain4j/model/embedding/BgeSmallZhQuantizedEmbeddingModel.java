package dev.langchain4j.model.embedding;

/**
 * Model: BAAI bge-small-zh (Chinese language) quantized (smaller and faster, but provides slightly inferior results)
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Dimensions: 512
 * <p>
 * It is recommended to add "为这个句子生成表示以用于检索相关文章：" prefix to a query.
 * <p>
 * More details <a href="https://huggingface.co/BAAI/bge-small-zh">here</a>
 */
public class BgeSmallZhQuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

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
