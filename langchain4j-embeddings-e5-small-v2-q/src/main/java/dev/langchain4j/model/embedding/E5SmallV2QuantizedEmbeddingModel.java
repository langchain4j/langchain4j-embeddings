package dev.langchain4j.model.embedding;

/**
 * Quantized Microsoft E5-small-v2 embedding model that runs within your Java application's process.
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Embedding dimensions: 384
 * <p>
 * It is recommended to use the "query:" prefix for queries and the "passage:" prefix for segments.
 * <p>
 * More details <a href="https://huggingface.co/intfloat/e5-small-v2">here</a>
 */
public class E5SmallV2QuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "e5-small-v2-q.onnx",
            "tokenizer.json",
            PoolingMode.MEAN
    );

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }

    @Override
    protected Integer knownDimension() {
        return 384;
    }
}
