package dev.langchain4j.model.embedding;

/**
 * SentenceTransformers all-MiniLM-L6-v2 embedding model that runs within your Java application's process.
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 256 tokens.
 * <p>
 * Embedding dimensions: 384
 * <p>
 * More details
 * <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2">here</a> and
 * <a href="https://www.sbert.net/docs/pretrained_models.html">here</a>
 */
public class AllMiniLmL6V2EmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "all-minilm-l6-v2.onnx",
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
