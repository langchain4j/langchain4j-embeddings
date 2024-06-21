package dev.langchain4j.model.embedding;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

/**
 * Quantized SentenceTransformers all-MiniLM-L6-v2 embedding model that runs within your Java application's process.
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
public class AllMiniLmL6V2QuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "all-minilm-l6-v2-q.onnx",
            "tokenizer.json",
            PoolingMode.MEAN
    );

    private final Executor executor;

    /**
     * Creates an instance of an {@code AllMiniLmL6V2QuantizedEmbeddingModel}.
     * Uses a fixed thread pool with the number of threads equal to the number of available processors.
     */
    public AllMiniLmL6V2QuantizedEmbeddingModel() {
        this(Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()));
    }

    /**
     * Creates an instance of an {@code AllMiniLmL6V2QuantizedEmbeddingModel}.
     *
     * @param executor The executor to use to parallelize the embedding process.
     */
    public AllMiniLmL6V2QuantizedEmbeddingModel(Executor executor) {
        this.executor = ensureNotNull(executor, "executor");
    }

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }

    @Override
    protected Executor executor() {
        return executor;
    }

    @Override
    protected Integer knownDimension() {
        return 384;
    }
}