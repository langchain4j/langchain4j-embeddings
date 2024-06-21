package dev.langchain4j.model.embedding;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

/**
 * Quantized BAAI bge-small-en embedding model that runs within your Java application's process.
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Embedding dimensions: 384
 * <p>
 * It is recommended to add "Represent this sentence for searching relevant passages:" prefix to a query.
 * <p>
 * More details <a href="https://huggingface.co/BAAI/bge-small-en">here</a>
 */
public class BgeSmallEnQuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "bge-small-en-q.onnx",
            "tokenizer.json",
            PoolingMode.CLS
    );

    private final Executor executor;

    /**
     * Creates an instance of an {@code BgeSmallEnQuantizedEmbeddingModel}.
     * Uses a fixed thread pool with the number of threads equal to the number of available processors.
     */
    public BgeSmallEnQuantizedEmbeddingModel() {
        this(Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()));
    }

    /**
     * Creates an instance of an {@code BgeSmallEnQuantizedEmbeddingModel}.
     *
     * @param executor The executor to use to parallelize the embedding process.
     */
    public BgeSmallEnQuantizedEmbeddingModel(Executor executor) {
        this.executor = ensureNotNull(executor, "executor");
    }

    @Override
    protected Executor executor() {
        return executor;
    }

    @Override
    protected OnnxBertBiEncoder model() {
        return MODEL;
    }

    @Override
    protected Integer knownDimension() {
        return 384;
    }
}
