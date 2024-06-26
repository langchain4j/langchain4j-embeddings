package dev.langchain4j.model.embedding.bge.small.en.v15;

import dev.langchain4j.model.embedding.AbstractInProcessEmbeddingModel;
import dev.langchain4j.model.embedding.OnnxBertBiEncoder;
import dev.langchain4j.model.embedding.PoolingMode;

import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static dev.langchain4j.internal.ValidationUtils.ensureNotNull;

/**
 * Quantized BAAI bge-small-en-v1.5 embedding model that runs within your Java application's process.
 * <p>
 * Maximum length of text (in tokens) that can be embedded at once: unlimited.
 * However, while you can embed very long texts, the quality of the embedding degrades as the text lengthens.
 * It is recommended to embed segments of no more than 512 tokens long.
 * <p>
 * Embedding dimensions: 384
 * <p>
 * It is recommended to add "Represent this sentence for searching relevant passages:" prefix to a query.
 * <p>
 * More details <a href="https://huggingface.co/BAAI/bge-small-en-v1.5">here</a>
 */
public class BgeSmallEnV15QuantizedEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "bge-small-en-v1.5-q.onnx",
            "bge-small-en-v1.5-tokenizer.json",
            PoolingMode.CLS
    );

    private final Executor executor;

    /**
     * Creates an instance of an {@code BgeSmallEnV15QuantizedEmbeddingModel}.
     * Uses a fixed thread pool with the number of threads equal to the number of available processors.
     */
    public BgeSmallEnV15QuantizedEmbeddingModel() {
        this(Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()));
    }

    /**
     * Creates an instance of an {@code BgeSmallEnV15QuantizedEmbeddingModel}.
     *
     * @param executor The executor to use to parallelize the embedding process.
     */
    public BgeSmallEnV15QuantizedEmbeddingModel(Executor executor) {
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
