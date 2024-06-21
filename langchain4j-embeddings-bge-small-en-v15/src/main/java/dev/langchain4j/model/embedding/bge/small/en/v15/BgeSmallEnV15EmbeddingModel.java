package dev.langchain4j.model.embedding.bge.small.en.v15;

import dev.langchain4j.model.embedding.AbstractInProcessEmbeddingModel;
import dev.langchain4j.model.embedding.OnnxBertBiEncoder;
import dev.langchain4j.model.embedding.PoolingMode;

/**
 * BAAI bge-small-en-v1.5 embedding model that runs within your Java application's process.
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
public class BgeSmallEnV15EmbeddingModel extends AbstractInProcessEmbeddingModel {

    private static final OnnxBertBiEncoder MODEL = loadFromJar(
            "bge-small-en-v1.5.onnx",
            "bge-small-en-v1.5-tokenizer.json",
            PoolingMode.CLS
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
