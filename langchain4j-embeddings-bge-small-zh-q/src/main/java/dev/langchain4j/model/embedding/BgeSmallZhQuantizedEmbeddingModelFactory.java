package dev.langchain4j.model.embedding;

import dev.langchain4j.spi.model.embedding.EmbeddingModelFactory;

public class BgeSmallZhQuantizedEmbeddingModelFactory implements EmbeddingModelFactory {

    @Override
    public EmbeddingModel create() {
        return new BgeSmallZhQuantizedEmbeddingModel();
    }
}
