package dev.langchain4j.model.embedding;

import dev.langchain4j.spi.model.embedding.EmbeddingModelFactory;

public class BgeSmallEnQuantizedEmbeddingModelFactory implements EmbeddingModelFactory {

    @Override
    public EmbeddingModel create() {
        return new BgeSmallEnQuantizedEmbeddingModel();
    }
}
