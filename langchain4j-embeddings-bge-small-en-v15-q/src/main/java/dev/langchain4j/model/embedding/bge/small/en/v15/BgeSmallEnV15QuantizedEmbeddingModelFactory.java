package dev.langchain4j.model.embedding.bge.small.en.v15;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.spi.model.embedding.EmbeddingModelFactory;

public class BgeSmallEnV15QuantizedEmbeddingModelFactory implements EmbeddingModelFactory {

    @Override
    public EmbeddingModel create() {
        return new BgeSmallEnV15QuantizedEmbeddingModel();
    }
}
