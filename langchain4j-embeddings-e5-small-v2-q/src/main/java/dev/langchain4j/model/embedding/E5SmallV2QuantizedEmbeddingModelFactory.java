package dev.langchain4j.model.embedding;

import dev.langchain4j.spi.model.embedding.EmbeddingModelFactory;

public class E5SmallV2QuantizedEmbeddingModelFactory implements EmbeddingModelFactory {

    @Override
    public EmbeddingModel create() {
        return new E5SmallV2QuantizedEmbeddingModel();
    }
}
