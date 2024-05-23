package dev.langchain4j.model.embedding.bge.small.zh.v15;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.spi.model.embedding.EmbeddingModelFactory;

public class BgeSmallZhV15QuantizedEmbeddingModelFactory implements EmbeddingModelFactory {

    @Override
    public EmbeddingModel create() {
        return new BgeSmallZhV15QuantizedEmbeddingModel();
    }
}
