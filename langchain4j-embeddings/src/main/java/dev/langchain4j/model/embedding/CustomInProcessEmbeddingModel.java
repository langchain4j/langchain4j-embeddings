package dev.langchain4j.model.embedding;

import java.nio.file.Path;

public class CustomInProcessEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private final OnnxBertBiEncoder model;

    public CustomInProcessEmbeddingModel(Path pathToModel) {
        model = loadFromFileSystem(pathToModel);
    }

    @Override
    protected OnnxBertBiEncoder model() {
        return model;
    }
}
