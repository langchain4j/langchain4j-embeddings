package dev.langchain4j.model.embedding;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * A custom embedding model that runs within your Java application's process.
 * Any BERT-based model (e.g., from HuggingFace) can be used, as long as it is in ONNX format.
 * Information on how to convert models into ONNX format can be found <a href="https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model">here</a>.
 * Many models already converted to ONNX format are available <a href="https://huggingface.co/Xenova">here</a>.
 */
public class CustomEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private final OnnxBertBiEncoder model;

    /**
     * Loads a custom embedding model.
     *
     * @param pathToModel The path to the .onnx model file (e.g., "/home/me/model.onnx").
     */
    public CustomEmbeddingModel(Path pathToModel) {
        model = loadFromFileSystem(pathToModel);
    }

    /**
     * Loads a custom embedding model.
     *
     * @param pathToModel The path to the .onnx model file (e.g., "/home/me/model.onnx").
     */
    public CustomEmbeddingModel(String pathToModel) {
        this(Paths.get(pathToModel));
    }

    @Override
    protected OnnxBertBiEncoder model() {
        return model;
    }
}
