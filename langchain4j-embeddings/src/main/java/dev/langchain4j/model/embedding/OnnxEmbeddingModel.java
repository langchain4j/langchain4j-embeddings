package dev.langchain4j.model.embedding;

import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * An embedding model that runs within your Java application's process.
 * Any BERT-based model (e.g., from HuggingFace) can be used, as long as it is in ONNX format.
 * Information on how to convert models into ONNX format can be found <a href="https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model">here</a>.
 * Many models already converted to ONNX format are available <a href="https://huggingface.co/Xenova">here</a>.
 */
public class OnnxEmbeddingModel extends AbstractInProcessEmbeddingModel {

    private final OnnxBertBiEncoder model;

    /**
     * @param pathToModel The path to the .onnx model file (e.g., "/home/me/model.onnx").
     */
    public OnnxEmbeddingModel(Path pathToModel) {
        model = loadFromFileSystem(pathToModel);
    }

    /**
     * @param pathToModel The path to the .onnx model file (e.g., "/home/me/model.onnx").
     */
    public OnnxEmbeddingModel(String pathToModel) {
        this(Paths.get(pathToModel));
    }

    @Override
    protected OnnxBertBiEncoder model() {
        return model;
    }
}
