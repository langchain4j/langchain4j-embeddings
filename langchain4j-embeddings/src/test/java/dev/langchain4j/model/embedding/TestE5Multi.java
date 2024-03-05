package dev.langchain4j.model.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.store.embedding.CosineSimilarity;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static dev.langchain4j.model.embedding.PoolingMode.MEAN;

class TestE5Multi {

    public static void main(String[] args) {
        Path modelPath = Paths.get("C:\\Users\\ljuba\\Downloads\\intfloat_multilingual-e5-base.onnx");
        Path tokenizerPath = Paths.get("C:\\Users\\ljuba\\Downloads\\tokenizer6.json");
        OnnxEmbeddingModel onnxEmbeddingModel = new OnnxEmbeddingModel(modelPath, tokenizerPath, MEAN);
        Embedding embedding = onnxEmbeddingModel.embed("Hello, how are you doing?").content();
        System.out.println(embedding);

        String storePath = "C:\\Users\\ljuba\\Downloads\\store3.json";

//        InMemoryEmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();
//        embeddingStore.add(embedding);
//        embeddingStore.serializeToFile(storePath);

        InMemoryEmbeddingStore<TextSegment> embeddingStore = InMemoryEmbeddingStore.fromFile(storePath);
        List<EmbeddingMatch<TextSegment>> relevant = embeddingStore.findRelevant(embedding, 2);
        System.out.println(CosineSimilarity.between(embedding, relevant.get(0).embedding()));

    }
}
