package dev.langchain4j.model.embedding.bge.small.zh.v15;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.bge.small.zh.v15.BgeSmallZhV15QuantizedEmbeddingModel;
import dev.langchain4j.store.embedding.CosineSimilarity;
import dev.langchain4j.store.embedding.RelevanceScore;
import org.junit.jupiter.api.Test;

import static dev.langchain4j.internal.Utils.repeat;
import static dev.langchain4j.model.embedding.internal.VectorUtils.magnitudeOf;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Percentage.withPercentage;

class BgeSmallZhV15QuantizedEmbeddingModelTest {

    @Test
    void should_embed() {

        EmbeddingModel model = new BgeSmallZhV15QuantizedEmbeddingModel();

        Embedding first = model.embed("你好").content();
        assertThat(first.vector()).hasSize(512);

        Embedding second = model.embed("您好").content();
        assertThat(second.vector()).hasSize(512);

        double cosineSimilarity = CosineSimilarity.between(first, second);
        assertThat(RelevanceScore.fromCosineSimilarity(cosineSimilarity)).isGreaterThan(0.94);
    }

    @Test
    void should_embed_510_token_long_text() {

        EmbeddingModel model = new BgeSmallZhV15QuantizedEmbeddingModel();

        String oneToken = "书 ";

        Embedding embedding = model.embed(repeat(oneToken, 510)).content();

        assertThat(embedding.vector()).hasSize(512);
    }

    @Test
    void should_embed_text_longer_than_510_tokens_by_splitting_and_averaging_embeddings_of_splits() {

        EmbeddingModel model = new BgeSmallZhV15QuantizedEmbeddingModel();

        String oneToken = "书 ";

        Embedding embedding510 = model.embed(repeat(oneToken, 510)).content();
        assertThat(embedding510.vector()).hasSize(512);

        Embedding embedding511 = model.embed(repeat(oneToken, 511)).content();
        assertThat(embedding511.vector()).hasSize(512);

        double cosineSimilarity = CosineSimilarity.between(embedding510, embedding511);
        assertThat(RelevanceScore.fromCosineSimilarity(cosineSimilarity)).isGreaterThan(0.99);
    }

    @Test
    void should_produce_normalized_vectors() {

        EmbeddingModel model = new BgeSmallZhV15QuantizedEmbeddingModel();

        String oneToken = "书 ";

        assertThat(magnitudeOf(model.embed(oneToken).content()))
                .isCloseTo(1, withPercentage(0.01));
        assertThat(magnitudeOf(model.embed(repeat(oneToken, 999)).content()))
                .isCloseTo(1, withPercentage(0.01));
    }
}