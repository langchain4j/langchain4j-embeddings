package dev.langchain4j.model.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.RelevanceScore;
import dev.langchain4j.store.embedding.Similarity;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static dev.langchain4j.internal.Utils.repeat;
import static dev.langchain4j.model.embedding.internal.VectorUtils.magnitudeOf;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.data.Percentage.withPercentage;

class BGE_SMALL_ZH_Q_EmbeddingModelTest {

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_embed() {

        EmbeddingModel model = new BGE_SMALL_ZH_Q_EmbeddingModel();

        Embedding first = model.embed("你好");
        assertThat(first.vector()).hasSize(512);

        Embedding second = model.embed("您好");
        assertThat(second.vector()).hasSize(512);

        assertThat(RelevanceScore.cosine(first.vector(), second.vector())).isGreaterThan(0.97);
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_embed_510_token_long_text() {

        EmbeddingModel model = new BGE_SMALL_ZH_Q_EmbeddingModel();

        String oneToken = "书 ";

        Embedding embedding = model.embed(repeat(oneToken, 510));

        assertThat(embedding.vector()).hasSize(512);
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_embed_text_longer_than_510_tokens_by_splitting_and_averaging_embeddings_of_splits() {

        EmbeddingModel model = new BGE_SMALL_ZH_Q_EmbeddingModel();

        String oneToken = "书 ";

        Embedding embedding510 = model.embed(repeat(oneToken, 510));
        assertThat(embedding510.vector()).hasSize(512);

        Embedding embedding511 = model.embed(repeat(oneToken, 511));
        assertThat(embedding511.vector()).hasSize(512);

        assertThat(Similarity.cosine(embedding510.vector(), embedding511.vector())).isGreaterThan(0.99);
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_produce_normalized_vectors() {

        EmbeddingModel model = new BGE_SMALL_ZH_Q_EmbeddingModel();

        String oneToken = "书 ";

        assertThat(magnitudeOf(model.embed(oneToken)))
                .isCloseTo(1, withPercentage(0.01));
        assertThat(magnitudeOf(model.embed(repeat(oneToken, 999))))
                .isCloseTo(1, withPercentage(0.01));
    }
}