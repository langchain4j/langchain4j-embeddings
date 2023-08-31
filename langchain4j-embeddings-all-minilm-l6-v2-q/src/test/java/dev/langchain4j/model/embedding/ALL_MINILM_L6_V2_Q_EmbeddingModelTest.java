package dev.langchain4j.model.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.RelevanceScore;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static dev.langchain4j.internal.Utils.repeat;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Percentage.withPercentage;

class ALL_MINILM_L6_V2_Q_EmbeddingModelTest {

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_embed() {

        EmbeddingModel model = new ALL_MINILM_L6_V2_Q_EmbeddingModel();

        Embedding first = model.embed("hi").get();
        assertThat(first.vector()).hasSize(384);

        Embedding second = model.embed("hello").get();
        assertThat(second.vector()).hasSize(384);

        assertThat(RelevanceScore.cosine(first.vector(), second.vector())).isGreaterThan(0.9);
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void embedding_should_have_similar_values_to_embedding_produced_by_sentence_transformers_python_lib() {

        EmbeddingModel model = new ALL_MINILM_L6_V2_Q_EmbeddingModel();

        Embedding embedding = model.embed("I love sentence transformers.").get();

        assertThat(embedding.vector()[0]).isCloseTo(-0.0803190097f, withPercentage(18));
        assertThat(embedding.vector()[1]).isCloseTo(-0.0171345081f, withPercentage(18));
        assertThat(embedding.vector()[382]).isCloseTo(0.0478825271f, withPercentage(18));
        assertThat(embedding.vector()[383]).isCloseTo(-0.0561899580f, withPercentage(18));
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_embed_510_token_long_text() {

        EmbeddingModel model = new ALL_MINILM_L6_V2_Q_EmbeddingModel();

        String oneToken = "hello ";

        Embedding embedding = model.embed(repeat(oneToken, 510)).get();

        assertThat(embedding.vector()).hasSize(384);
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_fail_to_embed_511_token_long_text() {

        EmbeddingModel model = new ALL_MINILM_L6_V2_Q_EmbeddingModel();

        String oneToken = "hello ";

        assertThatThrownBy(() -> model.embed(repeat(oneToken, 511)))
                .isExactlyInstanceOf(IllegalArgumentException.class)
                .hasMessageStartingWith("Cannot embed text longer than 510 tokens. The following text is 511 tokens long: hello hello");
    }
}