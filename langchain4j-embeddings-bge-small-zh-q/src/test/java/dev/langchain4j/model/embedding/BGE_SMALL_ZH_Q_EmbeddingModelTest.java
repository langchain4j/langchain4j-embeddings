package dev.langchain4j.model.embedding;

import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.store.embedding.RelevanceScore;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static dev.langchain4j.internal.Utils.repeat;
import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class BGE_SMALL_ZH_Q_EmbeddingModelTest {

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_embed() {

        EmbeddingModel model = new BGE_SMALL_ZH_Q_EmbeddingModel();

        Embedding first = model.embed("你好").get();
        assertThat(first.vector()).hasSize(512);

        Embedding second = model.embed("您好").get();
        assertThat(second.vector()).hasSize(512);

        assertThat(RelevanceScore.cosine(first.vector(), second.vector())).isGreaterThan(0.97);
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_embed_510_token_long_text() {

        EmbeddingModel model = new BGE_SMALL_ZH_Q_EmbeddingModel();

        String oneToken = "书 ";

        Embedding embedding = model.embed(repeat(oneToken, 510)).get();

        assertThat(embedding.vector()).hasSize(512);
    }

    @Test
    @Disabled("Temporary disabling. This test should run only when this or used (e.g. langchain4j-embeddings) module(s) change")
    void should_fail_to_embed_511_token_long_text() {

        EmbeddingModel model = new BGE_SMALL_ZH_Q_EmbeddingModel();

        String oneToken = "书 ";

        assertThatThrownBy(() -> model.embed(repeat(oneToken, 511)))
                .isExactlyInstanceOf(IllegalArgumentException.class)
                .hasMessageStartingWith("Cannot embed text longer than 510 tokens. The following text is 511 tokens long: 书 书");
    }
}