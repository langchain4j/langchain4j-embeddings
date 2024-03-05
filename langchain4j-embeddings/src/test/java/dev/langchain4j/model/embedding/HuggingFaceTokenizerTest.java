package dev.langchain4j.model.embedding;

import org.junit.jupiter.api.Test;

import static dev.langchain4j.internal.Utils.repeat;
import static org.assertj.core.api.Assertions.assertThat;

class HuggingFaceTokenizerTest {

    HuggingFaceTokenizer tokenizer = new HuggingFaceTokenizer();

    @Test
    void should_count_tokens_in_text_shorter_than_512_tokens() {

        String text = "Hello, how are you doing?";

        int tokenCount = tokenizer.estimateTokenCountInText(text);

        assertThat(tokenCount).isEqualTo(7);
    }

    @Test
    void should_count_tokens_in_text_longer_than_512_tokens() {

        String text = repeat("Hello, how are you doing?", 100);

        int tokenCount = tokenizer.estimateTokenCountInText(text);

        assertThat(tokenCount).isEqualTo(700);
    }
}