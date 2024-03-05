package dev.langchain4j.model.embedding;

import org.junit.jupiter.api.Test;

import java.util.List;

import static dev.langchain4j.internal.Utils.repeat;
import static org.assertj.core.api.Assertions.assertThat;

class BertTokenizerTest {

    BertTokenizer tokenizer = new BertTokenizer();

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

    @Test
    void should_tokenize() {

        List<String> tokens = tokenizer.tokenize("Hello, how are you doing?");

        assertThat(tokens).containsExactly(
                "hello",
                ",",
                "how",
                "are",
                "you",
                "doing",
                "?"
        );
    }

    @Test
    void should_return_token_id() {

        assertThat(tokenizer.tokenId("[CLS]")).isEqualTo(101);
        assertThat(tokenizer.tokenId("[SEP]")).isEqualTo(102);
        assertThat(tokenizer.tokenId("hello")).isEqualTo(7592);
    }
}