package dev.langchain4j.model.embedding.onnx;

import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.bert.BertFullTokenizer;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.Tokenizer;

import java.net.URL;
import java.util.List;

/**
 * @deprecated Use {@link HuggingFaceTokenizer} instead.
 */
@Deprecated(forRemoval = true)
public class BertTokenizer implements Tokenizer {

    private final BertFullTokenizer tokenizer;

    public BertTokenizer() {
        this.tokenizer = createTokenizerFrom(getClass().getResource("/bert-vocabulary-en.txt"));
    }

    public BertTokenizer(URL vocabularyFile) {
        this.tokenizer = createTokenizerFrom(vocabularyFile);
    }

    private static BertFullTokenizer createTokenizerFrom(URL vocabularyFile) {
        try {
            Vocabulary vocabulary = DefaultVocabulary.builder()
                    .addFromTextFile(vocabularyFile)
                    .build();
            return new BertFullTokenizer(vocabulary, true);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int estimateTokenCountInText(String text) {
        return tokenizer.tokenize(text).size();
    }

    @Override
    public int estimateTokenCountInMessage(ChatMessage message) {
        return estimateTokenCountInText(message.text());
    }

    @Override
    public int estimateTokenCountInMessages(Iterable<ChatMessage> messages) {
        int tokens = 0;
        for (ChatMessage message : messages) {
            tokens += estimateTokenCountInMessage(message);
        }
        return tokens;
    }

    public List<String> tokenize(String text) {
        return tokenizer.tokenize(text);
    }

    public long tokenId(String token) {
        return tokenizer.getVocabulary().getIndex(token);
    }
}
