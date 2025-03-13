package dev.langchain4j.model.embedding.onnx;

import ai.djl.huggingface.tokenizers.Encoding;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.model.Tokenizer;

import java.io.InputStream;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

import static java.nio.file.Files.newInputStream;

/**
 * A <a href="https://huggingface.co/">HuggingFace</a> tokenizer.
 * <br>
 * Uses DJL's {@link ai.djl.huggingface.tokenizers.HuggingFaceTokenizer} under the hood.
 * <br>
 * Requires {@code tokenizer.json} to instantiate.
 * An <a href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/main/tokenizer.json">example</a>.
 */
public class HuggingFaceTokenizer implements Tokenizer {

    private final ai.djl.huggingface.tokenizers.HuggingFaceTokenizer tokenizer;

    /**
     * Creates an instance of a {@code HuggingFaceTokenizer} using a built-in {@code tokenizer.json} file.
     */
    public HuggingFaceTokenizer() {

        Map<String, String> options = new HashMap<>();
        options.put("padding", "false");
        options.put("truncation", "false");

        this.tokenizer = createFrom(getClass().getResourceAsStream("/bert-tokenizer.json"), options);
    }

    /**
     * Creates an instance of a {@code HuggingFaceTokenizer} using a provided {@code tokenizer.json} file.
     *
     * @param pathToTokenizer The path to the tokenizer file (e.g., "/path/to/tokenizer.json")
     */
    public HuggingFaceTokenizer(Path pathToTokenizer) {
        this(pathToTokenizer, null);
    }

    /**
     * Creates an instance of a {@code HuggingFaceTokenizer} using a provided {@code tokenizer.json} file
     * and a map of DJL's tokenizer options.
     *
     * @param pathToTokenizer The path to the tokenizer file (e.g., "/path/to/tokenizer.json")
     * @param options         The DJL's tokenizer options
     */
    public HuggingFaceTokenizer(Path pathToTokenizer, Map<String, String> options) {
        try {
            this.tokenizer = createFrom(newInputStream(pathToTokenizer), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Creates an instance of a {@code HuggingFaceTokenizer} using a provided {@code tokenizer.json} file.
     *
     * @param pathToTokenizer The path to the tokenizer file (e.g., "/path/to/tokenizer.json")
     */
    public HuggingFaceTokenizer(String pathToTokenizer) {
        this(pathToTokenizer, null);
    }

    /**
     * Creates an instance of a {@code HuggingFaceTokenizer} using a provided {@code tokenizer.json} file
     * and a map of DJL's tokenizer options.
     *
     * @param pathToTokenizer The path to the tokenizer file (e.g., "/path/to/tokenizer.json")
     * @param options         The DJL's tokenizer options
     */
    public HuggingFaceTokenizer(String pathToTokenizer, Map<String, String> options) {
        try {
            this.tokenizer = createFrom(newInputStream(Paths.get(pathToTokenizer)), options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static ai.djl.huggingface.tokenizers.HuggingFaceTokenizer createFrom(InputStream tokenizer,
                                                                                 Map<String, String> options) {
        try {
            return ai.djl.huggingface.tokenizers.HuggingFaceTokenizer.newInstance(tokenizer, options);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public int estimateTokenCountInText(String text) {
        Encoding encoding = tokenizer.encode(text, false, true);
        return encoding.getTokens().length;
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
}
