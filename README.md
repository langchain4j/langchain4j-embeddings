# langchain4j-embeddings

Repository for LangChain4j's in-process embedding models.
In case of issues and feature requests, please submit them [here](https://github.com/langchain4j/langchain4j/issues/new/choose).

## Git LFS
This repository is separate from the [main repository](https://github.com/langchain4j/langchain4j) due to the large file sizes of the `.onnx` models.

`.onnx` files are stored in Git LFS.

To be able to run integration tests locally, you will need to have `.onnx` files downloaded from the LFS.

Please [install Git LFS](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) and do `git lfs pull`.
