# Third-party model provenance

Phase 4 does not copy source code from an external repository. B0 uses the scikit-learn `GaussianMixture` implementation under its upstream license. B1 and B2 are original compact PyTorch implementations based on the architecture descriptions in the Phase 4 brief; no repository code was reused.

B3 is an optional wrapper around Hugging Face Transformers `AutoModel` for a frozen WavLM/XLS-R-style reference. Transformers is not a default dependency, the model is disabled by default, and the pretrained model's own license and terms must be checked before use. B3 is a reference upper bound, not a lightweight deployment model.
