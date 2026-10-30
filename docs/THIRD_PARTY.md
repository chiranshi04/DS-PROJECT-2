# Third-party data, model, and code provenance

## Dataset and paper citations

- **IndicSynth:** Sharma, Divya V., Vijval Ekbote, and Anubha Gupta. “IndicSynth: A Large-Scale Multilingual Synthetic Speech Dataset for Low-Resource Indian Languages.” ACL 2025. [ACL Anthology](https://aclanthology.org/2025.acl-long.1070/). The repository and dataset audit record CC BY-NC 4.0 for IndicSynth. The paper reports approximately 4,000 hours, 989 target speakers, and 12 Indian languages.
- **IndicSUPERB/Kathbath:** Javed, Tahir, Kaushal Santosh Bhogale, Abhigyan Raman, Anoop Kunchukuttan, Pratyush Kumar, and Mitesh M. Khapra. “IndicSUPERB: A Speech Processing Universal Performance Benchmark for Indian languages.” [Official AI4Bharat repository](https://github.com/AI4Bharat/indicSUPERB) and [arXiv paper](https://arxiv.org/abs/2208.11761). The repository describes Kathbath and states that the dataset packaging is CC0 while raw IndicCorp text is not owned by AI4Bharat.
- **Indic-CodecFake:** Girish, Mohd Mujtaba Akhtar, Orchid Chetia Phukan, and Arun Balaji Buduru. “Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages.” Findings of ACL 2026. [ACL Anthology](https://aclanthology.org/2026.findings-acl.2159/). The paper states that IndicSUPERB real speech is used to construct the codec-synthesized benchmark. The dataset-specific audio license was not verified in this project.
- **SEA-Spoof:** Wu, Jinyang et al. “SEA-Spoof: Bridging the Gap in Multilingual Audio Deepfake Detection for South-East Asia.” [arXiv](https://arxiv.org/html/2509.19865v2). SEA-Spoof is a relevant external benchmark; access is gated and the repository audit records its academic-research restrictions.
- **MLAAD:** Müller, Nicolas M. et al. “MLAAD: The Multi-Language Audio Anti-Spoofing Dataset.” [arXiv](https://arxiv.org/abs/2401.09512). MLAAD is cited as related work and is not currently selected in the P5 protocol.

The authoritative access, language, license, and verification notes are maintained in [`docs/DATASETS.md`](DATASETS.md) and [`results/data_audit.json`](../results/data_audit.json). A paper citation does not grant permission to download, redistribute, or commercially use a dataset.

## Software and model provenance

Phase 4 does not copy source code from an external repository. B0 uses scikit-learn’s `GaussianMixture` implementation under scikit-learn’s upstream license. B1 and B2 are original compact PyTorch implementations based on the architecture descriptions in the project brief. The ONNX export uses PyTorch and ONNX Runtime APIs.

B3 is an optional wrapper around Hugging Face Transformers `AutoModel` for a frozen WavLM/XLS-R-style reference. Transformers is not a default dependency, the model is disabled by default, and the pretrained model’s own license and terms must be checked before use. No pretrained B3 weights are committed.
