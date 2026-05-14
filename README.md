# MicrobiomeEngine

**16S rRNA & Metagenomics Analysis Pipeline**

A pure-Python computational engine for microbiome analysis from 16S amplicon and metagenomic data.

## Features
- OTU clustering simulation (97% identity threshold)
- Alpha diversity: Shannon, Simpson, Chao1, Faith's PD
- Beta diversity: Bray-Curtis dissimilarity, PCoA, PERMANOVA
- Differential abundance testing (DESeq2-style NB model, BH FDR)
- PICRUSt-style functional pathway inference
- Dysbiosis index (F/B ratio + alpha diversity composite)

## Results
- 60 samples (30 healthy, 30 disease), 300 OTUs, 50 pathways
- PERMANOVA: F=1.544, p<0.0001
- Differential OTUs: 16 up, 19 down in disease
- F/B ratio: healthy=2.00, disease=7.26 (dysbiosis p=1.8e-4)

## Usage
```bash
pip install numpy scipy matplotlib
python microbiome_engine.py
```

## Tags
`microbiome` `metagenomics` `16s-rrna` `dysbiosis` `alpha-diversity` `permanova`
