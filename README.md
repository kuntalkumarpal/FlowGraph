# FlowGraph_SoftwareSecurity

### Environment Create 
```
# Re-create the environment
conda env create --file requirements.yml

# Reactivate the environment
conda activate pytorch 
```

### Data
* The preprocessed Cooking (COR) and Maintenance Manual (MAM) data are available in the ```data/``` directory
* For statistics of the pre-processed data, please refer to our paper
* Each folder has the train.jsonl, test.jsonl, and val.jsonl used for our experiments



### Model



# Paper

[Constructing Flow Graphs from Procedural Cybersecurity Texts](https://aclanthology.org/2021.findings-acl.345.pdf)

## Abstract
Following procedural texts written in natural languages is challenging. We must read the whole text to identify the relevant information or identify the instruction flows to complete a task, which is prone to failures. If such texts are structured, we can readily visualize instruction-flows, reason or infer a particular step, or even build automated systems to help novice agents achieve a goal. However, this structure recovery task is a challenge because of such texts’ diverse nature. This paper proposes to identify relevant information from such texts and generate information flows between sentences. We built a large annotated procedural text dataset (CTFW) in the cybersecurity domain (3154 documents). This dataset contains valuable instructions regarding software vulnerability analysis experiences. We performed extensive experiments on CTFW with our LM-GNN model variants in multiple settings. To show the generalizability of both this task and our method, we also experimented with procedural texts from two other domains (Maintenance Manual and Cooking), which are substantially different from cybersecurity. Our experiments show that Graph Convolution Network with BERT sentence embeddings outperforms BERT in all three domains.

## Reference

```
@inproceedings{pal-etal-2021-constructing,
    title = "Constructing Flow Graphs from Procedural Cybersecurity Texts",
    author = "Pal, Kuntal Kumar  and
      Kashihara, Kazuaki  and
      Banerjee, Pratyay  and
      Mishra, Swaroop  and
      Wang, Ruoyu  and
      Baral, Chitta",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.345",
    doi = "10.18653/v1/2021.findings-acl.345",
    pages = "3945--3957",
}
```
