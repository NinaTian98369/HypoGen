# HypoGen
Code and Data for "the paper [HypoGen: Hyperbole Generation with Commonsense and Counterfactual Knowledge](https://arxiv.org/pdf/2109.05097.pdf)" (findings of EMNLP 2021)

### Data folder

- Commonsense: contains <entity2 HasProperty entity2> triplets that we created using a simile corpus
- Hyperbole: contains the hyperbole dataset needed to train the two classifiers (hypo_so and hypo_red)



### Code folder

- HypoBertClas: code to train bert sentence classification model, i.e., clf1
- Hypo_gen: code to generate hyperboles



### Download Trained Model

- clf1_best_bert_model.pth: bert sentence classification model 
- comet_pretrained_model.pickle: COMeT pretrained model on ConceptNet
- reverse_comet_1e-05_adam_32_20000.pickle: Our reverse comet model

All these are available at https://drive.google.com/drive/folders/1aexFfPMD8mRSaq_pQukD8NSTemxp1A0u?usp=sharing



### Generate Hyperboles

```
python3 code/Hypo_gen/generate_hyperbole.py
```

 You need to change the path accordingly.

