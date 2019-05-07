### MM_Tag
This is an implementation of MM-Tag which is a MCTS enhanced MDP model. 

The dataset we used is [CoNLL2003](https://www.clips.uantwerpen.be/conll2003/ner/)  (Name Entity Recognition).


### how to use ?
#### prepare data
- download data folder
- modify data_path of setting to your data path in process_data.py. The original tagging schema is BIO, you can switch it to both IOB or BIOES. In MM_Tag, IOB is more effective.
- process data by typing the following command and it will generate conll03_setting['tag_format']_100d.p file under conll03 folder.
```
python process_data.py
```

#### train model
```
python mct_go.py --train
```

#### reload model and re-train
since the high time complexity, re-train function is provided.
```
python mct_go.py --restore
```

#### Evaluation 
We evaluate mcts, raw policy and value function in the section.
-  evaluate mcts
```
python mct_go.py --test_mcts
```
-  evaluate raw_policy
```
python mct_go.py --test_raw_policy
```

-  evaluate value
```
python mct_go.py --test_value
```

For the final result, the original evaluation script eval.pl  is available. Use the following command.
The input your_result.txt has four columns with -tab separated. The last two columns are the ground truth and the predicted result.

Then use the following command to do evaluation

```
./eval.pl  -d '\t'  < ./your_result.txt
```

more details can be seen at [link](https://blog.argcv.com/articles/2014.c)




#### result

##### performance comparison for all methods.
Model | Precision | Recall  | F1
---|--- |--- |--- 
BLSTM | 80.14% | 72.81% | 76.29%
BLSTM-CNN | 83.48% | 83.28% | 83.38%
CRF (random) | 82.93% | 84.78% |84.19%
BLSTM-CRF (random) | 83.61% | 84.78% |84.19% 
CRF (Glove) | 85.32% | 84.55% |84.93%
BLSTM-CRF (Glove) | 88.57% | 89.04% | 88.30% 
MM-Tag (random) | 84.19% | 86.28% | 85.22%


##### F1 w.r.t different search time K and raw policy P and value function V


k=100 | k=300 | k=600 | k=1000 | k=1500 | P | V
---|--- |--- |--- |--- |--- |---
85.11% | 85.16% | 85.20% | 85.22% | 85.22% | 85.17% | 83.11%





### Environment
- python 3
- tensorflow 1.0
- treelib
