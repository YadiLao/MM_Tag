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





### Environment
- python 3
- tensorflow 1.0
- treelib
