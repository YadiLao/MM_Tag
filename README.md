### MM_Tag
This is an implementation of MM-Tag which is a MCTS enhanced MDP model. The dataset we used is CoNLL2003 (Name Entity Recognition).


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
python mct_go.py
```

#### Evaluation 
We evaluate mcts, raw policy and value function in the section.


For the final result, the original evaluation script eval.pl  is available. Use the following command.

```
./eval.pl  -d '\t'  < ./your_result.txt
```




#### result





### Environment
- python 3
- tensorflow 1.0
- treelib
