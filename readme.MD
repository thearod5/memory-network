# Memory Network Implementation
Original model was gathered from [here](https://appliedmachinelearning.blog/2019/05/02/building-end-to-end-memory-network-for-question-answering-system-on-babi-facebook-data-set-python-keras-part-2/)
Seq2Seq model was adapted from [this examples](https://github.com/lukas/ml-class/blob/master/videos/seq2seq/train.py)

# How to Run
The file `test.txt` contains the context and query evaluated by `run.py`.

```
$ env/bin/python3 chatbot/run.py
```

I have been getting about 10s to answer on my machine.

# test.txt
Expected first line to contain context as one or more sentences.
The second line is expected to be a query or question about the context.

