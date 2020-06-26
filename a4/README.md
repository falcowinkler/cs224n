cs224n Assignment 4

In this endeavour we coded a neural machine translation model.

trained in 1 hour 30 on a tesla gpu

Some stats:
```
...
epoch 15, iter 95990, avg. loss 21.32, avg. ppl 2.92 cum. examples 63657, speed 10736.48 words/sec, time elapsed 5783.75 sec
epoch 15, iter 96000, avg. loss 24.10, avg. ppl 3.19 cum. examples 63977, speed 10747.60 words/sec, time elapsed 5784.37 sec
epoch 15, iter 96000, cum. loss 24.66, cum. ppl 3.32 cum. examples 63977
begin validation ...
validation: iter 96000, dev. ppl 7.212149
hit patience 5
hit #5 trial
early stop!
...
Final test score on corpus BLEU: 21.840554341281223
```

--- 
Original readme
# NMT Assignment
Note: Heavily inspired by the https://github.com/pcyin/pytorch_nmt repository
