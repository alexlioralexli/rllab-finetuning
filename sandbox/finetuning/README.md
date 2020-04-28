
# How to run `finetuning`

Sub-policy Adaptation for Hierarchical Reinforcement Learning as presented at ICLR 2020 by Alexander C. Li\*, Carlos Florensa\*, Ignasi Clavera, and Pieter Abbeel.  

[Check out our website!](https://sites.google.com/view/hippo-rl)

To reproduce the results, you should first have [rllab](https://github.com/rllab/rllab) and Mujoco v1.31 configured. 

Then you can do the following from the root directory of `rllab-finetuning`:
- Train HiPPO with randomized period from scratch via 
```python sandbox/finetuning/runs/pg_test.py --env_name antlowgeargather --algo hippo_random_p -trainsnn -random_init -p 15 -minp 5 -tpi 10 -e 3 -n_itr 2000 -eps 0.1 -bs 100000 -d 0.999 -mb -sbl -msb```
- Create a video of the final policy with
```python scripts/create_policy_video.py PATH_TO_PKL_FILE --name antgather_hipporandp --max_path_length 5000 --fps 50```
- Plot the policy performance with 
```python scripts/plot.py data/local/antlowgeargather-hippo-random-p-mb-msbl-trainablelat-fixedvec-latdim6-period5-15-lr0.003-tpi10-eps0.1-disc0.999-bs100000-h5000 --legend "HiPPO, random p" -t "Ant Gather" -sp data/figures/ant_hippo.png```
