# latency analysis
1. Make sure to have the whole dataset folder under analysis.
```
cd latency
python latency.py
```
2. Run , it will take 3-5 min to finish generating all 6 figures.

# AMM Swap simulations comparing
to regenerate the simulation used in the paper. It takes around 1 day to finish.
```
cd sim
python examples.py
```


# AMM data analysis
```
cd data
python data.py
``` 
1. downloads swap data from Uniswapv3 and Sushiswap
2. calculates the demand ratio for them
3. plots the ratio for the top Uniswapv3 and Sushiswap pools