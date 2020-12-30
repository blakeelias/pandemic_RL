# pandemic_RL

## Setup

Install dependencies:

```
pip3 install -r requirements.txt
cd code; pip install -e .
```

## Run

"Hello world":
```
python main.py --imported_cases_per_step_range 0 --num_population 100
```

Full experiment:
```
python main.py --num_population 1000  --imported_cases_per_step_range 1 --powers 1.0 --action_frequency 1 2 4 8 16 --tags no_clipping --horizon 96 --distr_family poisson
```

Debugging:
```
python main.py --num_population 20 --imported_cases_per_step_range 0 --powers 1.0 --action_frequency 1 2 3 4 --tags no_clipping --horizon 50 --distr_family poisson
```