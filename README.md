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

Vaccine scheduling:
```
python main.py --num_population 10000  --imported_cases_per_step_range 0 --powers 1.0 --action_frequency 1 --tags no_clipping --horizon 144 --distr_family poisson --R_0 4.0 --vaccine_start 0.5

```

Full experiment:
```
python main.py --num_population 1000  --imported_cases_per_step_range 1 --powers 1.0 --action_frequency 1 2 4 8 16 --tags no_clipping --horizon 96 --distr_family poisson
```

Debugging:
```
python main.py --num_population 20 --imported_cases_per_step_range 0 --powers 1.0 --action_frequency 1 4 8 24 --tags no_clipping --distr_family poisson --horizon 24 36 48 96
```


Param sweep experiment:
```
python main.py --num_population 10000 --imported_cases_per_step_range 0.2 --power 0.1 0.5 1.0 1.5 2.0 3.0 4.0 5.0 10.0 --action_frequency 1 --horizon 192 --distr_family nbinom --R_0 2.5  --dynamics SIR --no-policy-optimization --policy-comparison --vaccine_schedule sigmoid --cost_of_R_1_over_cost_per_case .01 .1 1.0 2.0 5.0 10.0 15.0 20.0 25.0 30.0 40 50 100 200 300 500 1000 --results_dir ../results_horizon_192_vaccine_sigmoid_param_sweep_SIR_average_trials_100_repeat
```