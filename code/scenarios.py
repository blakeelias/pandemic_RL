GDP_US = 2e13

class US:
    population = 400000000 # 400 million people -- i.e. approximate US population
    # imported_cases_per_step = 0.1
    # hospital_capacity = 1000000 # 1 million hospital beds -- https://www.aha.org/statistics/fast-facts-us-hospitals
    gdp_per_day = GDP_US / 365.0
    fraction_gdp_lost = 0.35
    death_rate = 0.01
    cost_per_life_year = 100000
    cost_per_death = 1e7 # $10,000,000 per death
    hospitalization_rate = 0.1
    cost_per_hospitalization = 50000 # $50k per hospitalization -- average amount billed to insurance (can dig up this reference if needed; it was on this order of magnitude)
    cost_per_case = death_rate * cost_per_death + hospitalization_rate * cost_per_hospitalization

    age_groups = [
        '<5',
        '5-17',
        '18-34',
        '35-49',
        '50-59',
        '60-64',
        '65-69',
        '70-74',
        '75-79',
        '80+',
    ]
    
    population_per_age_group = [
        0.058,
        0.167,
        0.243,
        0.193,
        0.125,
        0.059,
        0.05,
        0.041,
        0.027,
        0.039,
    ]

    life_years_lost_per_age_group = [
        0.001008714655,
        0.001643278669,
        0.02613648652,
        0.0955705416,
        0.2104680841,
        0.3317779913,
        0.5996554058,
        0.6905803497,
        0.6421921692,
        0.342975522,
    ]
    
class Test:
    population = 10000
    gdp_per_day = population / US.population * US.gdp_per_day
    fraction_gdp_lost = 0.35
    death_rate = 0.01
    cost_per_death = 1e7 # $10,000,000 per death
    hospitalization_rate = 0.1
    cost_per_hospitalization = 50000 # $50k per hospitalization -- average amount billed to insurance (can dig up this reference if needed; it was on this order of magnitude)
    cost_per_case = death_rate * cost_per_death + hospitalization_rate * cost_per_hospitalization


class Test2:
    population = 10000
    gdp_per_day = 1
    fraction_gdp_lost = 1
    death_rate = 0.0
    cost_per_death = 0
    hospitalization_rate = 1
    cost_per_hospitalization = 1
    cost_per_case = 1
