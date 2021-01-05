GDP_US = 2e13

class US:
    population = 400000000 # 400 million people -- i.e. approximate US population
    # imported_cases_per_step = 0.1
    # hospital_capacity = 1000000 # 1 million hospital beds -- https://www.aha.org/statistics/fast-facts-us-hospitals
    gdp_per_day = GDP_US / 365.0
    fraction_gdp_lost = 0.35
    death_rate = 0.01
    cost_per_death = 1e7 # $10,000,000 per death
    hospitalization_rate = 0.1
    cost_per_hospitalization = 50000 # $50k per hospitalization -- average amount billed to insurance (can dig up this reference if needed; it was on this order of magnitude)
    cost_per_case = death_rate * cost_per_death + hospitalization_rate * cost_per_hospitalization

class Test:
    population = 20
    gdp_per_day = population / US.population * US.gdp_per_day
    fraction_gdp_lost = 0.35
    death_rate = 0.01
    cost_per_death = 1e7 # $10,000,000 per death
    hospitalization_rate = 0.1
    cost_per_hospitalization = 50000 # $50k per hospitalization -- average amount billed to insurance (can dig up this reference if needed; it was on this order of magnitude)
    cost_per_case = death_rate * cost_per_death + hospitalization_rate * cost_per_hospitalization
