```
python .\src\process_temperature_steps.py testes/0-50/1teste0_50.csv testes/0-50/2teste0_50.csv testes/0-50/3teste0_50.csv 

python .\src\process_temperature_steps.py testes\50-100\1teste50_100.csv testes\50-100\2teste50_100.csv testes\50-100\3teste50_100.csv

python .\src\process_temperature_steps.py testes\100-50\1teste100_50.csv testes\100-50\2teste100_50.csv testes\100-50\3teste100_50.csv 


python .\src\test_plant\validate_model.py .\testes\50-100\averaged_step_response.csv 50 100

python .\src\optimize_model.py
```