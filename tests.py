# api call to http://127.0.0.1:5000/recommend?user_index=5&limit=10&model=pigwo

import requests


for i in range(2, 2101):
    print(f'Fetching {i} / 2100...')
    response_v1 = requests.get(f'http://127.0.0.1:5000/recommend?user_index={i}&limit=10&model=igwo')
    response_v2 = requests.get(f'http://127.0.0.1:5000/recommend?user_index={i}&limit=10&model=pigwo')
    print('Done Fetching')
