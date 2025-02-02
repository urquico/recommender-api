from faker import Faker
from pathlib import Path
fake = Faker()

for i in range(2, 2101):
    name = fake.name()
    email = fake.email()
    created_at = fake.date_time_this_year()
    updated_at = fake.date_time_this_year()
    country = fake.country()
    city = fake.city()
    gender = fake.random_element(elements=("male", "female"))
    age = fake.random_int(min=18, max=80)
    
    # save to a csv file
    # create a users folder
    Path(f"users").mkdir(parents=True, exist_ok=True)	
    
    with open(f"users/user_{i}.csv", "w") as f:
        f.write("id,name,email,created_at,updated_at,city,country,gender,age\n")
        f.write(f"{i},{name},{email},{created_at},{updated_at},{city},{country},{gender},{age}\n")
    
    