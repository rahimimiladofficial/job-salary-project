import glassdor_scraper as gs

import pandas as pd
path = "D:/dev/ds_salary_project/chromedriver.exe"

keyword = "data scientist"
records = 50

df = gs.get_jobs(keyword, records, True, path, 2)

df.to_excel(keyword+str(records)+'.xlsx', index=False)
