from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pandas as pd


def get_jobs(keyword, num_jobs, verbose, path, slp_time):
    '''Gathers jobs as a dataframe, scraped from Glassdoor'''

    # Initializing the webdriver
    options = webdriver.ChromeOptions()

    # Uncomment the line below if you'd like to scrape without a new Chrome window every time.
    # options.add_argument('headless')

    # Change the path to where chromedriver is in your home folder.
    driver = webdriver.Chrome(
        executable_path=path, options=options)
    driver.set_window_size(1000, 1000)

    url = 'https://www.glassdoor.com/Job/jobs.htm?sc.keyword="' + keyword + '"&locT=C&locId=1147401&locKeyword=San%20Francisco,%20CA&jobType=all&fromAge=-1&minSalary=0&includeNoSalaryJobs=true&radius=100&cityId=-1&minRating=0.0&industryId=-1&sgocId=-1&seniorityType=all&companyId=-1&employerSizes=0&applicationType=0&remoteWorkType=0'
    driver.get(url)
    jobs = []

    while len(jobs) < num_jobs:  # If true, should be still looking for new jobs.
        print(len(jobs))
        # Let the page load. Change this number based on your internet speed.
        # Or, wait until the webpage is loaded, instead of hardcoding it.
        time.sleep(slp_time)

        # Test for the "Sign Up" prompt and get rid of it.
        try:
            driver.find_element(By.XPATH, '//*[@id="MainCol"]/div[1]/ul/li[@data-selected="true"]').click()
            print('data-selected=True')
        except ElementClickInterceptedException:
            pass

        time.sleep(1)

        try:
            driver.find_element(By.XPATH, '//*[@id="JAModal"]/div/div[2]/span').click() # clicking to the X.
            print('clicking to the X')
        except NoSuchElementException:
            pass

        # Going through each job in this page
        job_buttons = driver.find_elements(By.XPATH, ".//article[@id='MainCol']//ul/li[@data-adv-type='GENERAL']") # jl for Job Listing. These are the buttons we're going to click.
        for job_button in job_buttons:
            print("round start *** ")

            print("Progress: {}".format("" + str(len(jobs)) + "/" + str(num_jobs)))
            if len(jobs) >= num_jobs:
                print("Job done ; break!")
                break

            job_button.click()  # You might
            print("job_button.click")
            time.sleep(1)
            collected_successfully = False

            while not collected_successfully:
                print("not collected_successfully")
                try:
                    print("first gather done 1")
                    company_name = driver.find_element(By.XPATH, './/div[@data-test="employerName"]').text
                    location = driver.find_element(By.XPATH, './/div[@data-test="location"]').text
                    job_title = driver.find_element(By.XPATH, './/div[@data-test="jobTitle"]').text
                    job_description = driver.find_element(By.CLASS_NAME, 'jobDescriptionContent').text
                    collected_successfully = True
                    print("first gather done")
                except:
                    print("first gather NOT done 2")
                    time.sleep(5)

            try:
                salary_estimate = driver.find_element(By.XPATH, "//*[@id='JDCol']//div[@class='css-w04er4 e1tk4kwz6']/div[4]/span").text
                print("salary_estimate gather done")
            except NoSuchElementException:
                print("salary_estimate gather NOT done")
                salary_estimate = -1  # You need to set a "not found value. It's important."

            try:
                rating = driver.find_element(By.XPATH, ".//div[@class='d-flex justify-content-between']/div/span").text
                print("rating gather done")
            except NoSuchElementException:
                print("rating gather NOT done")
                rating = -1  # You need to set a "not found value. It's important."

            # Printing for debugging
            if verbose:
                print("Job Title: {}".format(job_title))
                print("Salary Estimate: {}".format(salary_estimate))
                print("Job Description: {}".format(job_description[:500]))
                print("Rating: {}".format(rating))
                print("Company Name: {}".format(company_name))
                print("Location: {}".format(location))

            # Going to the Company tab...
            # clicking on this:
            # <div class="tab" data-tab-type="overview"><span>Company</span></div>

            try:
                driver.find_element(By.XPATH,
                            '//*[@id="JobDescriptionContainer"]/div[2]').click()

                try:
                    size = driver.find_element(By.XPATH,
                                './/div[@id="EmpBasicInfo"]//span[text()="Size"]//following-sibling::*').text
                    print("size gather done")

                except NoSuchElementException:
                    size = -1

                try:
                    founded = driver.find_element(By.XPATH,
                        './/div[@id="EmpBasicInfo"]//span[text()="Founded"]//following-sibling::*').text
                    print("founded gather done")
                except NoSuchElementException:
                    founded = -1

                try:
                    type_of_ownership = driver.find_element(By.XPATH,
                        './/div[@id="EmpBasicInfo"]//span[text()="Type"]//following-sibling::*').text
                    print("type_of_ownership gather done")
                except NoSuchElementException:
                    type_of_ownership = -1

                try:
                    industry = driver.find_element(By.XPATH,
                        './/div[@id="EmpBasicInfo"]//span[text()="Industry"]//following-sibling::*').text
                    print("industry gather done")
                except NoSuchElementException:
                    industry = -1

                try:
                    sector = driver.find_element(By.XPATH,
                        './/div[@id="EmpBasicInfo"]//span[text()="Sector"]//following-sibling::*').text
                    print("sector gather done")
                except NoSuchElementException:
                    sector = -1

                try:
                    revenue = driver.find_element(By.XPATH,
                        './/div[@id="EmpBasicInfo"]//span[text()="Revenue"]//following-sibling::*').text
                    print("revenue gather done")
                except NoSuchElementException:
                    revenue = -1


            except NoSuchElementException:  # Rarely, some job postings do not have the "Company" tab.
                size = -1
                founded = -1
                type_of_ownership = -1
                industry = -1
                sector = -1
                revenue = -1


            if verbose:

                print("Size: {}".format(size))
                print("Founded: {}".format(founded))
                print("Type of Ownership: {}".format(type_of_ownership))
                print("Industry: {}".format(industry))
                print("Sector: {}".format(sector))
                print("Revenue: {}".format(revenue))
                print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

            jobs.append({"Job Title": job_title,
                         "Salary Estimate": salary_estimate,
                         "Job Description": job_description,
                         "Rating": rating,
                         "Company Name": company_name,
                         "Location": location,
                         "Size": size,
                         "Founded": founded,
                         "Type of ownership": type_of_ownership,
                         "Industry": industry,
                         "Sector": sector,
                         "Revenue": revenue,
                         })

            print("first round ended")
            # add job to jobs

        # Clicking on the "next page" button
        try:
            driver.find_element(By.XPATH,'//*[@id="MainCol"]/div[2]/div/div[1]/button[7]/span').click()
        except NoSuchElementException:
            print("Scraping terminated before reaching target number of jobs. Needed {}, got {}.".format(num_jobs,
                                                                                                         len(jobs)))
            break

    return pd.DataFrame(jobs)  # This line converts the dictionary object into a pandas DataFrame.

