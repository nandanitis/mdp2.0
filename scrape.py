import streamlit as st
#Import All the important Dependencies
from bs4 import BeautifulSoup 
import requests
from requests_html import HTMLSession
from itertools import groupby
import pandas as pd

def scrape(input):
   #Getting URL and extracting only review Link from it
    givenURL = input
    extractURL = givenURL.split("FLIPKART")[0]+"FLIPKART"
    reviewURL = extractURL.replace("/p/", "/product-reviews/")

    # print(reviewURL)
    basicInf = {
    "totalRatings": None,
    "totalReviews": None,
    "overallRating": None,
    "rating54321": None,
    "totalLoop": None
}


    #Request and SoapScraper
    HEADERS =( {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36','Accept-Language':'en-us,en;q=0.5'})
    def multiplePage(link) :
        while True:
            page = requests.get(link, headers=HEADERS)
            soup = BeautifulSoup(page.content, "html.parser")
            stringCheck = str(soup)
            if page.status_code==200 :
                return soup


    #taking some basic info
    def basicInfo(reviewURL) :
        FirstPageSoap = multiplePage(reviewURL)
        combinedScore = FirstPageSoap.findAll('div', {'class': 'row _2afbiS'})
        print(combinedScore)
        totalRatings = (combinedScore[0].get_text()).split(' ')[0]
        totalReviews = (combinedScore[1].get_text()).split(' ')[0]
        basicInf["totalRatings"] = totalRatings
        basicInf["totalReviews"] = totalReviews
        print('Total Number of Ratings For Product : ', totalRatings)
        print('Total Number of Reviews For Product : ', totalReviews)

        overalRating = FirstPageSoap.find('div', {'class': '_2d4LTz'})
        overalRating = overalRating.get_text()
        basicInf["overallRating"]  = overalRating
        print('OveralRating For Product : ', overalRating )


        ratings54321 = FirstPageSoap.findAll('div', {'class': '_1uJVNT'})
        rating54321 = []
        for i in range(0,len(ratings54321)):
            rating54321.append(ratings54321[i].get_text())
        basicInf["rating54321"] = rating54321
        print('Number of 5 4 3 2 1 star ratings ',rating54321)

        totalLoop = int(totalReviews.replace(',', '')) // 10
        basicInf["totalLoop"] = totalLoop
        print(basicInf)

    basicInfo(reviewURL)

    #Creating Global Scrapper
    df = pd.DataFrame(columns=['Customer Name', 'Review title', 'Ratings', 'Date of Review', 'Reviews'])

    import time
    reviews=[]
  
    for j in range(1,basicInf["totalLoop"]) :
        time.sleep(0.8)

        print(reviewURL+'&page='+str(j))
        soup = multiplePage(reviewURL+'&page='+str(j))
        #soup=BeautifulSoup(response.content)
        

        #Get the Reviewer Name 
        reviewPerson = soup.find_all('p',{'class':'_2sc7ZR _2V5EHH'})
        cust_name = []
        for i in range(0,len(reviewPerson)):
            cust_name.append(reviewPerson[i].get_text())

        #Grab The Title of every Review
        title = soup.find_all('p',class_='_2-N8zT') 
        review_title = []
        for i in range(0,len(title)):
            review_title.append(title[i].get_text())


        #Rating value of each
        rating = soup.find_all('div',class_='_3LWZlK')
        rate = []
        for i in range(0,len(rating)):
            rate.append(rating[i].get_text())
        if len(rate) == 11 or j>10 :
            rate.pop(0)   #We got 11 review values , so poping the top 1


        #Get Review Description
        review = soup.find_all("div",class_="t-ZTKy")
        review_content = []
        for i in range(0,len(review)):
            review_content.append(review[i].get_text())
        for i in range(len(review_content)):     #Removing Read More from every Sentence
            review_content[i] = review_content[i].replace("READ MORE", "")


        #Get Reviewed Date
        reviewDate = soup.find_all('p',class_='_2sc7ZR')
        reviewDay = []
        for i in range(0,len(reviewDate)):
            reviewDay.append(reviewDate[i].get_text())
            reviewDay = [reviewDay[i] for i in range(1, len(reviewDay), 2)]
            reviewDay

        print(len(cust_name) , len(review_title), len(rate), len(review_content), len(reviewDay))
        # if not all_lengths_are_10 :
        if len(cust_name) == len(review_title) == len(rate) == len(reviewDay) == len(review_content):
            #Creating New DataFrame
            new_df = pd.DataFrame({
                'Customer Name': cust_name,
                'Review title': review_title,
                'Ratings': rate,
                'Date of Review': reviewDay,
            'Reviews': review_content
            })

        #Contacting to Global DataFrame
        df = pd.concat([df, new_df], ignore_index=True)

       

      
    return df