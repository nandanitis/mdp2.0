import pandas as pd
import numpy as np
import emoji
import re
import datetime
from dateutil.relativedelta import relativedelta



def emoji(df) :
    import re

    # Function to count the occurrences of a specific emoji in a given string
    def count_emoji(text, emoji_pattern):
        emojis = re.findall(emoji_pattern, text)
        return len(set(emojis))

    # Define the emoji patterns
    heart_emoji_pattern = r'❤️'  # Modify as per your heart emoji representation
    thumbs_up_emoji_pattern = r'👍'  # Modify as per your thumbs-up emoji representation
    thumbs_down_emoji_pattern = r'👎'  # Modify as per your thumbs-down emoji representation
    crying_emoji_pattern = r'😢'  # Modify as per your crying emoji representation
    df1 = pd.DataFrame()

    # Apply the functions to the 'Reviews' column
    df1['HeartEmojiCount'] = df['Reviews'].apply(lambda x: count_emoji(x, heart_emoji_pattern))
    df1['ThumbsUpEmojiCount'] = df['Reviews'].apply(lambda x: count_emoji(x, thumbs_up_emoji_pattern))
    df1['ThumbsDownEmojiCount'] = df['Reviews'].apply(lambda x: count_emoji(x, thumbs_down_emoji_pattern))
    df1['CryingEmojiCount'] = df['Reviews'].apply(lambda x: count_emoji(x, crying_emoji_pattern))
    return df1


def clean(df) :
    #Fiill Null values in Customer Names
    df['Customer Name'] = df['Customer Name'].fillna('User')

    # function to convert each value to number of months from current date

    def convert_to_months(date_str) :
        
        if 'day' in date_str:
            return 0
        elif 'month' in date_str:
            num_months = int(date_str.split()[0])
            return num_months
        else:
            date_obj = datetime.datetime.strptime(date_str, '%b, %Y')
            current_date = datetime.datetime.now()
            diff = current_date - date_obj
            months = round(diff.days/30)
            return months


    df["Date of Review"] = df["Date of Review"].apply(convert_to_months)

    # function to remove all the emojis

    from cleantext import clean
    def replace_emoji(sentence):
        return clean(sentence, no_emoji=True)

    df['Reviews'] = df['Reviews'].apply(replace_emoji)

