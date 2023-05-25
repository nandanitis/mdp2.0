import streamlit as st
import pandas as pd
from flipkartScrape import scrape
from clean import clean, emoji
from sentiment import categorical_variable_summary, vader, last18months, wordCloud, commonPhrases, emojiAnalysis, overalSentiment


st.set_page_config(page_title="EmoTrackr", page_icon="📈")
# Hide the Streamlit footer
hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def render_home_page():
    st.write(
        '<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
    st.title("Welcome to EmoTrackr📌")

    # website = st.selectbox("Select a website", ["Flipkart", "Amazon"])
    url = st.text_input("Enter the URL")

    # # Submit button
    # if st.button("Run Sentiment"):
    #     df = scrape(url)
    #     df = clean(df)

    # Specify the file name or file path relative to the current directory

    
    if 'firstLoop' not in st.session_state:
        st.session_state.firstLoop = False

    df = None
    if  not  st.session_state.firstLoop:
        st.session_state.firstLoop = True
        import os
        current_dir = os.getcwd()
        file_name = "E__reviews.csv"
        file_path = os.path.join(current_dir, file_name)
        dfEmoji = None
        # Read the file
        file = open("Clean.csv", encoding="utf8")
        df = pd.read_csv(file)
        df = vader(df)

    if 'df' not in st.session_state:
        st.session_state.df = df
    if 'count' not in st.session_state:
        st.session_state.count = False



    if not st.session_state.count:
        import time
        st.session_state.count = True
        left_co, cent_co,last_co = st.columns(3)
        image=None
        with cent_co:
            image =st.image("pikka.gif")
        def run_loop():
            progress_bar = st.progress(0)  
            message_placeholder = st.empty()
            for i in range(1, 12):  
                time.sleep(0.5)  
                message_placeholder.empty()
                progress_bar.progress(i / 11)
                message_placeholder.write("")
            message_placeholder.empty()
            progress_bar.empty()
        run_loop()
        image.empty()

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    option = st.radio("Select an option", ("Overview","SentimentScore", "Keywords", "WordCloud"), help="We have wide range of Analysers, Pick the Required one")

    if option == "Overview":
        overalSentiment(st.session_state.df)
        categorical_variable_summary(st.session_state.df, 'Ratings')
    elif option == "SentimentScore":
        last18months(st.session_state.df)
    elif option == "Keywords":
        commonPhrases(st.session_state.df)
    elif option == "WordCloud":
        wordCloud(st.session_state.df)


def render_about_page():
    st.write(
        '<style>div.block-container{padding-top:0.8rem;}</style>', unsafe_allow_html=True)
    st.title("About Us")
    st.write("At SentimentX, we are passionate about sentiment analysis and its potential to unlock valuable insights from text data. Sentiment analysis, also known as opinion mining, is the process of determining the emotional tone or sentiment expressed in a piece of text.")
    st.write("Our website aims to simplify sentiment analysis and make it accessible to everyone. With our advanced natural language processing algorithms, we analyze text data to identify and quantify sentiments such as positive, negative, or neutral. Whether it's customer reviews, social media posts, or any other text-based data, our platform provides valuable insights that can be used to make data-driven decisions and gain a deeper understanding of public opinion.")
    st.write("With SentimentX, you can easily analyze the sentiment of text data, visualize sentiment trends, and extract key insights. Our user-friendly interface and powerful analysis tools empower individuals and businesses to harness the power of sentiment analysis and leverage it for various applications, such as brand monitoring, customer feedback analysis, market research, and more.")
    st.write("Join us on this exciting journey as we uncover the sentiments hidden within the vast ocean of text data and help you derive meaningful value from it.")

    st.title("Meet the Team")

    from PIL import Image, ImageDraw, ImageOps
    import requests
    from io import BytesIO

    
    def make_image_rounded():
        # URL of the image
        image_url = "https://media.sproutsocial.com/uploads/2017/02/10x-featured-social-media-image-size.png"

        # Open the image from the URL
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Create a mask with a rounded rectangle shape
        mask = Image.new("L", image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, image.size[0], image.size[1]), fill=255)

        # Apply the mask to the image
        rounded_image = ImageOps.fit(image, mask.size, centering=(0.5, 0.5))
        rounded_image.putalpha(mask)
        return rounded_image
    
    original = make_image_rounded()
    
    col1, col2, col3 = st.columns(3)

    col1.image(original, use_column_width=True)
    col1.subheader("Nandan hegde")
    col1.write("CEO of EmoTracker")

    col2.image(original, use_column_width=True)
    col2.subheader("Lekha G patel")
    col2.write("Software Designer")

    col3.image(original, use_column_width=True)
    col3.subheader("Kushi Agrawal")
    col3.write("UI/UX Designer")

    
def render_contact_page():
    st.write(
        '<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
    st.title("Contact Us📞")
    st.write("Please fill out the form below to get in touch with us.")

    # Contact form
    name = st.text_input("Name")
    email = st.text_input("Email")
    message = st.text_area("Message", height=200)

    if name and email and message:
        st.success("Thank you for reaching out! We will get back to you soon.")
    else:
        st.warning("Please fill in all the required fields.")


def run_app():
    count = 1
    # Sidebar navigation
    nav_option = st.sidebar.selectbox(
        "Navigation",
        ("Home", "About", "Contact"),

    )

    # Render different pages based on the selected option
    if nav_option == "Home":
        render_home_page()
    elif nav_option == "About":
        render_about_page()
    elif nav_option == "Contact":
        render_contact_page()


# Run the app
if __name__ == '__main__':
    run_app()
