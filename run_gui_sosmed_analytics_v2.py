import joblib
import pandas as pd
from transformers import AutoModel, AutoTokenizer
import torch
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import altair as alt
import re
import streamlit as st
import torch.nn.functional as F
import joblib
from deep_translator import GoogleTranslator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from itertools import chain
from PIL import Image
import numpy as np
import plotly.express as px

from transformers import BertModel
from transformers import (AutoTokenizer)
import torch.nn as nn
from huggingface_hub import hf_hub_download

import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import psutil


#######################
# Page configuration
st.set_page_config(
    page_title="Social Media",
    page_icon="\U0001F310",
    layout="wide",
    initial_sidebar_state="expanded")

def my_custom_theme():
    return {
        'config': {
            'background': 'white',
            'view': {'stroke': 'transparent'},  # No border around the chart
            'title': {'fontSize': 20, 'font': 'Arial', 'color': 'black'},
            'axis': {
                'domainColor': 'gray',
                'gridColor': 'lightgray',
                'labelFontSize': 12,
                'titleFontSize': 14,
            },
        }
    }

# Register and enable the custom theme
alt.themes.register('my_custom_theme', my_custom_theme)
alt.themes.enable('my_custom_theme')


#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
 
}

[data-testid="stMetric"] {
    background-color: #DBE2F0;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
  color: #4B4C4E;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

class model(nn.Module):
    def __init__(self, checkpoint, freeze=False, device='cpu'):
        super().__init__()
        
        self.model = AutoModel.from_pretrained(checkpoint)
        hidden_sz = self.model.config.hidden_size
        # set device cuda or cpu
        self.device = device
        # freeze model
        if freeze:
            for layer in self.model.parameters():
                layer.requires_grad=False
        
    def forward(self, x, attention_mask=None):
        x = x.to(self.device)
        # pooler_output(seq,dim) 
        with torch.no_grad():
            model_out = self.model(x['input_ids'], x['attention_mask'], return_dict=True)
            
        embds = model_out.last_hidden_state # model_out[0][:,0]
        mean_pool = embds.sum(axis=1)/ x['attention_mask'].sum(axis=1).unsqueeze(axis=1)
        return mean_pool
    
@st.cache_data 
def predict_sentiments(sentences):
    #model load
    print(" -------------- Load Sentiment Model ---------")
    model_checkpoint = "indolem/indobert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    model_path = hf_hub_download(
        repo_id="zqomalita/sentimen-model",
        filename="sentiment_model.pth",
        repo_type="model")
    
    print("-----------Model Loaded. Start Prediction------------")
    # Load saved model
    model = BertClassifier(num_labels=3)
    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    model.to(device)
    st.write(f"Memory Usage Before Prediction: {psutil.virtual_memory().percent}%")
    
    model.eval()  
    max_length = 256 

    # Tokenize input
    encodings = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )

    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)

    # inference
    with torch.no_grad():
        outputs = model(input_ids, attention_mask) 
        probabilities = F.softmax(outputs, dim=1)  
        predicted_classes = torch.argmax(probabilities, dim=1)  

    # Map class 
    sentiment_mapping = {1: "positif", 0: "netral", 2: "negatif"}
    predicted_labels = [sentiment_mapping[class_idx.item()] for class_idx in predicted_classes]

    # list score
    scores = probabilities.cpu().numpy()

    return scores,predicted_labels


class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        # model load
        model_checkpoint = "indolem/indobert-base-uncased"
        self.bert = BertModel.from_pretrained(model_checkpoint)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 300),
            nn.ReLU(),
            nn.Linear(300, 150),
            nn.ReLU(),
            nn.Linear(150, 50),
            nn.ReLU(),
            nn.Linear(50, num_labels)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = outputs['last_hidden_state'][:, 0, :]
        x = self.classifier(x)
        return x


def get_topik(kata):
    print("------------Load Model----------")
    checkpoint = 'indolem/indobertweet-base-uncased'
    indobert = model(checkpoint, freeze=True)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    model_path = hf_hub_download(
        repo_id="zqomalita/kelas-topik",
        filename="model_topik_new.pkl",
        repo_type="model")
    
    model_load = joblib.load(model_path)
    print("Model path:", model_path)

    print("-------------Model Loaded. Start Prediction --------------")
    # dataloader
    final_embeddings =list()
    all_embeddings = []
    final_sentences = kata

    batch_sz = 200 # batch_size
    for idx in range(0, len(final_sentences), batch_sz):
        batch_sentences = final_sentences[idx:idx+batch_sz]
        for sent in batch_sentences:
            tokens = tokenizer(sent ,truncation='longest_first', return_tensors='pt', return_attention_mask=True, padding=True)
            embeddings = indobert(tokens)
            final_embeddings.extend(embeddings)
            all_embeddings = torch.stack(final_embeddings)

    return model_load.predict(pd.DataFrame(all_embeddings))
    

def clean_text(text):
    
    text = re.sub(r'\d+', '', text)
    # Menghapus spasi ekstra
    text = re.sub(r'\s+', ' ', text)
    # Menghapus mention
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  # removing mentions
    # Menghapus hashtag
    text = re.sub(r'#[A-Za-z0-9]+', '', text)  # removing hastag
    text = re.sub(r'RT[\s]+', '', text)  # removing RT
    text = re.sub(r"http\S+", '', text)  # removing link
    text = re.sub(r"\b[a-zA-Z]\b", "", text)

    text = re.sub(r'[^A-Za-z]+', ' ', text)  # removing all character non alphabet

    text = text.replace('\n', ' ')  # replace new line into space
  
    #text = GoogleTranslator(target='id').translate(text)
    text = text.lower() 
    text = text.strip(' ')
    
    #text = re.sub(r'http\S+', '', text)  # Remove URL
    #text = re.sub(r'@\w+', '', text)  # Remove mention
    #text = re.sub(r'#\w+', '', text)  # Remove hashtags
    #teks = re.sub(r'[^A-Za-z\s]', '', text)  # Remove special character dan angka
    #teks = teks.lower()  # Convert ke lowercase
    
    return text
def simple_tokenize(text):
    return text.split()
    
# tokenize, remove stop words
def preprocess_text_sastrawi(text):
    factory = StopWordRemoverFactory()
    stopword_sastrawi = factory.get_stop_words()
    try:
        if not isinstance(text, str):
            return ''
        # Lowercase
        text = text.lower()
        # Hapus URL
        text = re.sub(r"http\S+|www.\S+", "", text)
        # Hapus mention, hashtag, angka, simbol
        text = re.sub(r"@\w+|#\w+|\d+|[^\w\s]", "", text)
        # Tokenisasi sederhana
        tokens = simple_tokenize(text)
    
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocess_text_sastrawi: {e}")
        return ''
        
@st.cache_data
def load_slang_dict(_engine):
    df_slang = pd.read_csv("https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv")
    query = text(f"""SELECT sundanese, indonesian FROM word""")

    with _engine.connect() as conn:
        df_word = pd.read_sql(query, conn)

    df_word.columns = ["slang","formal"]
    df_slang_all = pd.concat([df_slang[["slang","formal"]], df_word])

    return dict(zip(df_slang_all["slang"], df_slang_all["formal"]))

# non-formal ke formal
def replace_slang_word(text, slang_word):
    try:
        if not isinstance(text, str):
            return ''
        words = text.split()
        print("------- word in replace slang",words)

        replaced_words = [slang_word.get(word, word) for word in words]

        return ' '.join(replaced_words)
    except Exception as e:
        print(f"Error in replace_slang_word: {e}")
        return ''


@st.cache_data
def get_tweets(hari,_engine):
    n_days_ago = (datetime.now() - timedelta(days=hari)).date()
    # --- Query dengan filter tanggal ---
    query = text(f"""SELECT * FROM twitter_post WHERE date >= :start_date""")

    with _engine.connect() as conn:
        df_twitter = pd.read_sql(query, conn, params={"start_date": n_days_ago})

    df_twitter.columns = ["id","created_at", "tanggal_post", "username", "full_text", "favorite_count", "reply_count","retweet_count"]
    df_twitter.drop(columns=["id"], axis=1,inplace=True)
    return df_twitter

@st.cache_data
def get_instapost(hari,_engine):
    n_days_ago = (datetime.now() - timedelta(days=hari)).date()

    # --- Query dengan filter tanggal ---
    query = text(f"""SELECT * FROM instagram_caption WHERE tanggal_post >= :start_date""")

    with _engine.connect() as conn:
        df_insta = pd.read_sql(query, conn, params={"start_date": n_days_ago})
    
    print(df_insta)
    df_insta.columns = ["post_id","tanggal_post", "waktu_post", "nama_akun", "caption", "jumlah_like", "jumlah_komentar"]
    df_insta.drop(columns=["post_id"], axis=1,inplace=True)

    return df_insta

@st.cache_data
def get_instacomment(hari,_engine):
    n_days_ago = (datetime.now() - timedelta(days=hari)).date()

    # --- Query dengan filter tanggal ---
    query = text(f"""SELECT * FROM instagram_comment WHERE tanggal_post >= :start_date""")

    with _engine.connect() as conn:
        df_insta = pd.read_sql(query, conn, params={"start_date": n_days_ago}) 

    df_insta.columns = ["id","tanggal_post","waktu_post","post_caption","nama_akun","username","full_text","tanggal_komentar", "post_id"]
    df_insta.drop(columns=["id","post_id"], axis=1,inplace=True)
    df_insta["full_text"] = df_insta["full_text"].replace({pd.NA: "", None: "", np.nan: ""})
    df_insta["full_text"] = df_insta["full_text"].astype(str)

    return df_insta

def make_wordcloud(text_cloud):
    mask = np.array(Image.open('mask kota bandung.jpg'))
    wordcloud = WordCloud(width=600, height=400, max_words=250, colormap='twilight', collocations=True, contour_width=1, mask=mask,contour_color='grey', background_color='white').generate(text_cloud)
    
    fig, ax = plt.subplots()
    print(wordcloud)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

def remove_stopword(words):
    factory = StopWordRemoverFactory()
    stopword_sastrawi = factory.get_stop_words()
    stopword_sastrawi.extend(["yg",'yang','biar','ada','enggak', "dg", "rt", "dgn", "ny", "d", 'kl', 'klo','kalau',
                              'kalo', 'amp', 'biar', 'bikin', 'bilang', 'jika','akan','selalu','aku','ke','di','saling',
                              'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 'untuk','tidak','tak','tbtb', 'wkwk','wkwkwk','wkwkwkwk','mulu',
                              'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 'usah','euy','kang','teh','teteh','akang','mbak','mas','om','tante','bapak','ibu',
                              'jd', 'jgn', 'sdh', 'aja', 'n', 't', 'ya','lho','lo','ajg','anjay','ajng'
                              'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt','aku','kamu','dia','mereka','kalian','&amp', 'yah'])
    return [word for word in words if word not in stopword_sastrawi]



def main():
    engine = create_engine(f"mysql+pymysql://opendata:Dataconf2023!@103.108.190.91/db_socialmedia")

    with st.sidebar:
        st.title('\U0001F310 Social Media Analysis')
        
        sosmed = ["Twitter","Instagram"]
        selected_sosmed = st.selectbox('Media Sosial', sosmed , index = len(sosmed)-1)
        hari = st.text_input("Analisis dalam n hari, contoh: 7")
        print("Jumlah Hari : ", hari)

        analyze = st.button('Analyze')

        
    placeholder = st.empty()
    if analyze:
        with placeholder:
            st.write("Processing...")
            print(selected_sosmed)
            if selected_sosmed=="Twitter":
                file = get_tweets(hari = int(hari), _engine=engine)
                tweet_ori = file['full_text']
                teks_metriks = 'Tweet'
                nama_kolom = ['Text','Topik','Sentimen','Tanggal Posting','Teks','Jumlah Like','Jumlah Reply','Jumlah Retweet']

            if selected_sosmed=="Instagram":
                file_post = get_instapost(int(hari), _engine=engine)
                file = get_instacomment(int(hari), _engine=engine)
                print(file_post)
                print(file)
                komen_ori = file['full_text']
                teks_metriks = 'Post'
                nama_kolom = ["Text","Topik","Sentimen","Tanggal Post","Waktu Post","Post Caption","Nama Akun","Username","Komentar","Tanggal Komentar"]
                print("------------ Data Loaded --------------")

            # Preprocessing Data
            
            file['full_text'] = file['full_text'].apply(lambda x: clean_text(x))
            processed_text = file['full_text'].apply(lambda x: preprocess_text_sastrawi(x))
            print(processed_text)

            # Text Normalization / Noise Removal
            slang_dict = load_slang_dict(engine)
            print(slang_dict)
            final_text = processed_text.apply(lambda x : replace_slang_word(x,slang_dict))
            clean_data = final_text
            print("---------------- Finish Cleaning --------------------")
            print(clean_data)

            print(" -------------- Analytics Process --------- ")
            st.write("Get Topics...")
            prediksi = get_topik(clean_data)
                
            # Klasifikasikan
            hasil = pd.Series(prediksi).map({1:'ekonomi', 2:'teknologi', 3:'hukum', 4:'kesehatan', 5:'politik', 6:'hiburan', 7:'infrastruktur dan lingkungan', 8:'pendidikan', 9:'lainnya' })
            topik = pd.concat([clean_data, hasil.to_frame()], axis=1)
            print(topik)

            df_class = topik
            df_class.columns = ["teks","topik"]
            
            
            word =  df_class['teks'].apply(simple_tokenize)
            print(word)

            # Wordcloud
            st.write("Get Wordcloud...")
            word_v = word.apply(remove_stopword)
            combined_list = list(chain(*word_v))
            text_cloud = " ".join(combined_list)
            placeholder.empty()

        col = st.columns((1, 4.5, 2.5), gap='medium')
        with col[0]:
            if selected_sosmed=="Twitter":
                st.markdown("#### Jumlah/Total")
                st.metric(teks_metriks, len(file))
                st.metric("User", file['username'].nunique())
                file['created_at'] = pd.to_datetime(file['created_at'], format="%a %b %d %H:%M:%S %z %Y").dt.date
                file_post = file
            if selected_sosmed=="Instagram":
                st.markdown("#### Jumlah/Total")
                st.metric(teks_metriks, len(file_post))
                st.metric("Akun", file_post['nama_akun'].nunique())
                st.metric("Komentar", pd.to_numeric(file_post['jumlah_komentar'],errors='coerce').sum())
                #file['created_at'] =file['created_at'].dt.date

            # Hitung jumlah hari unik
            unique_days_count = file_post['tanggal_post'].nunique()
            st.metric("Hari", unique_days_count)

        with col[1]:
            st.markdown("#### Pembicaraan Populer")
            make_wordcloud(text_cloud)
            print(text_cloud)

        with col[2]:
            st.markdown('#### Top Topik')
            df_class_count = df_class['topik'].value_counts()
            print(df_class_count)
            # Convert ke dataframe
            topik_counts_df = df_class_count.reset_index()
            topik_counts_df.columns = ['topik', 'jumlah']
            topik_counts_df.sort_values(by='jumlah', ascending=False).reset_index(drop=True)

            st.dataframe(topik_counts_df,
                        column_order=("topik", "jumlah"),
                        hide_index=True,
                        width=None,
                        column_config={
                            "topik": st.column_config.TextColumn(
                                "Topik",
                            ),
                            "jumlah": st.column_config.ProgressColumn(
                                "Jumlah",
                                format="%f",
                                min_value=0,
                                max_value=max(topik_counts_df.jumlah),
                            )}
                        )
                
        
        col2 =  st.columns(1)
        with col2[0]:
            st.markdown('#### Proporsi Sentimen')
            results = predict_sentiments(clean_data.to_list())
            print(results)
            
            polarity_score = pd.DataFrame(results[0], columns=['positif', 'netral', 'negatif'])
            polarity = results[1]
            print(polarity_score)
            print(pd.DataFrame(polarity).value_counts())

            vcount =  pd.DataFrame(polarity).value_counts().reset_index()
            vcount.columns=['polarity','count']

            fig = px.pie(vcount, values='count',height=300, width=200, names='polarity',color="polarity", color_discrete_map={"negatif":"#F36B59",
                                 "netral":"#8CA7C0",
                                 "positif":"#27AE60"})
            fig.update_layout(margin=dict(l=10, r=20, t=30, b=0),)
            st.plotly_chart(fig, use_container_width=True)
            
        col3 =  st.columns(1)
        with col3[0]:
            df_concat = pd.concat([df_class,pd.DataFrame(polarity)], axis=1)
            df_concat.columns = ['teks','topik','polaritas']
            df_concat.drop(columns=['teks'], inplace=True)
            print(df_concat)

            df_concat['topik2'] = df_concat['topik']
            df_concat = df_concat[df_concat['topik'] != 'pendidikan']
            grouped_df = df_concat.groupby(['topik', 'polaritas']).size().reset_index(name='jumlah').fillna(0)

            print(grouped_df)

            st.markdown("#### Sentimen Berdasar Topik")
            c = alt.Chart(grouped_df).mark_bar(size=20).encode(
                x=alt.X('polaritas:N', axis=None),
                y=alt.Y('jumlah:Q',  axis=alt.Axis(grid=True)),
                color=alt.Color('polaritas:N').legend(orient="bottom").scale(domain=['negatif','netral','positif'], range=['#F36B59','#8CA7C0','#27AE60']), 
                column=alt.Column('topik:O'),
                ).configure_header(labelOrient='bottom').configure_scale(bandPaddingInner=0,bandPaddingOuter=0.1,)
            st.altair_chart(c)
                                
        col4 = st.columns(1)    
        with col4[0]:
            if selected_sosmed=="Twitter":
                df_output = pd.concat([df_class,pd.DataFrame(polarity),file['tanggal_post'],tweet_ori,file['favorite_count'], file['reply_count'],file['retweet_count']], axis=1)
                df_output.dropna(subset=['full_text'])
            if selected_sosmed=="Instagram":
                df_output = pd.concat([df_class,pd.DataFrame(polarity),file['tanggal_post'],file['waktu_post'],file['post_caption'], file['nama_akun'],file['username'],file['full_text'],file['tanggal_komentar']],axis=1)
                df_output.dropna(subset=['full_text'])

            df_output.columns = nama_kolom
            df_output.drop(columns=['Text'],inplace=True)
            st.dataframe(df_output, hide_index=True, use_container_width=True)
            
    
if __name__ == "__main__":
    main()
