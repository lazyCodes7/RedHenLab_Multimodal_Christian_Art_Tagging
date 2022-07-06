import urllib.request
def save_image_links(df, location):
    for i in range(len(df)):
        link = df.iloc[i]['IMAGE_LINKS']
        name = df.iloc[i]['IMAGE_NAME']
        urllib.request.urlretrieve(link, location + name)