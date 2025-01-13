import os
import shutil
from bs4 import BeautifulSoup

# add metadata when building the docker image

# Streamlit package directory
streamlit_path = "/usr/local/lib/python3.12/site-packages/streamlit/static/index.html"

# Backup original index.html
shutil.copy2(streamlit_path, streamlit_path + ".bak")

# Read original index.html
with open(streamlit_path, "r") as file:
    html_content = file.read()

# Parse html
soup = BeautifulSoup(html_content, "html.parser")

meta_tags = [
    # title and description
    {
        "name": "title",
        "content": "Real-Time Personal YouTube Recommender Powered by LLMs + WNNs - a preview by AO Labs",
    },
    {
        "name": "description",
        "content": "AO Labs is building a more reliable alternative to deep learning and LLMs using continuously trainable, compute-efficient weightless neural networks.",
    },
    # Open Graph / Facebook
    {"property": "og:type", "content": "website"},
    {"property": "og:url", "content": "https://metatags.io/"},
    {
        "property": "og:title",
        "content": "Real-Time Personal YouTube Recommender Powered by LLMs + WNNs - a preview by AO Labs",
    },
    {
        "property": "og:description",
        "content": "AO Labs is building a more reliable alternative to deep learning and LLMs using continuously trainable, compute-efficient weightless neural networks.",
    },
    {"property": "og:image", "content": "https://raw.githubusercontent.com/aolabsai/Recommender/refs/heads/main/misc/aolabs-logo-horizontal-full-color-white-text.png"},
    # Twitter
    {"property": "twitter:card", "content": "summary_large_image"},
    {"property": "twitter:url", "content": "https://metatags.io/"},
    {
        "property": "twitter:title",
        "content": "Real-Time Personal YouTube Recommender Powered by LLMs + WNNs - a preview by AO Labs",
    },
    {
        "property": "twitter:description",
        "content": "AO Labs is building a more reliable alternative to deep learning and LLMs using continuously trainable, compute-efficient weightless neural networks.",
    },
    {"property": "twitter:image", "content": "https://raw.githubusercontent.com/aolabsai/Recommender/refs/heads/main/misc/aolabs-logo-horizontal-full-color-white-text.png"},
]

# add meta tags to head
for tag in meta_tags:
    new_tag = soup.new_tag("meta")
    for key, value in tag.items():
        new_tag[key] = value
    soup.head.append(new_tag)

# save modified html file
with open(streamlit_path, "w") as file:
    file.write(str(soup))

print("Meta tags have been added to the streamlit index.html")
