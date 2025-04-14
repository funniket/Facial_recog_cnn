from facial_recog_cnn.pipelines.data_download import download
from facial_recog_cnn.pipelines.extracting_data import extract_data

print("<------Data Download------->")
download()
print("<------Data Downlaoded-------->")

print("<------Extracting Data------->")
extract_data()
print("<------Data Extraction Complete-------->")